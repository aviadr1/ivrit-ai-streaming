import base64
import os

import faster_whisper
import tempfile

import numpy as np
import torch
import time
import requests
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import websockets
from pydantic import BaseModel
from typing import Optional
import sys
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)
#logging.getLogger("asyncio").setLevel(logging.DEBUG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Device selected: {device}')

model_name = 'ivrit-ai/faster-whisper-v2-d4'
logging.info(f'Loading model: {model_name}')
model = faster_whisper.WhisperModel(model_name, device=device)
logging.info('Model loaded successfully')

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024
logging.info(f'Max payload size set to: {MAX_PAYLOAD_SIZE} bytes')

app = FastAPI()


# class InputData(BaseModel):
#     type: str
#     data: Optional[str] = None  # Used for blob input
#     url: Optional[str] = None  # Used for url input
#
#
# def download_file(url, max_size_bytes, output_filename, api_key=None):
#     """
#     Download a file from a given URL with size limit and optional API key.
#     """
#     logging.debug(f'Starting file download from URL: {url}')
#     try:
#         headers = {}
#         if api_key:
#             headers['Authorization'] = f'Bearer {api_key}'
#             logging.debug('API key provided, added to headers')
#
#         response = requests.get(url, stream=True, headers=headers)
#         response.raise_for_status()
#
#         file_size = int(response.headers.get('Content-Length', 0))
#         logging.info(f'File size: {file_size} bytes')
#
#         if file_size > max_size_bytes:
#             logging.error(f'File size exceeds limit: {file_size} > {max_size_bytes}')
#             return False
#
#         downloaded_size = 0
#         with open(output_filename, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 downloaded_size += len(chunk)
#                 logging.debug(f'Downloaded {downloaded_size} bytes')
#                 if downloaded_size > max_size_bytes:
#                     logging.error('Downloaded size exceeds maximum allowed payload size')
#                     return False
#                 file.write(chunk)
#
#         logging.info(f'File downloaded successfully: {output_filename}')
#         return True
#
#     except requests.RequestException as e:
#         logging.error(f"Error downloading file: {e}")
#         return False


@app.get("/")
async def read_root():
    return {"message": "This is the Ivrit AI Streaming service."}


# async def transcribe_core_ws(audio_file):
#     ret = {'segments': []}
#
#     try:
#
#         logging.debug(f"Initiating model transcription for file: {audio_file}")
#
#         segs, _ = await asyncio.to_thread(model.transcribe, audio_file, language='he', word_timestamps=True)
#         logging.info('Transcription completed successfully.')
#     except Exception as e:
#         logging.error(f"Error during transcription: {e}")
#         raise e
#
#     # Track the new segments and update the last transcribed time
#     for s in segs:
#         logging.info(f"Processing segment with start time: {s.start} and end time: {s.end}")
#
#         # Only process segments that start after the last transcribed time
#         logging.info(f"New segment found starting at {s.start} seconds.")
#         words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]
#
#         seg = {
#             'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text,
#             'avg_logprob': s.avg_logprob, 'compression_ratio': s.compression_ratio,
#             'no_speech_prob': s.no_speech_prob, 'words': words
#         }
#         logging.info(f'Adding new transcription segment: {seg}')
#         ret['segements'].append(seg)
#
#         # Update the last transcribed time to the end of the current segment
#
#
# #logging.info(f"Returning {len(ret['new_segments'])} new segments and updated last transcribed time.")
#     return ret


import tempfile



def transcribe_core_ws(audio_file, last_transcribed_time):
    """
    Transcribe the audio file and return only the segments that have not been processed yet.

    :param audio_file: Path to the growing audio file.
    :param last_transcribed_time: The last time (in seconds) that was transcribed.
    :return: Newly transcribed segments and the updated last transcribed time.
    """
    logging.info(f"Starting transcription for file: {audio_file} from {last_transcribed_time} seconds.")

    ret = {'new_segments': []}
    new_last_transcribed_time = last_transcribed_time

    try:
        # Transcribe the entire audio file
        logging.debug(f"Initiating model transcription for file: {audio_file}")
        segs, _ = model.transcribe(audio_file, language='he', word_timestamps=True)
        logging.info('Transcription completed successfully.')
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise e

    # Track the new segments and update the last transcribed time
    for s in segs:
        logging.info(f"Processing segment with start time: {s.start} and end time: {s.end}")

        # Only process segments that start after the last transcribed time
        if s.start >= last_transcribed_time:
            logging.info(f"New segment found starting at {s.start} seconds.")
            words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]

            seg = {
                'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text,
                'avg_logprob': s.avg_logprob, 'compression_ratio': s.compression_ratio,
                'no_speech_prob': s.no_speech_prob, 'words': words
            }
            logging.info(f'Adding new transcription segment: {seg}')
            ret['new_segments'].append(seg)

            # Update the last transcribed time to the end of the current segment
            new_last_transcribed_time = max(new_last_transcribed_time, s.end)
            logging.debug(f"Updated last transcribed time to: {new_last_transcribed_time} seconds")

    #logging.info(f"Returning {len(ret['new_segments'])} new segments and updated last transcribed time.")
    return ret, new_last_transcribed_time


import tempfile


@app.websocket("/wtranscribe")
async def websocket_transcribe(websocket: WebSocket):
    logging.info("New WebSocket connection request received.")
    await websocket.accept()
    logging.info("WebSocket connection established successfully.")

    try:
        processed_segments = []  # Keeps track of the segments already transcribed
        accumulated_audio_size = 0  # Track how much audio data has been buffered
        accumulated_audio_time = 0  # Track the total audio duration accumulated
        last_transcribed_time = 0.0
        #min_transcription_time = 5.0  # Minimum duration of audio in seconds before transcription starts

        # A temporary file to store the growing audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            logging.info(f"Temporary audio file created at {temp_audio_file.name}")
            #temp_audio_filename = os.path.basename(temp_audio_file.name)
            output_directory = "/tmp"
            os.makedirs(output_directory, exist_ok=True)

            while True:
                try:
                    # Receive the next chunk of audio data
                    audio_chunk = await websocket.receive_bytes()
                    if not audio_chunk:
                        logging.warning("Received empty audio chunk, skipping processing hey.")
                        continue

                    chunk_counter = 0
                    # Create a new file for the chunk
                    chunk_filename = os.path.join(output_directory, f"audio_chunk_{chunk_counter}.wav")
                    chunk_counter += 1
                    with open(chunk_filename, 'wb') as audio_file:
                        audio_file.write(audio_chunk)
                    # Write audio chunk to file and accumulate size and time
                    temp_audio_file.write(audio_chunk)
                    temp_audio_file.flush()
                    accumulated_audio_size += len(audio_chunk)

                    # Estimate the duration of the chunk based on its size (e.g., 16kHz audio)
                    chunk_duration = len(audio_chunk) / (16000 * 2)  # Assuming 16kHz mono WAV (2 bytes per sample)
                    accumulated_audio_time += chunk_duration
                    logging.info(f"Received and buffered {len(audio_chunk)} bytes, total buffered: {accumulated_audio_size} bytes, total time: {accumulated_audio_time:.2f} seconds")

                    # Transcribe when enough time (audio) is accumulated (e.g., at least 5 seconds of audio)
                    #if accumulated_audio_time >= min_transcription_time:
                    #logging.info("Buffered enough audio time, starting transcription.")


                    # Call the transcription function with the last processed time
                    partial_result, last_transcribed_time = transcribe_core_ws(temp_audio_file.name, last_transcribed_time)
                    accumulated_audio_time = 0  # Reset the accumulated audio time
                    processed_segments.extend(partial_result['new_segments'])

                    # Reset the accumulated audio size after transcription
                    accumulated_audio_size = 0

                    # Send the transcription result back to the client with both new and all processed segments
                    response = {
                        "new_segments": partial_result['new_segments'],
                        "processed_segments": processed_segments,
                        "download_url": f"https://gigaverse-ivrit-ai-streaming.hf.space/download_audio/{os.path.basename(chunk_filename)})"
                    }
                    logging.info(f"Sending {len(partial_result['new_segments'])} new segments to the client.")
                    await websocket.send_json(response)

                except WebSocketDisconnect:
                    logging.info("WebSocket connection closed by the client.")
                    break

    except Exception as e:
        logging.error(f"Unexpected error during WebSocket transcription: {e}")
        await websocket.send_json({"error": str(e)})

    finally:
        logging.info("Cleaning up and closing WebSocket connection.")


from fastapi.responses import FileResponse


@app.get("/download_audio/{filename}")
async def download_audio(filename: str):
    file_path = f"/tmp/{filename}"

    # Ensure the file exists before serving it
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    else:
        return {"error": "File not found"}

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """WebSocket endpoint to handle client connections."""
#     await websocket.accept()
#     client_ip = websocket.client.host
#     logger.info(f"Client connected: {client_ip}")
#     sys.stdout.flush()
#     try:
#         await process_audio_stream(websocket)
#     except WebSocketDisconnect:
#         logger.info(f"Client disconnected: {client_ip}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         await websocket.close()
#
# async def process_audio_stream(websocket: WebSocket):
#     """Continuously receive audio chunks and initiate transcription tasks."""
#     sampling_rate = 16000
#     min_chunk_size = 5  # in seconds
#
#     transcription_task = None
#     chunk_counter = 0
#     total_bytes_received = 0
#
#     while True:
#         try:
#             # Receive audio data from client
#             data = await websocket.receive_bytes()
#             if not data:
#                 logger.info("No data received, closing connection")
#                 break
#             chunk_counter += 1
#             chunk_size = len(data)
#             total_bytes_received += chunk_size
#             #logger.debug(f"Received chunk {chunk_counter}: {chunk_size} bytes")
#
#             audio_chunk = process_received_audio(data)
#             #logger.debug(f"Processed audio chunk {chunk_counter}: {len(audio_chunk)} samples")
#             # Check if enough audio has been buffered
#             # if transcription_task is None or transcription_task.done():
#             #     # Start a new transcription task
#         #     # logger.info(f"Starting transcription task for {len(audio_buffer)} samples")
#             transcription_task = asyncio.create_task(
#                 transcribe_and_send(websocket, audio_chunk)
#             )
#
#             #logger.debug(f"Audio buffer size: {len(audio_buffer)} samples")
#         except Exception as e:
#             logger.error(f"Error receiving data: {e}")
#             break
#
#
# async def transcribe_and_send(websocket: WebSocket, audio_data):
#     """Run transcription in a separate thread and send the result to the client."""
#     logger.debug(f"Transcription task started for {len(audio_data)} samples")
#     transcription_result = await asyncio.to_thread(sync_transcribe_audio, audio_data)
#     if transcription_result:
#         try:
#             # Send the result as JSON
#             await websocket.send_json(transcription_result)
#             logger.info(f"Transcription JSON sent to client {transcription_result}")
#         except Exception as e:
#             logger.error(f"Error sending transcription: {e}")
#     else:
#         logger.warning("No transcription result to send")
#
# def sync_transcribe_audio(audio_data):
#     """Synchronously transcribe audio data using the ASR model and format the result."""
#     try:
#
#         logger.info('Starting transcription...')
#         segments, info = model.transcribe(
#             audio_data, language="he",compression_ratio_threshold=2.5, word_timestamps=True
#         )
#         logger.info('Transcription completed')
#
#         # Build the transcription result as per your requirement
#         ret = {'segments': []}
#
#         for s in segments:
#             logger.debug(f"Processing segment {s.id} with start time: {s.start} and end time: {s.end}")
#
#             # Process words in the segment
#             words = [{
#                 'start': float(w.start),
#                 'end': float(w.end),
#                 'word': w.word,
#                 'probability': float(w.probability)
#             } for w in s.words]
#
#             seg = {
#                 'id': int(s.id),
#                 'seek': int(s.seek),
#                 'start': float(s.start),
#                 'end': float(s.end),
#                 'text': s.text,
#                 'avg_logprob': float(s.avg_logprob),
#                 'compression_ratio': float(s.compression_ratio),
#                 'no_speech_prob': float(s.no_speech_prob),
#                 'words': words
#             }
#             logger.debug(f'Adding new transcription segment: {seg}')
#             ret['segments'].append(seg)
#
#             logger.debug(f"Total segments in transcription result: {len(ret['segments'])}")
#             return ret
#     except Exception as e:
#         logger.error(f"Transcription error: {e}")
#         return {}
#
# def process_received_audio(data):
#     """Convert received bytes into normalized float32 NumPy array."""
#     #logger.debug(f"Processing received audio data of size {len(data)} bytes")
#     audio_int16 = np.frombuffer(data, dtype=np.int16)
#     #logger.debug(f"Converted to int16 NumPy array with {len(audio_int16)} samples")
#
#     audio_float32 = audio_int16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
#     #logger.debug(f"Normalized audio data to float32 with {len(audio_float32)} samples")
#
#     return audio_float32
#
#








# @app.websocket("/wtranscribe")
# async def websocket_transcribe(websocket: WebSocket):
#     logging.info("New WebSocket connection request received.")
#     await websocket.accept()
#     logging.info("WebSocket connection established successfully.")
#
#     try:
#         while True:
#             try:
#                 audio_chunk = await websocket.receive_bytes()
#                 if not audio_chunk:
#                     logging.warning("Received empty audio chunk, skipping processing.")
#                     continue
#                 with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file: ##new temp file for every chunk
#                     logging.info(f"Temporary audio file created at {temp_audio_file.name}")
#                     # Receive the next chunk of audio data
#
#
#
#                     partial_result = await transcribe_core_ws(temp_audio_file.name)
#                     await websocket.send_json(partial_result)
#
#             except WebSocketDisconnect:
#                 logging.info("WebSocket connection closed by the client.")
#                 break
#
#     except Exception as e:
#         logging.error(f"Unexpected error during WebSocket transcription: {e}")
#         await websocket.send_json({"error": str(e)})
#
#     finally:
#         logging.info("Cleaning up and closing WebSocket connection.")
