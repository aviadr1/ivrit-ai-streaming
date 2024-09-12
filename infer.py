import base64
import faster_whisper
import tempfile
import torch
import time
import requests
import logging
from fastapi import FastAPI, HTTPException, WebSocket,WebSocketDisconnect
import websockets
from pydantic import BaseModel
from typing import Optional
import sys
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',handlers=[logging.StreamHandler(sys.stdout)])

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


class InputData(BaseModel):
    type: str
    data: Optional[str] = None  # Used for blob input
    url: Optional[str] = None  # Used for url input


def download_file(url, max_size_bytes, output_filename, api_key=None):
    """
    Download a file from a given URL with size limit and optional API key.
    """
    logging.debug(f'Starting file download from URL: {url}')
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            logging.debug('API key provided, added to headers')

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        file_size = int(response.headers.get('Content-Length', 0))
        logging.info(f'File size: {file_size} bytes')

        if file_size > max_size_bytes:
            logging.error(f'File size exceeds limit: {file_size} > {max_size_bytes}')
            return False

        downloaded_size = 0
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                logging.debug(f'Downloaded {downloaded_size} bytes')
                if downloaded_size > max_size_bytes:
                    logging.error('Downloaded size exceeds maximum allowed payload size')
                    return False
                file.write(chunk)

        logging.info(f'File downloaded successfully: {output_filename}')
        return True

    except requests.RequestException as e:
        logging.error(f"Error downloading file: {e}")
        return False

@app.get("/")
async def read_root():
    return {"message": "This is the Ivrit AI Streaming service."}


@app.post("/transcribe")
async def transcribe(input_data: InputData):
    logging.info(f'Received transcription request with data: {input_data}')
    datatype = input_data.type
    if not datatype:
        logging.error('datatype field not provided')
        raise HTTPException(status_code=400, detail="datatype field not provided. Should be 'blob' or 'url'.")

    if datatype not in ['blob', 'url']:
        logging.error(f'Invalid datatype: {datatype}')
        raise HTTPException(status_code=400, detail=f"datatype should be 'blob' or 'url', but is {datatype} instead.")

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'
        logging.debug(f'Created temporary directory: {d}')

        if datatype == 'blob':
            if not input_data.data:
                logging.error("Missing 'data' for 'blob' input")
                raise HTTPException(status_code=400, detail="Missing 'data' for 'blob' input.")
            logging.info('Decoding base64 blob data')
            mp3_bytes = base64.b64decode(input_data.data)
            open(audio_file, 'wb').write(mp3_bytes)
            logging.info(f'Audio file written: {audio_file}')
        elif datatype == 'url':
            if not input_data.url:
                logging.error("Missing 'url' for 'url' input")
                raise HTTPException(status_code=400, detail="Missing 'url' for 'url' input.")
            logging.info(f'Downloading file from URL: {input_data.url}')
            success = download_file(input_data.url, MAX_PAYLOAD_SIZE, audio_file, None)
            if not success:
                logging.error(f"Error downloading data from {input_data.url}")
                raise HTTPException(status_code=400, detail=f"Error downloading data from {input_data.url}")

        result = transcribe_core(audio_file)
        return {"result": result}


def transcribe_core(audio_file):
    logging.info('Starting transcription...')
    ret = {'segments': []}

    segs, _ = model.transcribe(audio_file, language='he', word_timestamps=True)
    logging.info('Transcription completed')

    for s in segs:
        words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]
        seg = {
            'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
            'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words
        }
        logging.info(f'Transcription segment: {seg}')
        ret['segments'].append(seg)

    return ret


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
        words= []
        logging.info(f"Processing segment with start time: {s.start} and end time: {s.end}")

        # Only process segments that start after the last transcribed time
        if s.start >= last_transcribed_time:
            logging.info(f"New segment found starting at {s.start} seconds.")
            for w in words:
                words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})
            seg = {
                'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text,
                'avg_logprob': s.avg_logprob, 'compression_ratio': s.compression_ratio,
                'no_speech_prob': s.no_speech_prob, 'words': words
            }
            logging.info(f'Adding new transcription segment: {seg}')
            ret['new_segments'].append(seg)

            # Update the last transcribed time to the end of the current segment
            new_last_transcribed_time = s.end
            logging.debug(f"Updated last transcribed time to: {new_last_transcribed_time} seconds")

    logging.info(f"Returning {len(ret['new_segments'])} new segments and updated last transcribed time.")
    return ret, new_last_transcribed_time


import tempfile


@app.websocket("/wtranscribe")
async def websocket_transcribe(websocket: WebSocket):
    logging.info("New WebSocket connection request received.")
    await websocket.accept()
    logging.info("WebSocket connection established successfully.")

    try:
        processed_segments = []  # Keeps track of the segments already transcribed
        last_transcribed_time = 0.0

        # A temporary file to store the growing audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            logging.info(f"Temporary audio file created at {temp_audio_file.name}")

            while True:
                try:
                    # Receive the next chunk of audio data
                    audio_chunk = await websocket.receive_bytes()
                    if not audio_chunk:
                        logging.warning("Received empty audio chunk, skipping processing.")
                        continue

                    # Write audio chunk to file and accumulate size and time
                    temp_audio_file.write(audio_chunk)
                    temp_audio_file.flush()

                    # Call the transcription function with the last processed time
                    partial_result, last_transcribed_time = transcribe_core_ws(temp_audio_file.name, last_transcribed_time)
                    accumulated_audio_time = 0  # Reset the accumulated audio time

                    response = {
                        "new_segments": partial_result['new_segments'],
                        "processed_segments": processed_segments
                    }
                    logging.info(f"Sending {len(partial_result['new_segments'])} new segments to the client.")
                    processed_segments.extend(partial_result['new_segments'])
                    await websocket.send_json(response)

                except WebSocketDisconnect:
                    logging.info("WebSocket connection closed by the client.")
                    break

    except Exception as e:
        logging.error(f"Unexpected error during WebSocket transcription: {e}")
        await websocket.send_json({"error": str(e)})

    finally:
        logging.info("Cleaning up and closing WebSocket connection.")



