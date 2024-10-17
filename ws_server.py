# Import the necessary components from whisper_online.py
import argparse
import asyncio
import io
import logging
import os
import sys
import time

# Define WebSocket endpoint
import uuid
from functools import wraps
from typing import Optional

import librosa
import numpy as np
import requests
import soundfile
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketDisconnect

from libs.whisper_streaming.whisper_online import (  # add_shared_args,
    ASRBase,
    OnlineASRProcessor,
    add_shared_args,
    asr_factory,
    load_audio_chunk,
)


def my_set_logging(args, logger, other="_server"):
    logging.basicConfig(format="%(levelname)s\t%(message)s")  # format='%(name)s
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


logging.basicConfig(format="%(levelname)s\t%(message)s", level=logging.INFO)  # format='%(name)s
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisper_online").setLevel(logging.INFO)


SAMPLING_RATE = 16000
WARMUP_FILE = "mono16k.test_hebrew.wav"
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"

app = FastAPI()
args = argparse.ArgumentParser()


def check_fp16_support():
    """
    Checks whether FP16 (half precision) is supported on the available CUDA device by
    examining the device's compute capability and logs relevant information about the
    PyTorch and CUDA versions.

    FP16 (half precision) is supported if:
      - The GPU's compute capability is >= 5.3
      - CUDA is available

    Returns:
      - A log with the PyTorch version, CUDA version, device information, and whether FP16 is supported.

    Example Output (FP16 Supported):
    -------------------------------
    PyTorch version: 1.12.0
    CUDA version: 11.6
    Device selected: cuda
    Device: NVIDIA Tesla V100-SXM2-16GB
    Compute Capability: (7, 0)
    FP16 support: Yes (Compute capability >= 5.3)

    Example Output (FP16 Not Supported):
    ------------------------------------
    PyTorch version: 1.12.0
    CUDA version: 11.6
    Device selected: cuda
    Device: NVIDIA GeForce GTX 750
    Compute Capability: (5, 0)
    FP16 support: No (Compute capability < 5.3)

    Example Output (No CUDA Device):
    --------------------------------
    PyTorch version: 1.12.0
    CUDA version: None
    Device selected: cpu
    No CUDA device found. FP16 support: Not available.
    """

    # Log the PyTorch version
    logging.info(f"PyTorch version: {torch.__version__}")

    # Log the installed CUDA version, this will be None if CUDA is not available
    logging.info(f"CUDA version: {torch.version.cuda}")

    # Determine if CUDA is available, otherwise default to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device selected: {device}")

    # Check if CUDA is available and inspect the device capabilities
    if torch.cuda.is_available():
        # Get the name of the CUDA device (e.g., "NVIDIA Tesla V100")
        device_name = torch.cuda.get_device_name(0)

        # Get the compute capability of the device (e.g., (7, 0) for Volta GPUs)
        compute_capability = torch.cuda.get_device_capability(0)

        # Log the device name and compute capability
        logging.info(f"Device: {device_name}")
        logging.info(f"Compute Capability: {compute_capability}")

        # Compute capability >= 5.3 indicates FP16 support for most GPUs
        if compute_capability >= (5, 3):
            logging.info("FP16 support: Yes (Compute capability >= 5.3)")
        else:
            logging.info("FP16 support: No (Compute capability < 5.3)")
    else:
        # If no CUDA device is available, log that FP16 is not supported
        logging.info("No CUDA device found. FP16 support: Not available.")


def drop_option_from_parser(parser, option_name):
    """
    Reinitializes the parser and copies all options except the specified option.

    Args:
        parser (argparse.ArgumentParser): The original argument parser.
        option_name (str): The option string to drop (e.g., '--model').

    Returns:
        argparse.ArgumentParser: A new parser without the specified option.
    """
    # Create a new parser with the same description and other attributes
    new_parser = argparse.ArgumentParser(
        description=parser.description, epilog=parser.epilog, formatter_class=parser.formatter_class
    )

    # Iterate through all the arguments of the original parser
    for action in parser._actions:
        if "-h" in action.option_strings:
            continue

        # Check if the option is not the one to drop
        if option_name not in action.option_strings:
            new_parser._add_action(action)

    return new_parser


def convert_to_mono_16k(input_wav: str, output_wav: str) -> None:
    """
    Converts any .wav file to mono 16 kHz.

    Args:
        input_wav (str): Path to the input .wav file.
        output_wav (str): Path to save the output .wav file with mono 16 kHz.
    """
    # Step 1: Load the audio file with librosa
    audio_data, original_sr = librosa.load(input_wav, sr=None, mono=False)  # Load at original sampling rate
    logger.info("Loaded audio with shape: %s, original sampling rate: %d" % (audio_data.shape, original_sr))

    # Step 2: If the audio has multiple channels, average them to make it mono
    if audio_data.ndim > 1:
        audio_data = librosa.to_mono(audio_data)

    # Step 3: Resample the audio to 16 kHz
    resampled_audio = librosa.resample(audio_data, orig_sr=original_sr, target_sr=SAMPLING_RATE)

    # Step 4: Save the resampled audio as a .wav file in mono at 16 kHz
    soundfile.write(output_wav, resampled_audio, SAMPLING_RATE)

    logger.info(f"Converted audio saved to {output_wav}")


def download_warmup_file():
    # Download the audio file if not already present
    audio_file_path = "test_hebrew.wav"
    if not os.path.exists(WARMUP_FILE):
        if not os.path.exists(audio_file_path):
            response = requests.get(AUDIO_FILE_URL)
            with open(audio_file_path, "wb") as f:
                f.write(response.content)

        convert_to_mono_16k(audio_file_path, WARMUP_FILE)


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    websocket: WebSocket
    asr: ASRBase
    online: OnlineASRProcessor
    min_limit: int
    read_task: Optional[asyncio.Task] = None

    is_first: bool = True
    last_end: Optional[float] = None
    wav_file: Optional[soundfile.SoundFile] = None


async def receive_audio_chunk(state: State) -> Optional[np.ndarray]:
    """
    Receives audio chunks from the WebSocket connection and processes them.

    Keeps the last read task open to improve performance.
    """
    # Initialize the buffer for audio chunks
    out = []

    while True:
        try:
            # Create the read task if not already created or completed
            if not state.read_task or state.read_task.done():
                state.read_task = asyncio.create_task(state.websocket.receive_bytes())

            current_size = sum(len(chunk) for chunk in out)
            # Wait for the read task with the timeout
            done, _ = await asyncio.wait([state.read_task], timeout=0, return_when=asyncio.FIRST_COMPLETED)

            # If the task is done, process the chunk
            if done:
                chunk = state.read_task.result()  # Get the result of the task

                # Reset the task so it will be created again next time
                state.read_task = None

                if chunk:
                    # Process the received chunk
                    audio_chunk = np.frombuffer(chunk, dtype="<i2")  # Little-endian 16-bit PCM
                    out.append(audio_chunk)

                    if current_size >= 4 * state.min_limit:
                        break
                else:
                    # If no chunk received, and enough data has been collected, stop
                    if current_size >= state.min_limit:
                        break
                    else:
                        await asyncio.sleep(0.5)

        except asyncio.TimeoutError:
            # Timeout without receiving, exit the loop if enough data was gathered
            if current_size >= state.min_limit:
                break
        except WebSocketDisconnect:
            # Handle WebSocket disconnection gracefully
            logger.info("WebSocket connection closed by the client.")
            break
        except Exception as e:
            # Log any unexpected errors
            logger.error(f"Error receiving audio chunk: {e}")
            break

    # If no audio chunks were received, return None
    if not out:
        return None

    # Concatenate the received audio chunks
    concatenated_audio = np.concatenate(out)

    # Use soundfile to handle audio processing
    sf = soundfile.SoundFile(
        io.BytesIO(concatenated_audio),
        channels=1,
        endian="LITTLE",
        samplerate=SAMPLING_RATE,
        subtype="PCM_16",
        format="RAW",
    )

    # Load the audio with librosa for further processing
    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)

    # Save the received audio to the .wav file if enabled
    if state.wav_file is not None:
        state.wav_file.buffer_write(concatenated_audio, dtype="int16")

    # If this is the first chunk and it's too short, skip processing
    if state.is_first and len(audio) < state.min_limit:
        logger.error("First chunk is too short, skipping")
        return None

    # Mark that the first chunk has been processed
    state.is_first = False

    return audio


def format_output_transcript(state, o) -> dict:
    # output format in stdout is like:
    # 0 1720 Takhle to je
    # - the first two words are:
    #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript

    # This function differs from whisper_online.output_transcript in the following:
    # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
    # Therefore, beg, is max of previous end and current beg outputed by Whisper.
    # Usually it differs negligibly, by appx 20 ms.

    if o[0] is not None:
        beg, end = o[0] * 1000, o[1] * 1000
        if state.last_end is not None:
            beg = max(beg, state.last_end)

        state.last_end = end
        print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
        return {
            "start": "%1.0f" % beg,
            "end": "%1.0f" % end,
            "text": "%s" % o[2],
        }
    else:
        logger.debug("No text in this segment")
        return None


def perf(func):
    """
    A decorator to measure and log the execution time of a function.

    Args:
        func: The function to decorate.

    Returns:
        Wrapped function with added performance logging.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start the timer
        start_time = time.perf_counter()

        # Execute the original function
        result = func(*args, **kwargs)

        # Stop the timer
        end_time = time.perf_counter()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Log the performance
        logging.info(f"{func.__name__} took {elapsed_time:.6f} seconds.")

        # Return the result of the original function
        return result

    return wrapper


def get_next_processing_duration(asr_processor: OnlineASRProcessor) -> float:
    """
    Returns the number of fractional seconds of audio that will be processed by the
    given OnlineASRProcessor when `process_iter` is run.

    Args:
        asr_processor (OnlineASRProcessor): The ASR processor instance to inspect.

    Returns:
        float: The number of seconds of audio that will be passed for transcription.
    """
    # Length of the audio buffer in samples
    buffer_length_in_samples = len(asr_processor.audio_buffer)

    # Convert samples to seconds using the sampling rate
    buffer_length_in_seconds = buffer_length_in_samples / asr_processor.SAMPLING_RATE

    return buffer_length_in_seconds


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, save_audio: bool = True):
    logger.info("New WebSocket connection request received.")
    await websocket.accept()
    logger.info("WebSocket connection established successfully.")

    parsed_args = args.parse_args(
        [
            "--lan",
            "he",
            "--model",
            "ivrit-ai/faster-whisper-v2-d4",
            # '--model', 'ivrit-ai/whisper-large-v3-turbo-d4-p1-take2',
            "--backend",
            "faster-whisper",
            "--vad",
            "--vac",
            "-l",
            "DEBUG",
            # "--buffer_trimming", "sentence",
            "--buffer_trimming_sec",
            "8",
            "--min-chunk-size",
            "1",
        ]
    )

    # Optionally create a unique .wav file for debugging
    if save_audio:
        wav_filename = f"received_audio_{uuid.uuid4()}.wav"
        logger.info(f"Saving received audio to file: {wav_filename}")
        wav_file = soundfile.SoundFile(
            wav_filename, mode="w", samplerate=SAMPLING_RATE, channels=1, subtype="PCM_16", endian="LITTLE"
        )

    def init_model():
        # initialize the ASR model
        logger.info("Loading whisper model...")
        asr, online = asr_factory(parsed_args)
        state = State(
            websocket=websocket,
            asr=asr,
            online=online,
            min_limit=parsed_args.min_chunk_size * SAMPLING_RATE,
            wav_file=wav_file if save_audio else None,
        )

        logger.info("Warming up the whisper model...")
        a = load_audio_chunk(WARMUP_FILE, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
        return state

    state = await asyncio.to_thread(init_model)

    @perf
    def process(audio):
        state.online.insert_audio_chunk(audio)
        logger.info(f"Processing audio chunk..., now having {get_next_processing_duration(state.online):.2f} seconds")
        o = state.online.process_iter()
        logger.info(f"After processing, buffer of {get_next_processing_duration(state.online):.2f} seconds left")
        return o

    try:
        while True:
            a = await receive_audio_chunk(state)  # Pass wav_file to save audio
            if a is None:
                break

            try:
                o = await asyncio.to_thread(process, a)
                if result := format_output_transcript(state, o):
                    await websocket.send_json(result)

            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed by the client.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during WebSocket transcription: {e}")
                await websocket.send_json({"error": str(e)})
    finally:
        logger.info("Cleaning up and closing WebSocket connection.")
        if save_audio:
            wav_file.close()  # Close the wav file


def main():
    global args
    add_shared_args(args)

    args = drop_option_from_parser(args, "--model")
    args.add_argument(
        "--model",
        type=str,
        help="Name size of the Whisper model to use. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )


check_fp16_support()
main()  # ran by fastapi/uvicorn cli

if __name__ == "__main__":
    uvicorn.run(app, port=4400)
