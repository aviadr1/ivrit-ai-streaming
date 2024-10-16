# Import the necessary components from whisper_online.py
import logging
import os

import librosa
import soundfile
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from libs.whisper_streaming.whisper_online import (
    ASRBase,
    OnlineASRProcessor,
    VACOnlineASRProcessor,
    add_shared_args,
    asr_factory,
    set_logging,
    create_tokenizer,
    load_audio,
    load_audio_chunk, OpenaiApiASR,
    set_logging
)

import argparse
import sys
import numpy as np
import io
import soundfile as sf
import wave
import requests
import argparse

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
WARMUP_FILE = "mono16k.test_hebrew.wav"
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"

is_first = True
asr, online = None, None
min_limit = None  # min_chunk*SAMPLING_RATE
app = FastAPI()


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
    sf.write(output_wav, resampled_audio, SAMPLING_RATE)

    logger.info(f"Converted audio saved to {output_wav}")

def download_warmup_file():
    # Download the audio file if not already present
    audio_file_path = "test_hebrew.wav"
    if not os.path.exists(WARMUP_FILE):
        if not os.path.exists(audio_file_path):
            response = requests.get(AUDIO_FILE_URL)
            with open(audio_file_path, 'wb') as f:
                f.write(response.content)

        convert_to_mono_16k(audio_file_path, WARMUP_FILE)


async def receive_audio_chunk(self, websocket: WebSocket):
    # receive all audio that is available by this time
    # blocks operation if less than self.min_chunk seconds is available
    # unblocks if connection is closed or a chunk is available
    out = []
    while sum(len(x) for x in out) < min_limit:
        raw_bytes = await websocket.receive_bytes()
        if not raw_bytes:
            break

        sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
        audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
        out.append(audio)

    if not out:
        return None

    conc = np.concatenate(out)
    if self.is_first and len(conc) < min_limit:
        return None
    self.is_first = False
    return conc

# Define WebSocket endpoint
@app.websocket("/ws_transcribe_streaming")
async def websocket_transcribe(websocket: WebSocket):
    logger.info("New WebSocket connection request received.")
    await websocket.accept()
    logger.info("WebSocket connection established successfully.")

    asr, online = asr_factory(args)

    # warm up the ASR because the very first transcribe takes more time than the others.
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    a = load_audio_chunk(WARMUP_FILE, 0, 1)
    asr.transcribe(a)
    logger.info("Whisper is warmed up.")
    global min_limit
    min_limit = args.min_chunk_size * SAMPLING_RATE

    try:
        out = []
        while True:
            try:
                # Receive JSON data
                raw_bytes = await websocket.receive_json()

                sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE,
                                         subtype="PCM_16", format="RAW")
                audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
                out.append(audio)

                # Call the transcribe function
                # segments, info = await asyncio.to_thread(model.transcribe,
                segments, info = model.transcribe(
                    audio_file_path,
                    language='he',
                    initial_prompt=input_data.init_prompt,
                    beam_size=5,
                    word_timestamps=True,
                    condition_on_previous_text=True
                )

                # Convert segments to list and serialize
                segments_list = list(segments)
                segments_serializable = [segment_to_dict(s) for s in segments_list]
                logger.info(get_raw_words_from_segments(segments_list))
                # Send the serialized segments back to the client
                await websocket.send_json(segments_serializable)

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed by the client.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during WebSocket transcription: {e}")
                await websocket.send_json({"error": str(e)})
    finally:
        logger.info("Cleaning up and closing WebSocket connection.")

def main():
    args = argparse.ArgumentParser()
    args = add_shared_args(args)
    args.parse_args([
        '--lan', 'he',
        '--model', 'ivrit-ai/faster-whisper-v2-d4',
        '--backend', 'faster-whisper',
        '--vad',
        # '--vac', '--buffer_trimming', 'segment', '--buffer_trimming_sec', '15', '--min_chunk_size', '1.0', '--vac_chunk_size', '0.04', '--start_at', '0.0', '--offline', '--comp_unaware', '--log_level', 'DEBUG'
    ])


    global asr, online


    uvicorn.run(app)