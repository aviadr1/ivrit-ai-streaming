import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import soundfile
import websockets.asyncio.client

# Define the default WebSocket endpoint
DEFAULT_WS_URL = "ws://localhost:8000/ws"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stream audio to the transcription WebSocket endpoint.")
    parser.add_argument("audio_file", help="Path to the input audio file.")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint URL.")
    parser.add_argument("--model", type=str, help="Model name to use for transcription.")
    parser.add_argument("--language", type=str, help="Language code for transcription.")
    parser.add_argument(
        "--response_format",
        type=str,
        default="verbose_json",
        choices=["text", "json", "verbose_json"],
        help="Response format.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for transcription.")
    parser.add_argument("--vad_filter", action="store_true", help="Enable voice activity detection filter.")
    parser.add_argument("--chunk_duration", type=float, default=0.2, help="Duration of each audio chunk in seconds.")
    return parser.parse_args()


def read_audio_in_chunks(audio_file, target_sr=16000, chunk_duration=1) -> List[np.ndarray]:
    """
    Reads a 16kHz mono audio file in 1-second chunks and returns them as little-endian
    16-bit integer arrays.
    """
    if not str(audio_file).endswith(".wav"):
        wav_file = Path(audio_file).with_suffix(".wav")
        if not wav_file.exists():
            command = f'ffmpeg -i "{audio_file}" -ac 1 -ar {target_sr} "{wav_file}"'
            print(f"Converting MP3 to WAV: {command}")
            os.system(command)
        audio_file = wav_file

    # Read audio using soundfile
    with soundfile.SoundFile(audio_file) as f:
        if f.samplerate != target_sr:
            raise ValueError(f"Unexpected sample rate {f.samplerate}. Expected {target_sr}.")

        # Read the entire audio file as an array
        audio_data = f.read(dtype="int16")

    # Calculate the number of samples per chunk
    samples_per_chunk = int(math.ceil(target_sr * chunk_duration))

    # Split the audio into chunks
    chunks = [audio_data[i : i + samples_per_chunk] for i in range(0, len(audio_data), samples_per_chunk)]

    return chunks


async def send_audio_chunks(ws, audio_chunks):
    """
    Asynchronously send audio chunks to the WebSocket server in real-time, ensuring no
    more than 10 seconds of audio is sent faster than real time.

    Args:
        ws: The WebSocket connection.
        audio_chunks: The list of audio chunks to send.
        chunk_duration: The duration of each chunk in seconds.
    """
    start_time = time.time()  # Record the start time
    total_sent_duration = 0  # Track total duration of audio sent
    realtime_gap = 4  # Maximum gap between real-time and audio sent
    for idx, chunk in enumerate(audio_chunks):
        # Send the current audio chunk
        await ws.send(chunk.tobytes())
        chunk_duration = len(chunk) / 16000  # Calculate the duration of the chunk in seconds
        total_sent_duration += chunk_duration  # Update the total duration of audio sent
        print(f"Sent chunk {idx + 1}/{len(audio_chunks)}, total audio duration sent: {total_sent_duration:.2f}s")

        # Check if we've sent more than 10 seconds of audio in less time
        elapsed_real_time = time.time() - start_time
        if total_sent_duration - elapsed_real_time > realtime_gap:
            # Wait for real time to catch up before sending more
            sleep_time = total_sent_duration - elapsed_real_time - realtime_gap
            print(f"Pausing for {sleep_time:.2f}s to match real-time streaming...")
            await asyncio.sleep(sleep_time)

        # Simulate sending in real-time (wait for the duration of the chunk)
        await asyncio.sleep(chunk_duration * 0.5)

    print("All audio chunks sent")


async def websocket_client(args):
    """
    Asynchronously connect to the WebSocket server, send audio chunks, and receive
    transcriptions.
    """
    audio_chunks = read_audio_in_chunks(args.audio_file, chunk_duration=args.chunk_duration)

    async with websockets.asyncio.client.connect(args.url, ping_interval=1, ping_timeout=30) as ws:
        print("WebSocket connection opened")

        # Start sending audio chunks
        send_task = asyncio.create_task(send_audio_chunks(ws, audio_chunks))

        # Receive transcriptions asynchronously
        try:
            async for message in ws:
                data = json.loads(message)
                print(f"Transcription: {data}")
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed normally")
        except Exception as e:
            print(f"Error receiving message: {e}")

        await send_task  # Ensure all chunks are sent before closing


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(websocket_client(args))
