import argparse
import json
import asyncio
from pathlib import Path
from typing import List
import websockets
import os
import librosa
import numpy as np

# Define the default WebSocket endpoint
DEFAULT_WS_URL = "ws://localhost:4400/ws"


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
    parser.add_argument("--chunk_duration", type=float, default=1.0, help="Duration of each audio chunk in seconds.")
    return parser.parse_args()


def read_audio_in_chunks(audio_file, target_sr=16000, chunk_duration=1) -> List[np.ndarray]:
    """
    Reads a 16kHz mono audio file in 1-second chunks and returns them as little-endian 16-bit integer arrays.
    """
    if not str(audio_file).endswith(".wav"):
        wav_file = Path(audio_file).with_suffix(".wav")
        if not wav_file.exists():
            command = f'ffmpeg -i "{audio_file}" -ac 1 -ar {target_sr} "{wav_file}"'
            print(f"Converting MP3 to WAV: {command}")
            os.system(command)
        audio_file = wav_file

    audio_data, sr = librosa.load(audio_file, sr=None, mono=True)

    if sr != target_sr:
        raise ValueError(f"Unexpected sample rate {sr}. Expected {target_sr}.")

    audio_data_int16 = (audio_data * 32767).astype(np.int16)

    if audio_data_int16.dtype.byteorder == '>' or (
            audio_data_int16.dtype.byteorder == '=' and np.dtype(np.int16).byteorder == '>'):
        print("Byte swap performed to convert to little-endian.")
        audio_data_little_endian = audio_data_int16.byteswap().newbyteorder('L')
    else:
        print("No byte swap needed. Already little-endian.")
        audio_data_little_endian = audio_data_int16

    samples_per_chunk = target_sr * chunk_duration
    chunks = [
        audio_data_little_endian[i:i + samples_per_chunk]
        for i in range(0, len(audio_data_little_endian), samples_per_chunk)
    ]

    return chunks


async def send_audio_chunks(ws, audio_chunks):
    """
    Asynchronously send audio chunks to the WebSocket server.
    """
    for idx, chunk in enumerate(audio_chunks):
        audio_bytes = chunk.astype('<f4').tobytes()  # Convert to little-endian float32
        await ws.send(audio_bytes)
        print(f"Sent chunk {idx + 1}/{len(audio_chunks)}")
        await asyncio.sleep(0.1)  # Simulate real-time streaming
    print("All audio chunks sent")


async def websocket_client(args):
    """
    Asynchronously connect to the WebSocket server, send audio chunks, and receive transcriptions.
    """
    audio_chunks = read_audio_in_chunks(args.audio_file)

    async with websockets.connect(args.url) as ws:
        print("WebSocket connection opened")

        # Start sending audio chunks
        send_task = asyncio.create_task(send_audio_chunks(ws, audio_chunks))

        # Receive transcriptions asynchronously
        try:
            async for message in ws:
                data = json.loads(message)
                if args.response_format == "verbose_json":
                    segments = data.get('segments', [])
                    for segment in segments:
                        print(f"Received segment: {segment['text']}")
                else:
                    print(f"Transcription: {data['text']}")
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed normally")
        except Exception as e:
            print(f"Error receiving message: {e}")

        await send_task  # Ensure all chunks are sent before closing


async def run_websocket_client(args):
    """
    Entry point to run the WebSocket client using asyncio.
    """
    loop = asyncio.get_event_loop()
    loop.run_until_complete(websocket_client(args))


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(websocket_client(args))
