import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List
import os
import asyncio
import websockets
from fastapi import FastAPI, WebSocket
import uvicorn
import threading
import time

app = FastAPI()

SAMPLING_RATE = 16000
CHUNK_DURATION = 1  # In seconds
CHUNK_SIZE = SAMPLING_RATE * CHUNK_DURATION  # Samples per chunk


### 1. Read Audio in Chunks ###
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

    # Read audio using soundfile
    with sf.SoundFile(audio_file) as f:
        if f.samplerate != target_sr:
            raise ValueError(f"Unexpected sample rate {f.samplerate}. Expected {target_sr}.")

        # Read the entire audio file as an array
        audio_data = f.read(dtype='int16')

    # Calculate the number of samples per chunk
    samples_per_chunk = target_sr * chunk_duration

    # Split the audio into chunks
    chunks = [
        audio_data[i:i + samples_per_chunk]
        for i in range(0, len(audio_data), samples_per_chunk)
    ]

    return chunks


### 2. Reassemble Chunks into WAV ###
def reassemble_chunks_to_wav(chunks: List[np.ndarray], output_wav: str, target_sr=16000):
    """
    Reassembles audio chunks into a .wav file using soundfile with little-endian 16-bit integer format.

    Args:
        chunks (List[np.ndarray]): List of audio chunks (arrays).
        output_wav (str): Path to the output .wav file.
        target_sr (int): Target sample rate (default is 16000 Hz).
    """
    # Concatenate the chunks back into a single array
    full_audio = np.concatenate(chunks)

    # Write the reassembled audio back to a .wav file with little-endian 16-bit PCM format
    sf.write(output_wav, full_audio, target_sr, subtype='PCM_16')

    print(f"Reassembled audio saved to {output_wav}")


### 3. WebSocket Server Endpoint ###
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    received_chunks = []

    try:
        while True:
            # Receive audio chunk as binary data
            chunk = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(chunk, dtype='<i2')  # Little-endian 16-bit PCM
            received_chunks.append(audio_chunk)
            print(f"Received chunk of size: {len(audio_chunk)}")

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Reassemble and save the audio when the connection is closed
        output_wav = "received_output.wav"
        reassemble_chunks_to_wav(received_chunks, output_wav)
        print(f"WebSocket connection closed. Audio saved as {output_wav}")


### 4. WebSocket Client to Send Audio Chunks ###
async def websocket_client(audio_chunks: List[np.ndarray], uri: str):
    async with websockets.connect(uri) as websocket:
        for idx, chunk in enumerate(audio_chunks):
            await websocket.send(chunk.tobytes())
            print(f"Sent chunk {idx + 1}/{len(audio_chunks)}")
            await asyncio.sleep(0.1)  # Simulate real-time streaming
        print("All chunks sent, closing connection.")
        await websocket.close()


### 5. Run FastAPI Server and WebSocket Client in Same Process ###
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_client():
    # Read audio and split it into chunks
    chunks = read_audio_in_chunks('lex.wav')
    # Run the WebSocket client to send the chunks
    asyncio.run(websocket_client(chunks, "ws://localhost:8000/ws"))


if __name__ == "__main__":
    # Run FastAPI server in a separate thread
    print('starting the server')
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server some time to start up
    time.sleep(2)

    # Run the WebSocket client to send audio chunks
    print('starting the client')
    run_client()

    # Wait for the server thread to finish
    print('shutting down')
    server_thread.join()
