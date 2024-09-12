import asyncio
import websockets
import wave
import requests

# Parameters for reading and sending the audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192  # Size of the audio chunk sent at a time
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod_serverless_whisper/main/me-hebrew.wav"  # Path to the mp3 file


async def send_audio(websocket):
    buffer_size = 1024 * 1024  # Buffer 1MB of audio data before sending for transcription
    audio_buffer = bytearray()  # Collect audio chunks directly in memory

    # Stream the audio file in real-time
    with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
        if response.status_code == 200:
            print("Starting to stream audio file...")

            for chunk in response.iter_content(chunk_size=8192):  # Stream in chunks of 8192 bytes
                if chunk:
                    # Append each chunk to the in-memory buffer
                    audio_buffer.extend(chunk)
                    print(f"Received audio chunk of size {len(chunk)} bytes.")

                    # Once we have buffered enough audio data, send it for transcription
                    if len(audio_buffer) >= buffer_size:
                        await websocket.send(audio_buffer)  # Send buffered data directly
                        print(f"Sent {len(audio_buffer)} bytes of audio data to the server for transcription.")
                        audio_buffer.clear()  # Clear buffer after sending
                        await asyncio.sleep(0.01)  # Simulate real-time streaming

            print("Finished sending audio.")
        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")

async def receive_transcription(websocket):
    while True:
        try:
            transcription = await websocket.recv()
            print(f"Received transcription: {transcription}")
        except Exception as e:
            print(f"Error receiving transcription: {e}")
            break

async def receive_transcription(websocket):
    while True:
        try:
            transcription = await websocket.recv()  # Receive transcription from the server
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error: {e}")
            break

import ssl
async def run_client():
    uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/ws/transcribe")  # Replace with your Hugging Face Space WebSocket URL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(uri, ssl=ssl_context, timeout=30) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_transcription(websocket)
        )

asyncio.run(run_client())
