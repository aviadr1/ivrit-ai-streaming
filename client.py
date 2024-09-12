import asyncio
import websockets
import wave

# Parameters for reading and sending the audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Size of the audio chunk sent at a time
AUDIO_FILE = "https://raw.githubusercontent.com/AshDavid12/hugging_face_ivrit_streaming/main/test_copy.mp3"  # Path to the mp3 file


async def send_audio(websocket):
    with wave.open(AUDIO_FILE, "rb") as wf:
        data = wf.readframes(CHUNK_SIZE)
        while data:
            await websocket.send(data)  # Send audio chunk to the server
            await asyncio.sleep(CHUNK_SIZE / SAMPLE_RATE)  # Simulate real-time by waiting for the duration of the chunk
            data = wf.readframes(CHUNK_SIZE)


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

    async with websockets.connect(uri, ssl=ssl_context) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_transcription(websocket)
        )

asyncio.run(run_client())
