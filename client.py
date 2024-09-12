import asyncio
import websockets
import requests
import ssl

# Parameters for reading and sending the audio
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod_serverless_whisper/main/me-hebrew.wav"  # Use WAV file

async def send_audio(websocket):
    # Stream the audio file in real-time
    with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
        if response.status_code == 200:
            print("Starting to stream audio file...")

            for chunk in response.iter_content(chunk_size=8192):  # Stream in chunks of 8192 bytes
                if chunk:
                    await websocket.send(chunk)  # Send each chunk over WebSocket
                    print(f"Sent audio chunk of size {len(chunk)} bytes")

            print("Finished sending audio.")
        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")

async def receive_transcription(websocket):
    while True:
        try:
            transcription = await websocket.recv()  # Receive transcription from the server
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error receiving transcription: {e}")
            break

async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.ping()
            print("Sent keepalive ping")
        except websockets.ConnectionClosed:
            print("Connection closed, stopping heartbeat")
            break
        await asyncio.sleep(30)  # Send ping every 30 seconds (adjust as needed)


async def run_client():
    uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/ws/transcribe")  # WebSocket URL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(uri, ssl=ssl_context, timeout=30) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_transcription(websocket),
            send_heartbeat(websocket)
        )

asyncio.run(run_client())
