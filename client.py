import asyncio
import websockets
import requests
import ssl

# Parameters for reading and sending the audio
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file

async def send_audio(websocket):
    buffer_size = 1024*512  # Buffer audio chunks up to 512KB before sending
    audio_buffer = bytearray()

    with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
        if response.status_code == 200:
            print("Starting to stream audio file...")

            for chunk in response.iter_content(chunk_size=1024):  # Stream in chunks
                if chunk:
                    audio_buffer.extend(chunk)
                    print(f"Received audio chunk of size {len(chunk)} bytes.")

                    # Send buffered audio data once it's large enough
                    if len(audio_buffer) >= buffer_size:
                        await websocket.send(audio_buffer)
                        print(f"Sent {len(audio_buffer)} bytes of audio data.")
                        audio_buffer.clear()
                        await asyncio.sleep(0.01)

            print("Finished sending audio.")
        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")

async def receive_transcription(websocket):
    while True:
        try:
            transcription = await websocket.recv()  # Receive transcription from the server
            print(f"Transcription: {transcription}")
            download_url = transcription.get('download_url')
            if download_url:
                print(f"Download URL: {download_url}")
                # Download the audio file
                response = requests.get(download_url)
                if response.status_code == 200:
                    with open("downloaded_audio.wav", "wb") as f:
                        f.write(response.content)
                    print("File downloaded successfully")
                else:
                    print(f"Failed to download file. Status code: {response.status_code}")
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
    uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/wtranscribe")  # WebSocket URL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(uri, ssl=ssl_context, timeout=60) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_transcription(websocket),
            send_heartbeat(websocket)
        )

asyncio.run(run_client())