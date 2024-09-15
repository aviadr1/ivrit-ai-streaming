import asyncio
import websockets
import requests
import ssl

# Parameters for reading and sending the audio
#AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/hugging_face_ivrit_streaming/main/long_hebrew.wav"


async def send_audio(websocket):
    buffer_size = 512 * 1024  #HAVE TO HAVE 512!!
    audio_buffer = bytearray()

    with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
        if response.status_code == 200:
            print("Starting to stream audio file...")

            for chunk in response.iter_content(chunk_size=512):  # Stream in chunks
                if chunk:
                    audio_buffer.extend(chunk)
                    #print(f"Received audio chunk of size {len(chunk)} bytes.")

                    # Send buffered audio data once it's large enough
                    if len(audio_buffer) >= buffer_size:
                        await websocket.send(audio_buffer)
                        #print(f"Sent {len(audio_buffer)} bytes of audio data.")
                        audio_buffer.clear()
                        await asyncio.sleep(0.001)

            print("Finished sending audio.")
        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")


async def receive_transcription(websocket):
    while True:
        try:

            transcription = await asyncio.wait_for(websocket.recv(), timeout=300)
            # Receive transcription from the server
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error receiving transcription: {e}")
            await asyncio.sleep(30)
            break


async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.ping()
            print("Sent keepalive ping")
        except websockets.ConnectionClosed:
            print("Connection closed, stopping heartbeat")
            break
        await asyncio.sleep(3)  # Send ping every 30 seconds (adjust as needed)


async def run_client():
    uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/wtranscribe")  # WebSocket URL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    while True:
        try:
            async with websockets.connect(uri, ssl=ssl_context, ping_timeout=500, ping_interval=20) as websocket:
                await asyncio.gather(
                    send_audio(websocket),
                    receive_transcription(websocket),
                    #send_heartbeat(websocket)
                )
        except websockets.ConnectionClosedError as e:
            print(f"WebSocket closed with error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print("Reconnecting in 5 seconds...")
        await asyncio.sleep(5)  # Wait 5 seconds before reconnecting

if __name__ == "__main__":
    asyncio.run(run_client())
