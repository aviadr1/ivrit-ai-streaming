import asyncio
import json
import wave

import websockets
import requests
import ssl

# Parameters for reading and sending the audio
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file

async def send_audio(websocket):
    buffer_size = 1024 * 16  # Send smaller chunks (16KB) for real-time processing

    # Download the WAV file locally
    # with requests.get(AUDIO_FILE_URL, stream=True) as response:
    #     if response.status_code == 200:
    #         with open('downloaded_audio.wav', 'wb') as f:
    #             for chunk in response.iter_content(chunk_size=1024):
    #                 f.write(chunk)
    #         print("Audio file downloaded successfully.")

            # Open the downloaded WAV file and extract PCM data
            with wave.open('test_copy.wav', 'rb') as wav_file:
                metadata = {
                    'sample_rate': wav_file.getframerate(),
                    'channels': wav_file.getnchannels(),
                    'sampwidth': wav_file.getsampwidth(),
                }

                # Send metadata to the server before sending the audio
                await websocket.send(json.dumps(metadata))
                print(f"Sent metadata: {metadata}")

                # Send the PCM audio data in chunks
                while True:
                    pcm_chunk = wav_file.readframes(buffer_size)
                    if not pcm_chunk:
                        break  # End of file

                    await websocket.send(pcm_chunk)  # Send raw PCM data chunk
                    #print(f"Sent PCM chunk of size {len(pcm_chunk)} bytes.")
                    await asyncio.sleep(0.01)  # Simulate real-time sending

        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")


async def receive_transcription(websocket):
    while True:
        try:
            transcription = await websocket.recv()  # Receive transcription from the server
            print(f"Transcription: {transcription}")
            transcription = json.loads(transcription)
            #download_url = transcription.get('download_url')
            # if download_url:
            #     print(f"Download URL: {download_url}")
            #     # Download the audio file
            #     response = requests.get(download_url)
            #     if response.status_code == 200:
            #         with open("downloaded_audio.wav", "wb") as f:
            #             f.write(response.content)
            #         print("File downloaded successfully")
            #     else:
            #         print(f"Failed to download file. Status code: {response.status_code}")
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