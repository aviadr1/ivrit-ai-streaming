import asyncio
import io

import numpy as np
import websockets
import requests
import ssl
import wave
import logging
import sys

# Parameters for reading and sending the audio
#AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/hugging_face_ivrit_streaming/main/long_hebrew.wav"



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

async def send_receive():
    uri = "wss://gigaverse-ivrit-ai-streaming.hf.space/ws"  # Update with your server's address if needed
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    logger.info(f"Connecting to server at {uri}")
    try:
        async with websockets.connect(uri,ssl=ssl_context) as websocket:
            logger.info("WebSocket connection established")
            # Start tasks for sending and receiving
            send_task = asyncio.create_task(send_audio(websocket))
            receive_task = asyncio.create_task(receive_transcriptions(websocket))
            await asyncio.gather(send_task, receive_task)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

async def send_audio(websocket):
    wav_file = AUDIO_FILE_URL  # Replace with the path to your WAV file
    logger.info(f"Opening WAV file: {wav_file}")

    try:
        # Download the WAV file
        response = requests.get(wav_file)
        response.raise_for_status()
        wav_bytes = io.BytesIO(response.content)


        # Send audio data in chunks directly from the WAV file
        chunk_size = 1024  # Sending data in chunks of 3200 bytes, which can be adjusted

        total_chunks = 0
        total_bytes_sent = 0

        # While loop to send audio data chunk by chunk
        while True:
            chunk = wav_bytes.read(chunk_size)
            if not chunk:
                break
            await websocket.send(chunk)
            total_chunks += 1
            total_bytes_sent += len(chunk)
            #logger.debug(f"Sent chunk {total_chunks}: {len(chunk)} bytes")
            #await asyncio.sleep(0.1)  # Simulate real-time streamin
            #logger.info(f"Finished sending audio data: {total_chunks} chunks sent, total bytes sent: {total_bytes_sent}")

    except Exception as e:
        logger.error(f"Send audio error: {e}")

    finally:
        logger.info("WAV file closed")

async def receive_transcriptions(websocket):
    try:
        logger.info("Starting to receive transcriptions")
        async for message in websocket:  # This is the same as websocket.recv()
            logger.info(f"Received transcription: {message}")
            print(f"Transcription: {message}")
    except Exception as e:
        logger.error(f"Receive transcription error: {e}")

if __name__ == "__main__":
    asyncio.run(send_receive())














# async def send_audio(websocket):
#     buffer_size = 512 * 1024  #HAVE TO HAVE 512!!
#     audio_buffer = bytearray()
#
#     with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
#         if response.status_code == 200:
#             print("Starting to stream audio file...")
#
#             for chunk in response.iter_content(chunk_size=1024):  # Stream in chunks
#                 if chunk:
#                     audio_buffer.extend(chunk)
#                     #print(f"Received audio chunk of size {len(chunk)} bytes.")
#
#                     # Send buffered audio data once it's large enough
#                 if len(audio_buffer) >= buffer_size:
#                     await websocket.send(audio_buffer)
#                         #print(f"Sent {len(audio_buffer)} bytes of audio data.")
#                     audio_buffer.clear()
#                     await asyncio.sleep(0.01)
#
#             print("Finished sending audio.")
#         else:
#             print(f"Failed to download audio file. Status code: {response.status_code}")
#
#
# async def receive_transcription(websocket):
#     while True:
#         try:
#
#             transcription = await websocket.recv()
#             # Receive transcription from the server
#             print(f"Transcription: {transcription}")
#         except Exception as e:
#             print(f"Error receiving transcription: {e}")
#             #await asyncio.sleep(30)
#             break
#
#
# async def send_heartbeat(websocket):
#     while True:
#         try:
#             await websocket.ping()
#             print("Sent keepalive ping")
#         except websockets.ConnectionClosed:
#             print("Connection closed, stopping heartbeat")
#             break
#         await asyncio.sleep(30)  # Send ping every 30 seconds (adjust as needed)
#
#
# async def run_client():
#     uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/wtranscribe")  # WebSocket URL
#     ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
#     ssl_context.check_hostname = False
#     ssl_context.verify_mode = ssl.CERT_NONE
#     while True:
#         try:
#             async with websockets.connect(uri, ssl=ssl_context, ping_timeout=1000, ping_interval=50) as websocket:
#                 await asyncio.gather(
#                     send_audio(websocket),
#                     receive_transcription(websocket),
#                     send_heartbeat(websocket)
#                 )
#         except websockets.ConnectionClosedError as e:
#             print(f"WebSocket closed with error: {e}")
#         # except Exception as e:
#         #     print(f"Unexpected error: {e}")
#         #
#         # print("Reconnecting in 5 seconds...")
#         # await asyncio.sleep(5)  # Wait 5 seconds before reconnecting
#
# asyncio.run(run_client())
