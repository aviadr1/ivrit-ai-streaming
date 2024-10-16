import asyncio
import json
import logging
import wave

import websockets
import requests
import ssl
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

# Parameters for reading and sending the audio
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file

from pydub import AudioSegment


# Convert and resample audio before writing it to WAV
# Convert and resample audio before writing it to WAV
def convert_to_mono_16k(audio_file_path):
    logging.info(f"Starting audio conversion to mono and resampling to 16kHz for file: {audio_file_path}")

    try:
        # Load the audio file into an AudioSegment object
        audio_segment = AudioSegment.from_file(audio_file_path, format="wav")

        # Convert the audio to mono and resample it to 16kHz
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

        logging.info("Audio conversion to mono and 16kHz completed successfully.")
    except Exception as e:
        logging.error(f"Error during audio conversion: {e}")
        raise e

    # Return the modified AudioSegment object
    return audio_segment


async def send_audio(websocket):
    print(f"hi")
    buffer_size = 1024 * 16  # Send smaller chunks (16KB) for real-time processing
    logging.info("Converting the audio to mono and 16kHz.")

    try:
        converted_audio = convert_to_mono_16k('test_copy.wav')
    except Exception as e:
        logging.error(f"Failed to convert audio: {e}")
        return

    # Send metadata to the server
    metadata = {
        'sample_rate': 16000,  # Resampled rate
        'channels': 1,  # Converted to mono
        'sampwidth': 2  # Assuming 16-bit audio
    }
    await websocket.send(json.dumps(metadata))
    logging.info(f"Sent metadata: {metadata}")

    try:
        raw_data = converted_audio.raw_data
        logging.info(f"Starting to send raw PCM audio data. Total data size: {len(raw_data)} bytes.")

        for i in range(0, len(raw_data), buffer_size):
            pcm_chunk = raw_data[i:i + buffer_size]
            await websocket.send(pcm_chunk)  # Send raw PCM data chunk
            logging.info(f"Sent PCM chunk of size {len(pcm_chunk)} bytes.")
            await asyncio.sleep(0.01)  # Simulate real-time sending

        logging.info("Completed sending all audio data.")
    except Exception as e:
        logging.error(f"Error while sending audio data: {e}")

    # Download the WAV file locally
    # with requests.get(AUDIO_FILE_URL, stream=True) as response:
    #     if response.status_code == 200:
    #         with open('downloaded_audio.wav', 'wb') as f:
    #             for chunk in response.iter_content(chunk_size=1024):
    #                 f.write(chunk)
    #         print("Audio file downloaded successfully.")

            # Open the downloaded WAV file and extract PCM data
    # with wave.open('test_copy.wav', 'rb') as wav_file:
    #     metadata = {
    #         'sample_rate': wav_file.getframerate(),
    #         'channels': wav_file.getnchannels(),
    #         'sampwidth': wav_file.getsampwidth(),
    #     }
    #
    #     # Send metadata to the server before sending the audio
    #     await websocket.send(json.dumps(metadata))
    #     print(f"Sent metadata: {metadata}")

        # # Send the PCM audio data in chunks
        # while True:
        #     pcm_chunk = wav_file.readframes(buffer_size)
        #     if not pcm_chunk:
        #         break  # End of file
        #
        #     await websocket.send(pcm_chunk)  # Send raw PCM data chunk
        #     #print(f"Sent PCM chunk of size {len(pcm_chunk)} bytes.")
        #     await asyncio.sleep(0.01)  # Simulate real-time sending

        # else:
        #     print(f"Failed to download audio file. Status code: {response.status_code}")


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