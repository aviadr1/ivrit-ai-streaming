import asyncio
import websockets
import requests
import ssl

# Parameters for reading and sending the audio
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"  # Use WAV file

processed_segments = set()

def process_transcription_results(transcription_result):
    global processed_segments  # Ensure we use the same set across multiple calls
    new_segments = []

    # Iterate over all segments in the transcription result
    for segment in transcription_result.get("segments", []):
        # You can use a unique identifier like 'id' or a combination of 'start' and 'end' times
        segment_id = segment.get("id")

        # Check if the segment is already processed
        if segment_id not in processed_segments:
            # Process the new segment (do your actual processing here)
            new_segments.append(segment)

            # Mark the segment as processed by adding its 'id' to the set
            processed_segments.add(segment_id)
            print(f"Processed segment ID: {segment_id}")
        else:
            print(f"Skipping already processed segment ID: {segment_id}")

    # Return only new segments that have not been processed before
    return new_segments


async def send_audio(websocket):
    buffer_size = 512 * 1024  # Buffer audio chunks up to 512KB before sending
    audio_buffer = bytearray()

    with requests.get(AUDIO_FILE_URL, stream=True, allow_redirects=False) as response:
        if response.status_code == 200:
            print("Starting to stream audio file...")

            for chunk in response.iter_content(chunk_size=1024):  # Stream in chunks
                if chunk:
                    audio_buffer.extend(chunk)
                    #print(f"Received audio chunk of size {len(chunk)} bytes.")

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
            # new_segments = process_transcription_results(transcription)
            # # Now handle only new segments
            # if new_segments:
            #     for segment in new_segments:
            #         print(f"New Segment: {segment['text']}")
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
        await asyncio.sleep(120)  # Send ping every 30 seconds (adjust as needed)


async def run_client():
    uri = ("wss://gigaverse-ivrit-ai-streaming.hf.space/wtranscribe")  # WebSocket URL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(uri, ssl=ssl_context, timeout=600) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_transcription(websocket),
            send_heartbeat(websocket)
        )

asyncio.run(run_client())
