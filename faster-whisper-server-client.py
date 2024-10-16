import argparse
import json
import threading
import time
import websocket
import os

import librosa
import numpy as np

# Define the default WebSocket endpoint
DEFAULT_WS_URL = "ws://localhost:8000/v1/audio/transcriptions"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stream audio to the transcription WebSocket endpoint.")
    parser.add_argument("audio_file", help="Path to the input audio file.")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint URL.")
    parser.add_argument("--model", type=str, help="Model name to use for transcription.")
    parser.add_argument("--language", type=str, help="Language code for transcription.")
    parser.add_argument(
        "--response_format",
        type=str,
        default="verbose_json",
        choices=["text", "json", "verbose_json"],
        help="Response format.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for transcription.")
    parser.add_argument("--vad_filter", action="store_true", help="Enable voice activity detection filter.")
    parser.add_argument("--chunk_duration", type=float, default=1.0, help="Duration of each audio chunk in seconds.")
    return parser.parse_args()


def preprocess_audio(audio_file, target_sr=16000):
    """
    Load the audio file, convert to mono 16kHz, and return the audio data.
    """
    if audio_file.endswith(".mp3"):
        # Convert MP3 to WAV using ffmpeg
        wav_file = audio_file.replace(".mp3", ".wav")
        if not os.path.exists(wav_file):
            command = f'ffmpeg -i "{audio_file}" -ac 1 -ar {target_sr} "{wav_file}"'
            print(f"Converting MP3 to WAV: {command}")
            os.system(command)
        audio_file = wav_file

    print(f"Loading audio file {audio_file}")
    audio_data, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    return audio_data, sr

def chunk_audio(audio_data, sr, chunk_duration):
    """
    Split the audio data into chunks of specified duration.
    """
    chunk_samples = int(chunk_duration * sr)
    total_samples = len(audio_data)
    chunks = [
        audio_data[i:i + chunk_samples]
        for i in range(0, total_samples, chunk_samples)
    ]
    print(f"Split audio into {len(chunks)} chunks of {chunk_duration} seconds each.")
    return chunks


def build_query_params(args):
    """
    Build the query parameters for the WebSocket URL based on command-line arguments.
    """
    params = {}
    if args.model:
        params["model"] = args.model
    if args.language:
        params["language"] = args.language
    if args.response_format:
        params["response_format"] = args.response_format
    if args.temperature is not None:
        params["temperature"] = str(args.temperature)
    if args.vad_filter:
        params["vad_filter"] = "true"
    return params


def websocket_url_with_params(base_url, params):
    """
    Append query parameters to the WebSocket URL.
    """
    from urllib.parse import urlencode

    if params:
        query_string = urlencode(params)
        url = f"{base_url}?{query_string}"
    else:
        url = base_url
    return url


def on_message(ws, message):
    """
    Callback function when a message is received from the server.
    """
    try:
        data = json.loads(message)
        # Accumulate transcriptions
        if ws.args.response_format == "verbose_json":
            segments = data.get('segments', [])
            ws.transcriptions.extend(segments)
            for segment in segments:
                print(f"Received segment: {segment['text']}")
        else:
            # For 'json' or 'text' format
            ws.transcriptions.append(data)
            print(f"Transcription: {data['text']}")
    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")


def on_error(ws, error):
    """
    Callback function when an error occurs.
    """
    print(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    """
    Callback function when the WebSocket connection is closed.
    """
    print("WebSocket connection closed")


def on_open(ws):
    """
    Callback function when the WebSocket connection is opened.
    """
    print("WebSocket connection opened")
    ws.transcriptions = []  # Initialize the list to store transcriptions


def send_audio_chunks(ws, audio_chunks, sr):
    """
    Send audio chunks to the WebSocket server.
    """
    for idx, chunk in enumerate(audio_chunks):
        # Ensure little-endian format
        audio_bytes = chunk.astype('<f4').tobytes()
        ws.send(audio_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
        print(f"Sent chunk {idx + 1}/{len(audio_chunks)}")
        time.sleep(0.1)  # Small delay to simulate real-time streaming
    print("All audio chunks sent")
    # Optionally, wait to receive remaining messages
    time.sleep(2)
    ws.close()
    print("Closed WebSocket connection")



def format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    """
    total_milliseconds = int(seconds * 1000)
    hours = total_milliseconds // (3600 * 1000)
    minutes = (total_milliseconds % (3600 * 1000)) // (60 * 1000)
    secs = (total_milliseconds % (60 * 1000)) // 1000
    milliseconds = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def generate_srt(transcriptions):
    """
    Generate and print SRT content from transcriptions.
    """
    print("\nGenerated SRT:")
    for idx, segment in enumerate(transcriptions, 1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text']
        print(f"{idx}")
        print(f"{start_time} --> {end_time}")
        print(f"{text}\n")


def run_websocket_client(args):
    """
    Run the WebSocket client to stream audio and receive transcriptions.
    """
    audio_data, sr = preprocess_audio(args.audio_file)
    audio_chunks = chunk_audio(audio_data, sr, args.chunk_duration)

    params = build_query_params(args)
    ws_url = websocket_url_with_params(args.url, params)

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.args = args  # Attach args to ws to access inside callbacks

    # Run the WebSocket in a separate thread to allow sending and receiving simultaneously
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()

    # Wait for the connection to open
    while not ws.sock or not ws.sock.connected:
        time.sleep(0.1)

    # Send the audio chunks
    send_audio_chunks(ws, audio_chunks, sr)

    # Wait for the WebSocket thread to finish
    ws_thread.join()

    # Generate SRT if transcriptions are available
    if hasattr(ws, 'transcriptions') and ws.transcriptions:
        generate_srt(ws.transcriptions)
    else:
        print("No transcriptions received.")


if __name__ == "__main__":
    args = parse_arguments()
    run_websocket_client(args)
