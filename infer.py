import base64
import faster_whisper
import tempfile
import logging
import torch
import sys
import requests
import os

import whisper_online

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Load the FasterWhisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'ivrit-ai/faster-whisper-v2-d3-e3'

try:
    lan = 'he'
    logging.info(f"Attempting to initialize FasterWhisperASR with device: {device}")
    model = whisper_online.FasterWhisperASR(lan=lan, modelsize=model_name)
    logging.info("FasterWhisperASR model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize FasterWhisperASR model: {e}")

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024

def download_file(url, max_size_bytes, output_filename, api_key=None):
    """Download a file from a given URL with size limit and optional API key."""
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size > max_size_bytes:
            print(f"File size exceeds the limit: {file_size} bytes.")
            return False
        downloaded_size = 0
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > max_size_bytes:
                    print(f"Download stopped: size limit exceeded.")
                    return False
                file.write(chunk)
        print(f"File downloaded successfully: {output_filename}")
        return True
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def transcribe_core_whisper(audio_file):
    """Transcribe the audio file using FasterWhisper."""
    logging.info(f"Transcribing audio file: {audio_file}")
    ret = {'segments': []}
    try:
        segs, dummy = model.transcribe(audio_file, language='he', word_timestamps=True)
        for s in segs:
            words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]
            seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
                   'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words}
            ret['segments'].append(seg)
        logging.info("Transcription completed successfully.")
    except Exception as e:
        logging.error(f"Error during transcription: {e}", exc_info=True)
    return ret

def transcribe_whisper(job):
    """Main transcription handler."""
    logging.info(f"Processing job: {job}")
    datatype = job.get('input', {}).get('type')
    if not datatype:
        return {"error": "datatype field not provided. Should be 'blob' or 'url'."}
    if datatype not in ['blob', 'url']:
        return {"error": f"Invalid datatype: {datatype}."}

    api_key = job.get('input', {}).get('api_key')
    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'
        if datatype == 'blob':
            mp3_bytes = base64.b64decode(job['input']['data'])
            with open(audio_file, 'wb') as f:
                f.write(mp3_bytes)
        elif datatype == 'url':
            success = download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                return {"error": f"Failed to download from {job['input']['url']}"}

        result = transcribe_core_whisper(audio_file)
        return {'result': result}

# Example job input to test locally
if __name__ == "__main__":
    test_job = {
        "input": {
            "type": "url",
            "url": "https://github.com/metaldaniel/HebrewASR-Comparison/raw/main/HaTankistiot_n12-mp3.mp3",
        }
    }
    print(transcribe_whisper(test_job))
