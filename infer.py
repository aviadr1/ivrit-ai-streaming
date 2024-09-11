import base64
import faster_whisper
import tempfile
import torch
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Device selected: {device}')

model_name = 'ivrit-ai/faster-whisper-v2-d4'
logging.info(f'Loading model: {model_name}')
model = faster_whisper.WhisperModel(model_name, device=device)
logging.info('Model loaded successfully')

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024
logging.info(f'Max payload size set to: {MAX_PAYLOAD_SIZE} bytes')

app = FastAPI()


class InputData(BaseModel):
    type: str
    data: Optional[str] = None  # Used for blob input
    url: Optional[str] = None  # Used for url input


def download_file(url, max_size_bytes, output_filename, api_key=None):
    """
    Download a file from a given URL with size limit and optional API key.
    """
    logging.debug(f'Starting file download from URL: {url}')
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            logging.debug('API key provided, added to headers')

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        file_size = int(response.headers.get('Content-Length', 0))
        logging.info(f'File size: {file_size} bytes')

        if file_size > max_size_bytes:
            logging.error(f'File size exceeds limit: {file_size} > {max_size_bytes}')
            return False

        downloaded_size = 0
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                logging.debug(f'Downloaded {downloaded_size} bytes')
                if downloaded_size > max_size_bytes:
                    logging.error('Downloaded size exceeds maximum allowed payload size')
                    return False
                file.write(chunk)

        logging.info(f'File downloaded successfully: {output_filename}')
        return True

    except requests.RequestException as e:
        logging.error(f"Error downloading file: {e}")
        return False


@app.post("/transcribe")
async def transcribe(input_data: InputData):
    logging.INFO(f'Received transcription request with data: {input_data}')
    datatype = input_data.type
    if not datatype:
        logging.error('datatype field not provided')
        raise HTTPException(status_code=400, detail="datatype field not provided. Should be 'blob' or 'url'.")

    if datatype not in ['blob', 'url']:
        logging.error(f'Invalid datatype: {datatype}')
        raise HTTPException(status_code=400, detail=f"datatype should be 'blob' or 'url', but is {datatype} instead.")

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'
        logging.debug(f'Created temporary directory: {d}')

        if datatype == 'blob':
            if not input_data.data:
                logging.error("Missing 'data' for 'blob' input")
                raise HTTPException(status_code=400, detail="Missing 'data' for 'blob' input.")
            logging.info('Decoding base64 blob data')
            mp3_bytes = base64.b64decode(input_data.data)
            open(audio_file, 'wb').write(mp3_bytes)
            logging.info(f'Audio file written: {audio_file}')
        elif datatype == 'url':
            if not input_data.url:
                logging.error("Missing 'url' for 'url' input")
                raise HTTPException(status_code=400, detail="Missing 'url' for 'url' input.")
            logging.info(f'Downloading file from URL: {input_data.url}')
            success = download_file(input_data.url, MAX_PAYLOAD_SIZE, audio_file, None)
            if not success:
                logging.error(f"Error downloading data from {input_data.url}")
                raise HTTPException(status_code=400, detail=f"Error downloading data from {input_data.url}")

        result = transcribe_core(audio_file)
        return {"result": result}


def transcribe_core(audio_file):
    logging.info('Starting transcription...')
    ret = {'segments': []}

    segs, _ = model.transcribe(audio_file, language='he', word_timestamps=True)
    logging.info('Transcription completed')

    for s in segs:
        words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]
        seg = {
            'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
            'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words
        }
        logging.debug(f'Transcription segment: {seg}')
        ret['segments'].append(seg)

    return ret
