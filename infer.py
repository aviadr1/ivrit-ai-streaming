import base64
import faster_whisper
import tempfile
import torch
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'ivrit-ai/faster-whisper-v2-d4'
model = faster_whisper.WhisperModel(model_name, device=device)

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024

app = FastAPI()


class InputData(BaseModel):
    type: str
    data: Optional[str] = None  # Used for blob input
    url: Optional[str] = None  # Used for url input
    api_key: Optional[str] = None


def download_file(url, max_size_bytes, output_filename, api_key=None):
    """
    Download a file from a given URL with size limit and optional API key.
    """
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        file_size = int(response.headers.get('Content-Length', 0))

        if file_size > max_size_bytes:
            return False

        downloaded_size = 0
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > max_size_bytes:
                    return False
                file.write(chunk)

        return True

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


@app.post("/transcribe")
async def transcribe(input_data: InputData):
    datatype = input_data.type
    if not datatype:
        raise HTTPException(status_code=400, detail="datatype field not provided. Should be 'blob' or 'url'.")

    if datatype not in ['blob', 'url']:
        raise HTTPException(status_code=400, detail=f"datatype should be 'blob' or 'url', but is {datatype} instead.")

    api_key = input_data.api_key

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'

        if datatype == 'blob':
            if not input_data.data:
                raise HTTPException(status_code=400, detail="Missing 'data' for 'blob' input.")
            mp3_bytes = base64.b64decode(input_data.data)
            open(audio_file, 'wb').write(mp3_bytes)
        elif datatype == 'url':
            if not input_data.url:
                raise HTTPException(status_code=400, detail="Missing 'url' for 'url' input.")
            success = download_file(input_data.url, MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                raise HTTPException(status_code=400, detail=f"Error downloading data from {input_data.url}")

        result = transcribe_core(audio_file)
        return {"result": result}


def transcribe_core(audio_file):
    print('Transcribing...')

    ret = {'segments': []}
    segs, _ = model.transcribe(audio_file, language='he', word_timestamps=True)
    for s in segs:
        words = [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability} for w in s.words]
        seg = {
            'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
            'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words
        }
        print(seg)
        ret['segments'].append(seg)

    return ret


# Make sure Uvicorn starts correctly when deployed
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
