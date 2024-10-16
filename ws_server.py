# Import the necessary components from whisper_online.py
import logging
import os
from logging import DEBUG
from typing import Optional

import librosa
import soundfile
import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketDisconnect

from libs.whisper_streaming.whisper_online import (
    ASRBase,
    OnlineASRProcessor,
    VACOnlineASRProcessor,
    # add_shared_args,
    asr_factory,
    set_logging,
    create_tokenizer,
    load_audio,
    load_audio_chunk, OpenaiApiASR,
    set_logging, add_shared_args
)

import argparse
import sys
import numpy as np
import io
import soundfile
import wave
import requests
import argparse

# from libs.whisper_streaming.whisper_online_server import online

def my_set_logging(args,logger,other="_server"):
    logging.basicConfig(#format='%(name)s
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online"+other).setLevel(args.log_level)
#    logging.getLogger("whisper_online_server").setLevel(args.log_level)


logging.basicConfig(#format='%(name)s
            format='%(levelname)s\t%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisper_online").setLevel(logging.DEBUG)

SAMPLING_RATE = 16000
WARMUP_FILE = "mono16k.test_hebrew.wav"
AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"

app = FastAPI()
args = argparse.ArgumentParser()

# def my_add_shared_args(parser):
#     """shared args for simulation (this entry point) and server
#     parser: argparse.ArgumentParser object
#     """
#     parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
#     # AVIAD: allow custom model
#     parser.add_argument('--model', type=str, help="Name size of the Whisper model to use. The model is automatically downloaded from the model hub if not present in model cache dir.")
#     parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
#     parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
#     parser.add_argument('--lan', '--language', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
#     parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
#     parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "openai-api"],help='Load only this backend for Whisper processing.')
#     parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
#     parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
#     parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
#     parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
#     parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
#     parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level", default='DEBUG')


def drop_option_from_parser(parser, option_name):
    """
    Reinitializes the parser and copies all options except the specified option.

    Args:
        parser (argparse.ArgumentParser): The original argument parser.
        option_name (str): The option string to drop (e.g., '--model').

    Returns:
        argparse.ArgumentParser: A new parser without the specified option.
    """
    # Create a new parser with the same description and other attributes
    new_parser = argparse.ArgumentParser(
        description=parser.description,
        epilog=parser.epilog,
        formatter_class=parser.formatter_class
    )

    # Iterate through all the arguments of the original parser
    for action in parser._actions:
        if "-h" in action.option_strings:
            continue

        # Check if the option is not the one to drop
        if option_name not in action.option_strings :
            new_parser._add_action(action)

    return new_parser

def convert_to_mono_16k(input_wav: str, output_wav: str) -> None:
    """
    Converts any .wav file to mono 16 kHz.

    Args:
        input_wav (str): Path to the input .wav file.
        output_wav (str): Path to save the output .wav file with mono 16 kHz.
    """
    # Step 1: Load the audio file with librosa
    audio_data, original_sr = librosa.load(input_wav, sr=None, mono=False)  # Load at original sampling rate
    logger.info("Loaded audio with shape: %s, original sampling rate: %d" % (audio_data.shape, original_sr))

    # Step 2: If the audio has multiple channels, average them to make it mono
    if audio_data.ndim > 1:
        audio_data = librosa.to_mono(audio_data)

    # Step 3: Resample the audio to 16 kHz
    resampled_audio = librosa.resample(audio_data, orig_sr=original_sr, target_sr=SAMPLING_RATE)

    # Step 4: Save the resampled audio as a .wav file in mono at 16 kHz
    sf.write(output_wav, resampled_audio, SAMPLING_RATE)

    logger.info(f"Converted audio saved to {output_wav}")

def download_warmup_file():
    # Download the audio file if not already present
    audio_file_path = "test_hebrew.wav"
    if not os.path.exists(WARMUP_FILE):
        if not os.path.exists(audio_file_path):
            response = requests.get(AUDIO_FILE_URL)
            with open(audio_file_path, 'wb') as f:
                f.write(response.content)

        convert_to_mono_16k(audio_file_path, WARMUP_FILE)




class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    websocket: WebSocket
    asr: ASRBase
    online: OnlineASRProcessor
    min_limit: int

    is_first: bool = True
    last_end: Optional[float] = None

async def receive_audio_chunk(state: State) -> Optional[np.ndarray]:
    # receive all audio that is available by this time
    # blocks operation if less than self.min_chunk seconds is available
    # unblocks if connection is closed or a chunk is available
    out = []
    while sum(len(x) for x in out) < state.min_limit:
        raw_bytes = await state.websocket.receive_bytes()
        if not raw_bytes:
            break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
        sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
        audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
        out.append(audio)
    if not out:
        return None
    flat_out = np.concatenate(out)
    if state.is_first and len(flat_out) < state.min_limit:
        return None

    state.is_first = False
    return flat_out

def format_output_transcript(state, o) -> dict:
    # output format in stdout is like:
    # 0 1720 Takhle to je
    # - the first two words are:
    #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript

    # This function differs from whisper_online.output_transcript in the following:
    # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
    # Therefore, beg, is max of previous end and current beg outputed by Whisper.
    # Usually it differs negligibly, by appx 20 ms.

    if o[0] is not None:
        beg, end = o[0]*1000,o[1]*1000
        if state.last_end is not None:
            beg = max(beg, state.last_end)

        state.last_end = end
        print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
        return {
            "start": "%1.0f" % beg,
            "end": "%1.0f" % end,
            "text": "%s" % o[2],
        }
    else:
        logger.debug("No text in this segment")
        return None

# Define WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection request received.")
    await websocket.accept()
    logger.info("WebSocket connection established successfully.")

    parsed_args = args.parse_args([
        '--lan', 'he',
        '--model', 'ivrit-ai/faster-whisper-v2-d4',
        '--backend', 'faster-whisper',
        '--vad',
        "-l", "DEBUG",
        # '--vac', '--buffer_trimming', 'segment', '--buffer_trimming_sec', '15', '--min_chunk_size', '1.0', '--vac_chunk_size', '0.04', '--start_at', '0.0', '--offline', '--comp_unaware', '--log_level', 'DEBUG'
    ])

    # initialize the ASR model
    logger.info("Loading whisper model...")
    asr, online = asr_factory(parsed_args)
    state = State(
        websocket=websocket,
        asr=asr,
        online=online,
        min_limit=parsed_args.min_chunk_size * SAMPLING_RATE,
    )

    # warm up the ASR because the very first transcribe takes more time than the others.
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    logger.info("Warming up the whisper model...")
    a = load_audio_chunk(WARMUP_FILE, 0, 1)
    asr.transcribe(a)
    logger.info("Whisper is warmed up.")

    try:
        while True:
            a = await receive_audio_chunk(state)
            if a is None:
                break
            state.online.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                if result := format_output_transcript(state, o):
                    await websocket.send_json(result)

            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed by the client.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during WebSocket transcription: {e}")
                await websocket.send_json({"error": str(e)})
    finally:
        logger.info("Cleaning up and closing WebSocket connection.")



def main():
    global args
    add_shared_args(args)

    args = drop_option_from_parser(args, '--model')
    args.add_argument('--model', type=str,
                      help="Name size of the Whisper model to use. The model is automatically downloaded from the model hub if not present in model cache dir.")




main()
if __name__ == "__main__":
    uvicorn.run(app, port=4400)
