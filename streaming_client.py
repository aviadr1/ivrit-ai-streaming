# remote_whisper.py

import sys
import time
import logging
import os
from wave import Wave_read

import requests

import json
import base64
import numpy as np
import soundfile as sf
import io

import librosa

# Import the necessary components from whisper_online.py
from libs.whisper_streaming.whisper_online import (
    ASRBase,
    OnlineASRProcessor,
    VACOnlineASRProcessor,
    add_shared_args,
    asr_factory as original_asr_factory,
    set_logging,
    create_tokenizer,
    load_audio,
    load_audio_chunk, OpenaiApiASR,
)
from model import dict_to_segment, get_raw_words_from_segments

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)



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
    target_sr = 16000
    resampled_audio = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)

    # Step 4: Save the resampled audio as a .wav file in mono at 16 kHz
    sf.write(output_wav, resampled_audio, target_sr)

    logger.info(f"Converted audio saved to {output_wav}")


# Example usage:
# convert_to_mono_16k('input_audio.wav', 'output_audio_16k_mono.wav')


# Define the RemoteFasterWhisperASR class
class RemoteFasterWhisperASR(ASRBase):
    """Uses a remote FasterWhisper model via WebSocket."""
    sep = ""  # Same as FasterWhisperASR

    def load_model(self, *args, **kwargs):
        import websocket
        self.ws = websocket.WebSocket()
        # Replace with your server address
        server_address = "ws://localhost:8000/ws_transcribe_streaming"  # Update with the actual server address
        self.ws.connect(server_address)
        logger.info(f"Connected to remote ASR server at {server_address}")

    def transcribe(self, audio, init_prompt=""):
        # Convert audio data to WAV bytes
        if isinstance(audio, str):
            # If audio is a filename, read the file
            with open(audio, 'rb') as f:
                audio_bytes = f.read()
        elif isinstance(audio, np.ndarray):
            # Write audio data to a buffer in WAV format
            audio_bytes_io = io.BytesIO()
            sf.write(audio_bytes_io, audio, samplerate=16000, format='WAV', subtype='PCM_16')
            audio_bytes = audio_bytes_io.getvalue()
        else:
            raise ValueError("Unsupported audio input type")

        # Encode to base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        data = {
            'audio': audio_b64,
            'init_prompt': init_prompt
        }
        self.ws.send(json.dumps(data))
        response = self.ws.recv()
        segments = json.loads(response)
        segments = [dict_to_segment(s) for s in segments]
        logger.info(get_raw_words_from_segments(segments))
        return segments

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

# Update asr_factory to include RemoteFasterWhisperASR
def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and Online ASR Processor instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    elif backend == "remote-faster-whisper":
        asr_cls = RemoteFasterWhisperASR
    else:
        # Use the original asr_factory for other backends
        return original_asr_factory(args, logfile)

    # For RemoteFasterWhisperASR
    t = time.time()
    logger.info(f"Initializing Remote Faster Whisper ASR for language '{args.lan}'...")
    asr = asr_cls(modelsize=args.model, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
    e = time.time()
    logger.info(f"Initialization done. It took {round(e - t, 2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(
            args.min_chunk_size,
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec)
        )
    else:
        online = OnlineASRProcessor(
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec)
        )

    return asr, online

# Now, write the main function that uses RemoteFasterWhisperASR
def main():
    import argparse
    import sys
    import numpy as np
    import io
    import soundfile as sf
    import wave

    # Download the audio file if not already present
    AUDIO_FILE_URL = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"
    audio_file_path = "test_hebrew.wav"
    mono16k_audio_file_path = "mono16k." + audio_file_path
    if not os.path.exists(audio_file_path):
        response = requests.get(AUDIO_FILE_URL)
        with open(audio_file_path, 'wb') as f:
            f.write(response.content)

    if not os.path.exists(mono16k_audio_file_path):
        convert_to_mono_16k(audio_file_path, mono16k_audio_file_path)

    # Set up arguments
    class Args:
        def __init__(self):
            self.audio_path = mono16k_audio_file_path
            self.lan = 'he'
            self.model = None  # Not used in RemoteFasterWhisperASR
            self.model_cache_dir = None
            self.model_dir = None
            self.backend = 'remote-faster-whisper'
            self.task = 'transcribe'
            self.vad = False
            self.vac = True  # Use VAC as default
            self.buffer_trimming = 'segment'
            self.buffer_trimming_sec = 15
            self.min_chunk_size = 1.0
            self.vac_chunk_size = 0.04
            self.start_at = 0.0
            self.offline = False
            self.comp_unaware = False
            self.log_level = 'DEBUG'

    args = Args()

    # Set up logging
    set_logging(args, logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000

    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=sys.stderr)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # Load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path, 0, 1)

    # Warm up the ASR because the very first transcribe takes more time
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg

    def output_transcript(o, now=None):
        # Output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - The first three numbers are:
        #   - Emission time from the beginning of processing, in milliseconds
        #   - Begin and end timestamp of the text segment, as estimated by Whisper model
        # - The next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), flush=True)
        else:
            # No text, so no output
            pass

    end = 0
    while True:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        end = time.time() - start
        a = load_audio_chunk(audio_path, beg, end)
        beg = end
        online.insert_audio_chunk(a)

        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"Assertion error: {e}")
            pass
        else:
            output_transcript(o)
        now = time.time() - start
        logger.debug(f"## Last processed {end:.2f} s, now is {now:.2f}, latency is {now - end:.2f}")

        if end >= duration:
            break
    now = None

    o = online.finish()
    output_transcript(o, now=now)

if __name__ == "__main__":
    main()
