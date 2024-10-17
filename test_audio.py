import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List
import os

def check_endianness(audio_file):
    with sf.SoundFile(audio_file) as f:
        data = f.read(dtype='int16')
        if data.dtype.byteorder == '>':
            print("Data is big-endian.")
        elif data.dtype.byteorder == '<':
            print("Data is little-endian.")
        else:
            print("Data byte order is native to the system (little-endian on most systems).")

def read_audio_in_chunks(audio_file, target_sr=16000, chunk_duration=1) -> List[np.ndarray]:
    """
    Reads a 16kHz mono audio file in 1-second chunks and returns them as little-endian 16-bit integer arrays.
    """
    if not str(audio_file).endswith(".wav"):
        wav_file = Path(audio_file).with_suffix(".wav")
        if not wav_file.exists():
            command = f'ffmpeg -i "{audio_file}" -ac 1 -ar {target_sr} "{wav_file}"'
            print(f"Converting MP3 to WAV: {command}")
            os.system(command)
        audio_file = wav_file

    # Read audio using soundfile
    with sf.SoundFile(audio_file) as f:
        if f.samplerate != target_sr:
            raise ValueError(f"Unexpected sample rate {f.samplerate}. Expected {target_sr}.")

        # Read the entire audio file as an array
        audio_data = f.read(dtype='int16')
        check_endianness(audio_file)

    # Calculate the number of samples per chunk
    samples_per_chunk = target_sr * chunk_duration

    # Split the audio into chunks
    chunks = [
        audio_data[i:i + samples_per_chunk]
        for i in range(0, len(audio_data), samples_per_chunk)
    ]

    return chunks


def reassemble_chunks_to_wav(chunks: List[np.ndarray], output_wav: str, target_sr=16000):
    """
    Reassembles audio chunks into a .wav file using soundfile with little-endian 16-bit integer format.

    Args:
        chunks (List[np.ndarray]): List of audio chunks (arrays).
        output_wav (str): Path to the output .wav file.
        target_sr (int): Target sample rate (default is 16000 Hz).
    """
    # Concatenate the chunks back into a single array
    full_audio = np.concatenate(chunks)

    # Write the reassembled audio back to a .wav file with little-endian 16-bit PCM format
    sf.write(output_wav, full_audio, target_sr, subtype='PCM_16')

    print(f"Reassembled audio saved to {output_wav}")


import io
import librosa


def save_chunks_with_librosa(chunks: List[np.ndarray], output_wav: str, target_sr=16000):
    """
    Receives chunks, processes them using librosa, and saves them to a .wav file.
    Args:
        chunks (List[np.ndarray]): List of audio chunks (arrays).
        output_wav (str): Path to the output .wav file.
        target_sr (int): Target sample rate (default is 16000 Hz).
    """
    concatenated_audio = np.concatenate(chunks)

    # Wrap the audio in a BytesIO stream and then process it with librosa
    raw_bytes = concatenated_audio.tobytes()

    sf_file = sf.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=target_sr,
                           subtype="PCM_16", format="RAW")

    # Use librosa to load the audio from the soundfile object
    audio, _ = librosa.load(sf_file, sr=target_sr, dtype=np.float32)

    # Save the processed audio to a file using soundfile
    sf.write(output_wav, audio, target_sr, subtype='PCM_16')

    print(f"Processed audio saved to {output_wav}")



# Read and split audio into chunks
chunks = read_audio_in_chunks('lex.wav')

# Reassemble and save the chunks into a new .wav file
reassemble_chunks_to_wav(chunks, 'lex.output.wav')

save_chunks_with_librosa(chunks, 'lex_librosa_output.wav')
