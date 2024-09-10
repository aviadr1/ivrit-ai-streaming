import faster_whisper
import requests
from pydub import AudioSegment
import io

# Load the faster-whisper model that supports Hebrew
model = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d4")

# URL of the mp3 audio file
audio_url = "https://github.com/metaldaniel/HebrewASR-Comparison/raw/main/HaTankistiot_n12-mp3.mp3"

# Download the mp3 audio file from the URL
response = requests.get(audio_url)
if response.status_code != 200:
    raise Exception("Failed to download audio file")

# Load the mp3 audio into an in-memory buffer
mp3_audio = io.BytesIO(response.content)

# Convert the mp3 audio to wav using pydub (in memory)
audio = AudioSegment.from_file(mp3_audio, format="mp3")
wav_audio = io.BytesIO()
audio.export(wav_audio, format="wav")
wav_audio.seek(0)  # Reset the pointer to the beginning of the buffer

# Save the in-memory wav audio to a temporary file-like object
with io.BytesIO(wav_audio.read()) as temp_wav_file:
    # Perform the transcription
    segments, info = model.transcribe(temp_wav_file, language="he")

    # Print transcription results
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
