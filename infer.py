import faster_whisper
import requests
import tempfile
import os

# Load the faster-whisper model that supports Hebrew
model = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d4")

# URL of the audio file (replace this with the actual URL of your audio)
audio_url = "https://github.com/AshDavid12/runpod-serverless-forked/blob/main/me-hebrew.wav"

# Download the audio file from the URL
response = requests.get(audio_url)
if response.status_code != 200:
    raise Exception("Failed to download audio file")

# Create a temporary file to store the audio
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
    tmp_audio_file.write(response.content)
    tmp_audio_file_path = tmp_audio_file.name

# Perform the transcription
segments, info = model.transcribe(tmp_audio_file_path, language="he")

# Print transcription results
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# Clean up the temporary file
os.remove(tmp_audio_file_path)



















# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import requests
# import soundfile as sf
# import io


# # Load the Whisper model and processor from Hugging Face Model Hub
# model_name = "openai/whisper-base"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
#
# # Use GPU if available, otherwise use CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
#
# # URL of the audio file
# audio_url = "https://www.signalogic.com/melp/EngSamples/Orig/male.wav"
#
# # Download the audio file
# response = requests.get(audio_url)
# audio_data = io.BytesIO(response.content)
#
# # Read the audio using soundfile
# audio_input, _ = sf.read(audio_data)
#
# # Preprocess the audio for Whisper
# inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
# attention_mask = inputs['input_features'].ne(processor.tokenizer.pad_token_id).long()
#
# # Move inputs and attention mask to the correct device
# inputs = {key: value.to(device) for key, value in inputs.items()}
# attention_mask = attention_mask.to(device)
#
# # Generate the transcription with attention mask
# with torch.no_grad():
#     predicted_ids = model.generate(
#         inputs["input_features"],
#         attention_mask=attention_mask  # Pass attention mask explicitly
#     )
# # Decode the transcription
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#
# # Print the transcription result
# print("Transcription:", transcription)
