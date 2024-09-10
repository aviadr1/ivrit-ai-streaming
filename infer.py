import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import requests
import soundfile as sf
import io

# Load the Whisper model and processor from Hugging Face Model Hub
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# URL of the audio file
audio_url = "https://www.signalogic.com/melp/EngSamples/Orig/male.wav"

# Download the audio file
response = requests.get(audio_url)
audio_data = io.BytesIO(response.content)

# Read the audio using soundfile
audio_input, _ = sf.read(audio_data)

# Preprocess the audio for Whisper
inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
attention_mask = inputs['input_features'].ne(processor.tokenizer.pad_token_id).long()

# Move inputs and attention mask to the correct device
inputs = {key: value.to(device) for key, value in inputs.items()}
attention_mask = attention_mask.to(device)

# Generate the transcription with attention mask
with torch.no_grad():
    predicted_ids = model.generate(
        inputs["input_features"],
        attention_mask=attention_mask  # Pass attention mask explicitly
    )
# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Print the transcription result
print("Transcription:", transcription)
