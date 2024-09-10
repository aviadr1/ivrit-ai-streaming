import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import requests
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Print initialization of the application
print("FastAPI application started.")

# Load the Whisper model and processor
model_name = "openai/whisper-base"
print(f"Loading Whisper model: {model_name}")

try:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    print(f"Model {model_name} successfully loaded.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise e

# Move model to the appropriate device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model is using device: {device}")


@app.post("/transcribe/")
def transcribe_audio_url(audio_url: str = Form(...)):
    # Download the audio file from the provided URL
    try:
        response = requests.get(audio_url)
        if response.status_code != 200:
            return {"error": f"Failed to download audio from URL. Status code: {response.status_code}"}
        print(f"Successfully downloaded audio from URL: {audio_url}")
        audio_data = io.BytesIO(response.content)  # Store audio data in memory
    except Exception as e:
        print(f"Error downloading the audio file: {e}")
        return {"error": f"Error downloading the audio file: {e}"}

    # Process the audio
    try:
        audio_input, _ = sf.read(audio_data)  # Read the audio from the in-memory BytesIO
        print(f"Audio file from URL successfully read.")
    except Exception as e:
        print(f"Error reading the audio file: {e}")
        return {"error": f"Error reading the audio file: {e}"}

    # Preprocess the audio for Whisper
    try:
        inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
        print(f"Audio file preprocessed for transcription.")
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return {"error": f"Error processing the audio file: {e}"}

    # Move inputs to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("Inputs moved to the appropriate device.")

    # Generate the transcription
    try:
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        print("Transcription successfully generated.")
    except Exception as e:
        print(f"Error during transcription generation: {e}")
        return {"error": f"Error during transcription generation: {e}"}

    # Decode the transcription
    try:
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print("Transcription successfully decoded.")
    except Exception as e:
        print(f"Error decoding the transcription: {e}")
        return {"error": f"Error decoding the transcription: {e}"}

    return {"transcription": transcription}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Whisper transcription API"}

if __name__ == "__main__":
    # Print when starting the FastAPI server
    print("Starting FastAPI server with Uvicorn...")

    # Run the FastAPI app on the default port (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)
