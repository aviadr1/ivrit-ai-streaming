import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
from datetime import datetime

# Ensure the log directory exists (optional if needed)
log_directory = "/app/logs"
os.makedirs(log_directory, exist_ok=True)

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
async def transcribe_audio(file: UploadFile = File(...)):
    # Print file upload start
    print(f"Received audio file: {file.filename}")

    # Save the uploaded file
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb+") as f:
            f.write(await file.read())
        print(f"File saved to: {file_location}")
    except Exception as e:
        print(f"Error saving the file: {e}")
        return {"error": f"Error saving the file: {e}"}

    # Load the audio file and preprocess it
    try:
        audio_input, _ = sf.read(file_location)
        print(f"Audio file {file.filename} successfully read.")

        inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
        print(f"Audio file preprocessed for transcription.")
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return {"error": f"Error processing the audio file: {e}"}

    # Move inputs to the same device as the model
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

    # Clean up the temporary file
    try:
        os.remove(file_location)
        print(f"Temporary file {file_location} deleted.")
    except Exception as e:
        print(f"Error deleting the temporary file: {e}")

    return {"transcription": transcription}


if __name__ == "__main__":
    # Print when starting the FastAPI server
    print("Starting FastAPI server with Uvicorn...")

    # Run the FastAPI app on the default port (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)
