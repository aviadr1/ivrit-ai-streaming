import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="transcription_log.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Initialize FastAPI app
app = FastAPI()

# Log initialization of the application
logging.info("FastAPI application started.")

# Load the Whisper model and processor
model_name = "openai/whisper-base"
logging.info(f"Loading Whisper model: {model_name}")

try:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    logging.info(f"Model {model_name} successfully loaded.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise e

# Move model to the appropriate device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logging.info(f"Model is using device: {device}")


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Log file upload start
    logging.info(f"Received audio file: {file.filename}")
    start_time = datetime.now()

    # Save the uploaded file
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb+") as f:
            f.write(await file.read())
        logging.info(f"File saved to: {file_location}")
    except Exception as e:
        logging.error(f"Error saving the file: {e}")
        return {"error": f"Error saving the file: {e}"}

    # Load the audio file and preprocess it
    try:
        audio_input, _ = sf.read(file_location)
        logging.info(f"Audio file {file.filename} successfully read.")

        inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
        logging.info(f"Audio file preprocessed for transcription.")
    except Exception as e:
        logging.error(f"Error processing the audio file: {e}")
        return {"error": f"Error processing the audio file: {e}"}

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logging.info("Inputs moved to the appropriate device.")

    # Generate the transcription
    try:
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        logging.info("Transcription successfully generated.")
    except Exception as e:
        logging.error(f"Error during transcription generation: {e}")
        return {"error": f"Error during transcription generation: {e}"}

    # Decode the transcription
    try:
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logging.info("Transcription successfully decoded.")
    except Exception as e:
        logging.error(f"Error decoding the transcription: {e}")
        return {"error": f"Error decoding the transcription: {e}"}

    # Clean up the temporary file
    try:
        os.remove(file_location)
        logging.info(f"Temporary file {file_location} deleted.")
    except Exception as e:
        logging.error(f"Error deleting the temporary file: {e}")

    end_time = datetime.now()
    time_taken = end_time - start_time
    logging.info(f"Transcription completed in {time_taken.total_seconds()} seconds.")

    return {"transcription": transcription, "processing_time_seconds": time_taken.total_seconds()}


if __name__ == "__main__":
    # Log application start
    logging.info("Starting FastAPI server with Uvicorn...")

    # Run the FastAPI app on the default port (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)
