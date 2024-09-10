# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for soundfile and any other audio-related processing
RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for Hugging Face Spaces (git for model fetching)
RUN apt-get install -y git

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Hugging Face Spaces will expose port 7860 by default for web applications
EXPOSE 7860

# Command to run the transcription script or API server on Hugging Face
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "7860"]

