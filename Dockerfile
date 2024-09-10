# Use an official Python runtime as a base image
from python:3.11.1-buster

# Set the working directory
WORKDIR /

# Create a writable cache directory for Hugging Face
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

# Set the environment variable for the Hugging Face cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Hugging Face Spaces will expose port 7860 by default for web applications
EXPOSE 7860

# Command to run the transcription script or API server on Hugging Face
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "7860"]

