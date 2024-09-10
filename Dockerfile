# Use an official Python runtime as a base image
FROM python:3.11.1-buster

# Set the working directory
WORKDIR /

# Create a writable cache directory for Hugging Face
RUN mkdir -p /hf_cache && chmod -R 777 /app/hf_cache

# Set the environment variable for the Hugging Face cache
ENV HF_HOME=/hf_cache

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the Python transcription script directly
CMD ["python", "infer.py.py"]
