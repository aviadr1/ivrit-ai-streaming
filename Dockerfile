FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.11

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libcublas11 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Create a writable cache directory for Hugging Face
RUN mkdir -p /hf_cache && chmod -R 777 /hf_cache

# Set the environment variable for the Hugging Face cache
ENV HF_HOME=/hf_cache

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Run FastAPI with Uvicorn
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port","7860","--timeout-keep-alive","300","--timeout-graceful-shutdown", "60"]
