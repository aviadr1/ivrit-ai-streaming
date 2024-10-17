---
title: Ivrit Ai Streaming
emoji: 📈
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
license: mit
---


# Hebrew Streaming Speech-to-Text (STT) Server

This project builds a streaming Speech-to-Text (STT) server for Hebrew. It leverages a combination of powerful tools to enable real-time transcription using models compatible with **faster-whisper**.

## Components

1. **Ivrit-AI Model**: 
   - We use the model [ivrit-ai/faster-whisper-v2-d4](https://huggingface.co/ivrit-ai/faster-whisper-v2-d4) for Hebrew STT. This code is designed to work with any **faster-whisper** compatible model.
   
2. **Whisper-Streaming**:
   - The project includes `whisper-streaming` (embedded via Git subtree) to convert batch-based Whisper models into real-time streaming models.

3. **Custom WebSocket Endpoint**:
   - A WebSocket (`/ws/`) endpoint is implemented to handle audio streaming in real-time. The current implementation expects raw PCM16 (int16) audio as input and returns JSON responses containing timestamps and recognized text.

4. **Docker with CUDA**:
   - The server is built with Docker, utilizing CUDA for GPU acceleration. It is designed to be run on Hugging Face or any GPU-enabled infrastructure (non-serverless).

5. **WebSocket Client**:
   - Sample client code is included for testing the WebSocket server and sending audio streams.

## WebSocket Endpoint (`/ws/`)

- **Input**: The `/ws/` WebSocket currently accepts raw PCM16 audio (int16 format).
- **Output**: It returns JSON objects containing timestamps and recognized text.

### WebSocket Client Example
The project contains an example WebSocket client that streams audio to the `/ws/` endpoint:

## Docker Setup

The Docker container is CUDA-enabled and can be deployed on any GPU-supported infrastructure such as Hugging Face or any cloud-based server.

## TODO

Several improvements are planned for this project. Please refer to [TODO.md](./TODO.md) for a list of upcoming features and enhancements. Some of the planned updates include:

- Adding a handshake to negotiate audio formats and sample rates (e.g., PCM16, ogg, 16kHz, 48kHz).
- Supporting resampling using `torchaudio`.
- Ogg format support for LiveKit integration via `pyogg`.
- Integration of the LiveKit plugin.

## Contributing

Feel free to open issues and pull requests to improve the project. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
