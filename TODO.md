# TODO

## 1. Implement Handshake on WebSocket Start

- **Objective**: When the `/ws/` WebSocket connection starts, there needs to be a handshake process that negotiates the audio format and sampling rate to be used.
  - **Supported formats**: 
    - `PCM16` (currently supported)
    - `ogg` (to support LiveKit integration)
  - **Supported sampling rates**:
    - Default: `16kHz` (currently supported)
    - Additional: `48kHz` (supported by LiveKit)
  - **Action Items**:
    - Implement a handshake to negotiate between client and server on the chosen format and sample rate.
    - Ensure the server handles both formats and rates dynamically based on the handshake.

## 2. Support Resampling Audio on Server

- **Objective**: Add functionality to resample incoming audio to match the required input format for the Whisper model.
  - **Resampling Tool**: Use `torchaudio` for resampling.
  - **Code Sample**:

    ```python
    import torchaudio

    def resample_audio(input_waveform, original_sample_rate, target_sample_rate=16000):
        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            return resampler(input_waveform)
        return input_waveform
    ```

  - **Action Items**:
    - Integrate this resampling logic within the audio processing pipeline.
    - Automatically resample audio based on the sampling rate negotiated during the handshake.

## 3. Support Decoding OGG Files

- **Objective**: Add support for decoding `ogg` format files using `pyogg`.
  - **Decoding Tool**: Use `pyogg` for decoding ogg files.
  - **Code Sample**:

    ```python
    import pyogg

    def decode_ogg(data):
        ogg_decoder = pyogg.VorbisFile(data)
        pcm_data = ogg_decoder.as_array()
        return pcm_data, ogg_decoder.frequency  # pcm_data is in PCM format
    ```

  - **Action Items**:
    - Integrate `pyogg` decoding into the audio pipeline for `ogg` format.
    - Handle different sample rates post-decoding and ensure it is compatible with the Whisper model.

## 4. Add LiveKit Plugin Code to the Repository

- **Objective**: Integrate the LiveKit plugin to support audio streams in LiveKit-supported formats.
  - **Action Items**:
    - Add the necessary LiveKit plugin code to this repository.
    - Ensure that LiveKit integration includes support for the `ogg` format and the necessary sampling rates (e.g., `48kHz`).
    - Test the LiveKit integration with the `/ws/` WebSocket functionality.
