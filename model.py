# Function to convert segments to dictionaries
from faster_whisper.transcribe import Segment, Word



# Function to dump a Word instance to a dictionary
def word_to_dict(word: Word) -> dict:
    return {
        "start": word.start,
        "end": word.end,
        "word": word.word,
        "probability": word.probability
    }

# Function to load a Word instance from a dictionary
def dict_to_word(data: dict) -> Word:
    return Word(
        start=data["start"],
        end=data["end"],
        word=data["word"],
        probability=data["probability"]
    )

# Function to dump a Segment instance to a dictionary
def segment_to_dict(segment: Segment) -> dict:
    return {
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob,
        "words": [word_to_dict(word) for word in segment.words] if segment.words else None
    }

# Function to load a Segment instance from a dictionary
def dict_to_segment(data: dict) -> Segment:
    return Segment(
        id=data["id"],
        seek=data["seek"],
        start=data["start"],
        end=data["end"],
        text=data["text"],
        tokens=data["tokens"],
        temperature=data["temperature"],
        avg_logprob=data["avg_logprob"],
        compression_ratio=data["compression_ratio"],
        no_speech_prob=data["no_speech_prob"],
        words=[dict_to_word(word) for word in data["words"]] if data["words"] else None
    )