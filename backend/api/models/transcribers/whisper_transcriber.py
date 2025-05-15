import os
import math
import logging

from openai import OpenAI
from pydub import AudioSegment

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


class Whisper_Transcriber:
    """
    A class for transcribing audio files using OpenAI's Whisper model.

    Attributes:
        config (dict): Configuration dictionary containing settings like model type and debug mode.
        debug (bool): Flag to enable or disable debugging logs.
        model (whisper.Whisper): Loaded Whisper model for transcription.
    """

    def __init__(self, config: dict):
        """
        Initializes the WhisperTranscriber with a specific model size.

        Args:
            config (dict): Configuration settings, including 'WHISPER_MODEL' and 'DEBUG'.
        """
        self.debug = config.get("debug", False)
        self.config = config.get("whisper", {})
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, audio_path: str, video_id: str) -> str:
        base_dir = os.path.join(
            os.getcwd(),
            self.config.get("downloads_dir", "downloads"),
            video_id,
        )
        transcript_path = os.path.join(
            base_dir,
            f"{video_id}{self.config.get('transcription_extension', '.txt')}",
        )

        if self.debug and os.path.exists(transcript_path):
            logger.info("Transcription already exists.")
            with open(transcript_path, "r", encoding="utf-8") as file:
                return file.read()

        if self.debug:
            logger.info("Starting transcription...")

        transcribed_text = ""

        if os.path.getsize(audio_path) <= MAX_FILE_SIZE_BYTES:
            # File is within size limit, process directly
            with open(audio_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="json"
                )
                transcribed_text = result.text
        else:
            if self.debug:
                size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                logger.info(
                    f"Audio file exceeds 25MB ({size_mb:.2f} MB), splitting into chunks..."
                )

            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            estimated_size_per_ms = os.path.getsize(audio_path) / duration_ms
            chunk_duration_ms = int(MAX_FILE_SIZE_BYTES / estimated_size_per_ms)

            chunks = math.ceil(duration_ms / chunk_duration_ms)

            os.makedirs(base_dir, exist_ok=True)

            for i in range(chunks):
                start_ms = i * chunk_duration_ms
                end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]

                chunk_path = os.path.join(
                    base_dir, f"{video_id}_{i+1}{self.config.get('mp3_ext', '.mp3')}"
                )

                chunk.export(chunk_path, format="mp3")

                with open(chunk_path, "rb") as audio_file:
                    result = self.client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file, response_format="json"
                    )
                    transcribed_text += result.text

                if self.debug:
                    logger.info(f"Processed chunk {i + 1} of {chunks}")

                os.remove(chunk_path)

        if self.debug:
            with open(transcript_path, "w", encoding="utf-8") as file:
                file.write(transcribed_text)
            logger.info(f"Transcript saved at: {transcript_path}")

        return transcribed_text
