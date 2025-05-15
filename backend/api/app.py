import os
import json
import math
import logging
import requests
import tiktoken
import feedparser

from openai import OpenAI
from flask_cors import CORS
from yt_dlp import YoutubeDL
from datetime import datetime
from dotenv import load_dotenv
from pydub import AudioSegment
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify


MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


# Load env & config
load_dotenv(override=True)
with open("config.json") as f:
    config = json.load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Helper functions
def get_chat_completion(client: OpenAI, messages: List[dict], model: str) -> str:
    """
    Calls the OpenAI API to generate a response based on given messages.

    Parameters:
    - client (OpenAI): OpenAI client instance.
    - messages (List[dict]): List of messages in the format required for OpenAI's API.
    - model (str): The model to be used for the API call.

    Returns:
    - str: The generated response from OpenAI.
    """
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    choices = response.choices

    if choices:
        return choices[0].message.content
    else:
        logger.error("No choices returned from OpenAI API.")
        raise RuntimeError("Failed to retrieve a summary from OpenAI API.")


def chunk_on_delimiter(
    text: str, max_tokens: int, delimiter: str, debug: bool
) -> List[str]:
    """
    Splits a given text into smaller chunks based on a specified delimiter.

    Parameters:
    - text (str): The text to be split.
    - max_tokens (int): Maximum token count per chunk.
    - delimiter (str): The delimiter used to split the text.
    - debug (bool): Whether to log debugging information.

    Returns:
    - List[str]: A list of text chunks.
    """
    chunks = text.split(delimiter)
    combined_chunks, dropped_chunk_count = _combine_chunks_with_no_minimum(
        chunks,
        max_tokens,
        chunk_delimiter=delimiter,
        add_ellipsis_for_overflow=True,
        debug=debug,
    )
    if dropped_chunk_count > 0 and debug:
        logger.warning(f"{dropped_chunk_count} chunks were dropped due to overflow")

    # Ensure each chunk ends with the delimiter
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


def _combine_chunks_with_no_minimum(
    chunks: List[str],
    max_tokens: int,
    chunk_delimiter="\n\n",
    header: Optional[str] = None,
    add_ellipsis_for_overflow=False,
    debug: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Combines small text chunks into larger chunks without exceeding the maximum token limit.

    Parameters:
    - chunks (List[str]): List of text chunks.
    - max_tokens (int): Maximum allowed tokens per chunk.
    - chunk_delimiter (str, optional): Delimiter used to join chunks. Defaults to "\n\n".
    - header (Optional[str], optional): Optional header to be added at the start of each chunk.
    - add_ellipsis_for_overflow (bool, optional): Whether to add "..." if a chunk is too large.
    - debug (bool, optional): Whether to enable debugging logs.

    Returns:
    - Tuple[List[str], int]: A tuple containing the list of combined chunks and the count of dropped chunks.
    """
    dropped_chunk_count = 0
    output, candidate_indices = [], []
    candidate = [] if header is None else [header]

    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]

        if num_tokens_from_text(chunk_delimiter.join(chunk_with_header)) > max_tokens:
            if debug:
                logger.warning(f"Chunk overflow")
            if (
                add_ellipsis_for_overflow
                and num_tokens_from_text(chunk_delimiter.join(candidate + ["..."]))
                <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # Skip this chunk as it exceeds max tokens

        extended_candidate_token_count = num_tokens_from_text(
            chunk_delimiter.join(candidate + [chunk])
        )

        # If adding this chunk exceeds max_tokens, save the candidate and start a new one
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            candidate = chunk_with_header  # Reset candidate
            candidate_indices = [chunk_i]
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)

    # Add any remaining candidate chunks
    if (header is not None and len(candidate) > 1) or (
        header is None and len(candidate) > 0
    ):
        output.append(chunk_delimiter.join(candidate))

    return output, dropped_chunk_count


def num_tokens_from_text(text: str) -> int:
    """
    Computes the number of tokens in a given text string.

    Parameters:
    - text (str): Input text.

    Returns:
    - int: The estimated token count.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


########################

# Clases


class YT_Downloader:
    """Downloads YouTube videos as MP3 and retrieves metadata."""

    def __init__(self, config: dict):
        self.debug = config.get("debug", False)
        self.config = config.get("youtube", {})

    def download_episode(
        self, source_url: str, episode_name: str | None
    ) -> Tuple[str, dict]:
        """Downloads both the MP3 file and metadata, then logs key details."""
        self.source_url = source_url.split("&")[0]
        self.video_id = self.source_url.split("=")[-1]

        mp3_path = self._download_mp3()
        metadata = self._download_metadata()

        return mp3_path, metadata

    def _download_mp3(self) -> str:
        """Downloads the video as an MP3 file."""
        return self._download_file(
            extension=self.config.get("mp3_ext", ".mp3"), audio_only=True
        )

    def _download_metadata(self) -> dict:
        """Downloads video metadata as a JSON file and returns its contents."""
        metadata_path = self._download_file(
            extension=self.config.get("metadata_ext", ".info.json")
        )

        try:
            with open(metadata_path, "r", encoding="utf8") as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Failed to read metadata file: {e}")
            return {}

    def _download_file(self, extension: str, audio_only: bool = False) -> str:
        """Handles downloading the requested file type."""
        output_path = os.path.join(
            os.getcwd(),
            self.config.get("downloads_dir"),
            self.video_id,
            f"{self.video_id}{extension}",
        )

        if self.debug:
            if os.path.exists(output_path):
                logger.info(f"File already exists ({extension}).")
                return output_path

        with YoutubeDL(
            self._get_ydl_opts(self.config.get("downloads_dir"), audio_only)
        ) as ydl:
            try:
                ydl.download([self.source_url])
            except Exception as e:
                logger.error(f"Failed to download {extension}: {e}")
                raise

        if self.debug:
            logger.info(f"Successfully downloaded file ({extension}).")

        return output_path

    def _get_ydl_opts(self, output_dir: str, audio_only: bool = False) -> dict:
        """
        Generates configuration options for yt-dlp based on download requirements.

        This function prepares options for yt-dlp, specifying output format,
        download type (audio or metadata), and necessary post-processing steps.

        Args:
            output_dir (str): The directory where the downloaded files should be saved.
            audio_only (bool, optional): Whether to download only audio. Defaults to False.

        Returns:
            dict: The yt-dlp configuration options.
        """
        opts = {
            "outtmpl": os.path.join(output_dir, "%(id)s", "%(id)s.%(ext)s"),
        }

        if audio_only:
            opts.update(
                {
                    "format": "bestaudio/best",
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192",
                        }
                    ],
                }
            )
            return opts
        opts.update(
            {
                "skip_download": True,
                "writeinfojson": True,
            }
        )
        return opts


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


class RSS_Feed_Downloader:
    """
    A class for downloading podcast episodes from an RSS feed.

    Attributes:
        config (dict): Configuration settings, including debug mode.
        debug (bool): Flag indicating whether debug logging is enabled.
    """

    def __init__(self, config: dict):
        """
        Initializes the RSS_Feed_Downloader with the given configuration.

        Parameters:
            config (dict): Configuration dictionary, where "DEBUG" can be set to True for logging.
        """
        self.debug = config.get("debug", False)
        self.config = config.get("rss_feed", {})

    def download_episode(self, source_url: str, episode_name: str | None) -> str:
        """
        Downloads a podcast episode from the given RSS feed URL.

        Parameters:
            source_url (str): The URL of the RSS feed.
            episode_name (str | None): The name of the episode to download. If None, defaults to the latest episode.

        Returns:
            tuple: (file_path (str), episode_name (str), episode_id (str)) if successful.

        Raises:
            ValueError: If the episode is not found or no audio file is available.
        """
        # Retrieve episode details
        entry, channel_name = self._get_episode_entry(source_url, episode_name)

        if not entry:
            raise ValueError("Episode not found. Please check the episode name.")

        if "enclosures" not in entry and not entry.enclosures:
            raise ValueError("No audio enclosure available.")

        # Extract episode URL and generate filename
        mp3_url = entry.enclosures[0].href
        episode_id = mp3_url.split("/")[-1].split(".")[0]

        output_dir = os.path.join(
            os.getcwd(), self.config.get("downloads_dir", "downloads"), episode_id
        )
        os.makedirs(os.path.join(output_dir, episode_id), exist_ok=True)
        file_path = os.path.join(
            output_dir, episode_id, episode_id + self.config.get("mp3_ext", ".mp3")
        )

        metadata = self._get_metadata(entry)
        metadata["id"] = episode_id
        metadata["channel"] = channel_name

        if self.debug and os.path.exists(file_path):
            logger.info("Episode already downloaded.")
            return file_path, metadata

        # Download the episode in chunks
        response = requests.get(mp3_url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=self.config.get("chunk_size", 8192)
            ):
                file.write(chunk)

        if self.debug:
            logger.info("Successfully downloaded episode.")

        return file_path, metadata

    def _get_episode_entry(
        self, source_url: str, episode_name: str
    ) -> Tuple[dict, str] | None:
        """
        Retrieves an episode entry from an RSS feed.

        Parameters:
            source_url (str): The URL of the RSS feed.
            episode_name (str): The name of the episode to find.

        Returns:
            tuple: (entry (dict), channel_name (str)) if found, otherwise None.
        """
        feed = feedparser.parse(source_url)
        channel_name = feed.get("channel", {}).get("title", "")
        for entry in feed.entries:
            if episode_name.lower() == entry.title.lower():
                return entry, channel_name
        return None, None

    def _get_metadata(self, entry: dict) -> dict:
        """
        Extracts metadata from an RSS feed entry.

        Parameters:
            entry (dict): The RSS feed entry.

        Returns:
            dict: A dictionary containing the episode metadata.
        """
        raw_duration = entry.get("itunes_duration")
        if raw_duration and raw_duration.isdigit():
            duration_string = self._format_duration(int(raw_duration))
        else:
            duration_string = raw_duration

        published_parsed = entry.get("published_parsed")
        dt = datetime(*published_parsed[:6])
        date_str = dt.strftime("%Y-%m-%d")

        return {
            "title": entry.get("title", ""),
            "thumbnail": entry.get("image", {}).get("href"),
            "duration_string": duration_string,
            "release_date": date_str,
        }

    def _format_duration(self, seconds: int) -> str:
        """
        Formats a duration in seconds into a human-readable string.

        Parameters:
            seconds (int): The duration in seconds.

        Returns:
            str: The formatted duration string.
        """
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes}:{secs:02}"


class OpenAI_Summarizer:
    """
    A class for summarizing podcast transcripts using OpenAI's GPT models.
    """

    DEFAULT_SYSTEM_PROMPT = """
        # Role and Objective
        You are a seasoned podcast transcript summarization expert charged with distilling a full transcript into concise, interconnected summaries.

        # Instructions
        1. The transcript is divided into labeled segments using markers like `--- Chunk 1 ---`. Detect each boundary clearly.  
        2. For *each* chunk, compose **exactly one or two** bullet points—no more, no fewer—capturing key insights, tone, and notable quotes.  
        3. Ensure bullets build on one another to preserve narrative flow; use brief transitions (e.g., “**Building on this…**”, “Subsequently…”).  
        4. Do not add, remove, or reorder chunks: generate exactly N bullets for N chunks, in sequential order.  
        5. Format your response in Markdown:
            - Use `- ` for bullets.
            - **Bold** to highlight the key takeaway in each bullet.
            - *Italics* for nuance or tone.
            - Inline code (``) for any quoted text or technical terms.


        # Reasoning Steps
        a. Parse all '--- Chunk X ---' markers to segment the transcript.  
        b. For chunk X, isolate core ideas, then compose a five-sentence summary that may recall earlier context for cohesion.  
        c. Maintain a positive, engaging tone throughout.

        # Final Instruction
        Now, think step by step, review all labeled chunks, and output the bullet-point summaries exactly as specified in the instructions.
        """

    def __init__(self, config: dict):
        """
        Initializes the OpenAI Summarizer.

        Parameters:
        - config (dict): Configuration dictionary containing settings, including whether debugging is enabled.
        """
        self.debug = config.get("debug", False)
        self.config = config.get("openai", {})
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def summarize(
        self,
        text: str,
        detail: float = 0,
        minimum_chunk_size: Optional[int] = 500,
        chunk_delimiter: str = ".",
    ):
        """
        Summarizes a given text by splitting it into chunks and summarizing each individually.

        Parameters:
        - text (str): The text to be summarized.
        - detail (float, optional): Value between 0 and 1 indicating the level of detail (0 = highly summarized, 1 = detailed). Defaults to 0.
        - minimum_chunk_size (Optional[int], optional): Minimum chunk size for splitting text. Defaults to 500 tokens.
        - chunk_delimiter (str, optional): Delimiter used to split the text into chunks. Defaults to ".".

        Returns:
        - str: The final compiled summary of the text.
        """
        # Ensure detail value is within valid range
        assert 0 <= detail <= 1

        # Determine number of chunks dynamically based on the desired detail level
        min_chunks = 1
        max_chunks = len(
            chunk_on_delimiter(
                text=text,
                max_tokens=minimum_chunk_size,
                delimiter=chunk_delimiter,
                debug=self.debug,
            )
        )
        num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

        # Calculate chunk size based on total document length and target chunk count
        document_length = num_tokens_from_text(text)
        chunk_size = max(minimum_chunk_size, document_length // num_chunks)
        text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter, self.debug)

        if self.debug:
            logger.info(f"Total tokens in document: {document_length}")
            logger.info(
                f"Splitting the text into {len(text_chunks)} chunks to be summarized."
            )
            logger.info(
                f"Chunk lengths are {[num_tokens_from_text(x) for x in text_chunks]}"
            )

        labeled = []
        for idx, chunk in enumerate(text_chunks, start=1):
            labeled.append(f"--- Chunk {idx} ---\n{chunk.strip()}")
        query = "\n\n".join(labeled)

        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}"},
        ]

        return get_chat_completion(
            self.client, messages, self.config.get("model", "gpt-3.5-turbo")
        )


#########################


yt_downloader = YT_Downloader(config=config)
summarizer = OpenAI_Summarizer(config=config)
transcriber = Whisper_Transcriber(config=config)
rss_downloader = RSS_Feed_Downloader(config=config)

app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URL")])


@app.route("/api/summarize", methods=["POST"])
def summarize_endpoint():
    """
    Expects JSON with:
      - source_url: str
      - episode_name: str | null
      - detail_level: float (0.0–1.0)
      - platform: "youtube" or "rss"
    Returns JSON with:
      - success: bool
      - summary: str (if success)
      - error: str (if not)
    """
    data = request.get_json()
    source_url = data.get("source_url")
    episode_name = data.get("episode_name")
    detail_level = data.get("detail_level", 0.0)
    platform = data.get("platform")

    if platform == "youtube":
        downloader = yt_downloader
    else:
        downloader = rss_downloader

    try:
        # 1) Download
        mp3_path, metadata = downloader.download_episode(source_url, episode_name)
        logger.info(f"Downloaded {metadata.get('title', '')}")

        # 2) Transcribe
        text = transcriber.transcribe(
            audio_path=mp3_path, video_id=metadata.get("id", "")
        )
        logger.info("Transcription complete")

        # 3) Summarize
        summary = summarizer.summarize(text, detail=detail_level)
        logger.info("Summarization complete")

        return (
            jsonify(
                {
                    "success": True,
                    "title": metadata.get("title", ""),
                    "summary": summary,
                    "thumbnail": metadata.get("thumbnail", ""),
                    "channel": metadata.get("channel", ""),
                    "duration_string": metadata.get("duration_string", ""),
                    "release_date": metadata.get("release_date", ""),
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception("Error in /summarize")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def hello():
    return "Backend app is running...", 200


if __name__ == "__main__":
    app.run()
