import os
import json
import logging

from typing import Tuple
from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)


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
