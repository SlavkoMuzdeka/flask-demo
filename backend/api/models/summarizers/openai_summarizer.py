import os
import logging

from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
from utils.openai_utils import (
    chunk_on_delimiter,
    get_chat_completion,
    num_tokens_from_text,
)

load_dotenv(override=True)

logger = logging.getLogger(__name__)


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
