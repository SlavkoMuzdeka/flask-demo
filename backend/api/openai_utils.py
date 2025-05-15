import logging
import tiktoken

from openai import OpenAI
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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
