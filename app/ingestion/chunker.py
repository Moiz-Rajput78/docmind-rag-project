from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving meaningful content.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces from each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove excessive blank lines
    cleaned_lines = []

    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    text = "\n".join(cleaned_lines)

    # Remove excessive spaces
    text = " ".join(text.split())

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text:
            Clean document text.

        chunk_size:
            Maximum number of characters per chunk.

        chunk_overlap:
            Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_chunks(
    records: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Clean and chunk loaded document records while preserving metadata.
    """

    chunked_records = []

    for record in records:
        original_text = record.get("text", "")

        cleaned_text = clean_text(original_text)

        chunks = chunk_text(
            cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_number, chunk in enumerate(chunks, start=1):
            metadata = dict(record.get("metadata", {}))

            metadata["chunk_number"] = chunk_number
            metadata["chunk_size"] = chunk_size
            metadata["chunk_overlap"] = chunk_overlap

            chunked_records.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return chunked_records