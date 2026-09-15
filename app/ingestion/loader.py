from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load a PDF file page-by-page.

    Each page becomes a separate document record so that
    page numbers can later be used for source citations.
    """
    records = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        records.append(
            {
                "text": text,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": "pdf",
                    "document_type": "pdf",
                    "page": page_number,
                },
            }
        )

    return records


def load_docx(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load text from a DOCX document.

    DOCX does not provide reliable page boundaries through
    python-docx, so the page field is set to None.
    """
    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    return [
        {
            "text": full_text,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "docx",
                "document_type": "docx",
                "page": None,
            },
        }
    ]


def load_txt(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load a plain text file.
    """
    text = file_path.read_text(encoding="utf-8")

    return [
        {
            "text": text,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "txt",
                "document_type": "txt",
                "page": None,
            },
        }
    ]


def load_document(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load one supported document.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix == ".docx":
        return load_docx(file_path)

    if suffix == ".txt":
        return load_txt(file_path)

    raise ValueError(
        f"Unsupported file type: {file_path.suffix}"
    )


def discover_documents(root_dir: Path) -> List[Path]:
    """
    Recursively discover all supported documents.
    """
    documents = []

    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file_path)

    return sorted(documents)


def load_all_documents(root_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover and load every supported document under root_dir.
    """
    all_records = []

    documents = discover_documents(root_dir)

    for file_path in documents:
        records = load_document(file_path)
        all_records.extend(records)

    return all_records