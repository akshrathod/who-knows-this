from pathlib import Path

import docx
import fitz


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_document(file_path: Path) -> str:
    """Extract text from a supported document."""
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return file_path.read_text(encoding="utf-8")

    if extension == ".docx":
        document = docx.Document(str(file_path))
        return "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())

    if extension == ".pdf":
        with fitz.open(str(file_path)) as document:
            pages = [page.get_text("text").strip() for page in document]
        return "\n\n".join(page for page in pages if page)

    raise ValueError(f"Unsupported file type: {extension}")
