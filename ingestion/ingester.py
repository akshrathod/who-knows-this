import tempfile
from pathlib import Path

from ingestion.document_parser import SUPPORTED_EXTENSIONS, parse_document
from storage.s3_storage import download_file, list_files


def _build_document(path: Path, text: str, source: str) -> dict:
    return {
        "doc_id": path.stem,
        "filename": path.name,
        "source": source,
        "text": text,
        "char_count": len(text),
    }


def ingest_file(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return [_build_document(path, parse_document(path), "local_file")]


def ingest_folder(folder_path: str) -> list[dict]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise ValueError(f"No supported files found in: {folder_path}")
    return [_build_document(path, parse_document(path), "local_folder") for path in files]


def ingest_s3(prefix: str = "raw/", limit: int | None = None) -> list[dict]:
    keys = [key for key in list_files(prefix) if Path(key).suffix.lower() in SUPPORTED_EXTENSIONS]
    if limit is not None:
        keys = keys[:limit]

    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, key in enumerate(keys):
            # The index prevents equal filenames in different S3 folders from colliding locally.
            path = Path(temp_dir) / f"{index}_{Path(key).name}"
            download_file(key, path)
            document = _build_document(path, parse_document(path), "s3")
            document["doc_id"] = Path(key).stem
            document["filename"] = Path(key).name
            document["s3_key"] = key
            documents.append(document)
    return documents


def ingest(source: str = "s3", path: str | None = None, limit: int | None = None) -> list[dict]:
    """Ingest mixed PDF, DOCX, and TXT documents from S3 or locally."""
    if source == "s3":
        return ingest_s3(prefix=path or "raw/", limit=limit)
    if source == "file" and path:
        return ingest_file(path)
    if source == "folder" and path:
        return ingest_folder(path)
    raise ValueError("Use source='s3', 'file', or 'folder'; file and folder require a path")
