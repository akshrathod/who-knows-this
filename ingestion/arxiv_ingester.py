"""
Build a bounded research-paper corpus from arXiv.

This is the first data-acquisition step for the expert-discovery MVP. It keeps
scope explicit: collect papers for configured research topics, save PDFs under
data/raw/, and write a metadata registry that downstream parsing and graph
building can use as evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import arxiv
import boto3
from dotenv import load_dotenv

load_dotenv()


DEFAULT_CONFIG_PATH = Path("config/research_corpus_topics.json")
LOCAL_RAW_DIR = Path("data/raw")
REGISTRY_FILENAME = "registry.json"
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "talent-profiling-raw-docs")


@dataclass
class CorpusPaper:
    """Serializable metadata for one collected paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    pdf_url: str
    entry_id: str
    topic_name: str
    topic_query: str
    local_path: str
    s3_key: str | None = None
    downloaded: bool = False


def get_s3_client():
    """Create an S3 client from environment credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def load_topic_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load corpus topic configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Topic config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not config.get("topics"):
        raise ValueError(f"Topic config has no topics: {config_path}")
    return config


def sanitize_filename(value: str, max_length: int = 90) -> str:
    """Turn an arXiv title into a readable filesystem-safe stem."""
    keep = set("abcdefghijklmnopqrstuvwxyz0123456789_- ")
    cleaned = "".join(c if c.lower() in keep else "_" for c in value)
    return "_".join(cleaned.split())[:max_length].strip("_") or "untitled_paper"


def arxiv_id_from_entry(entry_id: str) -> str:
    """Extract the stable arXiv id from an entry URL."""
    return entry_id.rstrip("/").split("/")[-1]


def paper_to_record(paper, topic: dict, output_dir: Path) -> CorpusPaper:
    """Convert an arXiv result object into project registry metadata."""
    arxiv_id = arxiv_id_from_entry(paper.entry_id)
    filename = f"{sanitize_filename(paper.title)}__{arxiv_id.replace('/', '_')}.pdf"
    local_path = output_dir / filename

    return CorpusPaper(
        arxiv_id=arxiv_id,
        title=paper.title,
        authors=[author.name for author in paper.authors],
        abstract=paper.summary.replace("\n", " ").strip(),
        categories=list(paper.categories),
        primary_category=paper.primary_category,
        published=str(paper.published),
        updated=str(paper.updated),
        pdf_url=paper.pdf_url,
        entry_id=paper.entry_id,
        topic_name=topic["name"],
        topic_query=topic["query"],
        local_path=str(local_path),
    )


def search_topic(topic: dict, papers_per_topic: int) -> Iterable:
    """Yield arXiv results for one configured topic."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic["query"],
        max_results=papers_per_topic,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    yield from client.results(search)


def download_pdf(pdf_url: str, destination: Path) -> bool:
    """Download a PDF unless it already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(pdf_url, destination)
        return True
    except Exception as e:
        print(f"    download failed: {e}")
        return False


def upload_to_s3(s3, local_path: Path, s3_key: str) -> bool:
    """Upload one file to S3. Returns False instead of failing the corpus run."""
    try:
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        return True
    except Exception as e:
        print(f"    S3 upload failed: {e}")
        return False


def load_existing_registry(path: Path) -> dict[str, dict]:
    """Load previously collected papers keyed by arXiv id."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    keyed = {}
    for index, record in enumerate(records):
        key = record.get("arxiv_id")
        if not key and record.get("entry_id"):
            key = arxiv_id_from_entry(record["entry_id"])
            record["arxiv_id"] = key
        if not key and record.get("pdf_url"):
            key = arxiv_id_from_entry(record["pdf_url"].removesuffix(".pdf"))
            record["arxiv_id"] = key
        if not key:
            key = f"legacy-{index}-{sanitize_filename(record.get('title', 'untitled'))}"
        keyed[key] = record
    return keyed


def save_registry(path: Path, records: list[dict]) -> None:
    """Persist registry records in a stable order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(
        records,
        key=lambda item: (
            item.get("topic_name", "Legacy Corpus"),
            item.get("title", ""),
        ),
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def build_corpus(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = LOCAL_RAW_DIR,
    papers_per_topic: int | None = None,
    download: bool = True,
    upload_s3: bool = False,
    sleep_seconds: float = 0.5,
) -> list[dict]:
    """
    Search arXiv for configured topics and update data/raw/registry.json.

    Use download=False for a cheap metadata-only run. Use upload_s3=True only
    when AWS credentials and a bucket are configured.
    """
    config = load_topic_config(config_path)
    per_topic = papers_per_topic or int(config.get("papers_per_topic", 10))
    registry_path = output_dir / REGISTRY_FILENAME
    existing = load_existing_registry(registry_path)
    s3 = get_s3_client() if upload_s3 else None

    print(f"Building corpus from {len(config['topics'])} topics")
    print(f"Registry: {registry_path}")
    print(f"Papers per topic: {per_topic}")
    print(f"Download PDFs: {download}")
    print(f"Upload to S3: {upload_s3}\n")

    for topic in config["topics"]:
        print(f"Topic: {topic['name']}")
        for paper in search_topic(topic, per_topic):
            record = paper_to_record(paper, topic, output_dir)
            previous = existing.get(record.arxiv_id, {})

            local_path = Path(previous.get("local_path") or record.local_path)
            record.local_path = str(local_path)
            record.downloaded = bool(previous.get("downloaded", False))
            record.s3_key = previous.get("s3_key")

            print(f"  - {record.title[:80]}")

            if download:
                record.downloaded = download_pdf(record.pdf_url, local_path)

            if upload_s3 and s3 and record.downloaded:
                s3_key = f"raw/{local_path.name}"
                if upload_to_s3(s3, local_path, s3_key):
                    record.s3_key = s3_key

            existing[record.arxiv_id] = asdict(record)
            time.sleep(sleep_seconds)
        print()

    records = list(existing.values())
    save_registry(registry_path, records)

    if upload_s3 and s3 and registry_path.exists():
        upload_to_s3(s3, registry_path, f"raw/{REGISTRY_FILENAME}")

    downloaded_count = sum(1 for record in records if record.get("downloaded"))
    print(f"Corpus records: {len(records)}")
    print(f"Downloaded PDFs: {downloaded_count}")
    print(f"Saved registry: {registry_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded arXiv corpus for expert discovery.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_RAW_DIR)
    parser.add_argument("--papers-per-topic", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true", help="Collect metadata without downloading PDFs.")
    parser.add_argument("--upload-s3", action="store_true", help="Upload downloaded PDFs and registry to S3.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    build_corpus(
        config_path=args.config,
        output_dir=args.output_dir,
        papers_per_topic=args.papers_per_topic,
        download=not args.metadata_only,
        upload_s3=args.upload_s3,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
