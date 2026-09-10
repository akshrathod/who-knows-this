import os
from pathlib import Path

import boto3


S3_BUCKET = os.getenv("S3_BUCKET_NAME", "talent-profiling-raw-docs")


def get_client():
    """Create an S3 client using boto3's normal credential chain."""
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def list_files(prefix: str = "raw/", bucket: str = S3_BUCKET) -> list[str]:
    keys = []
    paginator = get_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def download_file(key: str, destination: Path, bucket: str = S3_BUCKET) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    get_client().download_file(bucket, key, str(destination))
    return destination


def upload_file(local_path: Path, key: str, bucket: str = S3_BUCKET) -> None:
    get_client().upload_file(str(local_path), bucket, key)
