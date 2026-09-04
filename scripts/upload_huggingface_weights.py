"""Validate and publish the frozen thesis checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "docs" / "huggingface-weights-manifest.csv"
MODEL_CARD_PATH = PROJECT_ROOT / "docs" / "huggingface-model-card.md"
NOTICE_PATH = PROJECT_ROOT / "ACADEMIC_USE_NOTICE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_entries() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))

    if not entries:
        raise ValueError("The checkpoint manifest is empty.")

    repository_paths: set[str] = set()
    for entry in entries:
        source = PROJECT_ROOT / entry["source_path"]
        repository_path = entry["repository_path"]
        if repository_path in repository_paths:
            raise ValueError(f"Duplicate repository path: {repository_path}")
        repository_paths.add(repository_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Unexpected size for {source}")
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {source}")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish the validated thesis checkpoints to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repository, for example USER/heat-treatment-world-model.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update a private model repository instead of a public one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and hashes without contacting Hugging Face Hub.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = validated_entries()
    total_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    print(f"Validated {len(entries)} checkpoints ({total_bytes} bytes).")
    if args.dry_run:
        return

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub or run with: "
            "uv run --with huggingface_hub python scripts/upload_huggingface_weights.py "
            "--repo-id USER/heat-treatment-world-model"
        ) from exc

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=MODEL_CARD_PATH),
        CommitOperationAdd(
            path_in_repo="weights-manifest.csv", path_or_fileobj=MANIFEST_PATH
        ),
        CommitOperationAdd(
            path_in_repo="ACADEMIC_USE_NOTICE.md", path_or_fileobj=NOTICE_PATH
        ),
    ]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=entry["repository_path"],
            path_or_fileobj=PROJECT_ROOT / entry["source_path"],
        )
        for entry in entries
    )
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Publish frozen thesis model checkpoints",
    )
    print(commit.commit_url)


if __name__ == "__main__":
    main()
