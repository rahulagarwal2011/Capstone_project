"""Batch data loading for filesystem sources.

Loads documents from JSON/JSONL/text files and computes deterministic hashes
for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Doc:
    """A single document for processing.

    Attributes:
        id: Unique document identifier.
        text: Document text content.
        metadata: Optional metadata dict.
    """

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        """SHA-256 hash of the document text."""
        return hashlib.sha256(self.text.encode()).hexdigest()


def load_documents(path: Path | str, max_docs: int | None = None) -> list[Doc]:
    """Load documents from a file or directory.

    Supports JSONL (one JSON object per line with 'id' and 'text' fields)
    and plain text (one doc per file).

    Args:
        path: Path to a JSONL file or directory of text files.
        max_docs: Maximum number of documents to load. None for all.

    Returns:
        List of Doc objects.
    """
    path = Path(path)
    docs: list[Doc] = []

    if path.is_file() and path.suffix in (".jsonl", ".json"):
        docs = list(_load_jsonl(path, max_docs))
    elif path.is_dir():
        for f in sorted(path.glob("*.txt")):
            if max_docs and len(docs) >= max_docs:
                break
            text = f.read_text(encoding="utf-8")
            docs.append(Doc(id=f.stem, text=text))
    else:
        logger.warning("unsupported_path", path=str(path))

    dataset_hash = compute_dataset_hash(docs)
    logger.info("documents_loaded", count=len(docs), dataset_sha256=dataset_hash)
    return docs


def _load_jsonl(path: Path, max_docs: int | None) -> Iterator[Doc]:
    """Load documents from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            obj = json.loads(line)
            yield Doc(
                id=obj.get("id", str(i)),
                text=obj["text"],
                metadata=obj.get("metadata", {}),
            )


def compute_dataset_hash(docs: list[Doc]) -> str:
    """Compute a deterministic SHA-256 hash over all document texts.

    Used for reproducibility tracking — logged to W&B and results JSON.

    Args:
        docs: List of documents.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    hasher = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.id):
        hasher.update(doc.text.encode())
    return hasher.hexdigest()
