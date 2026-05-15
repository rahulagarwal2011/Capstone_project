"""Semantic chunking via Sentence-BERT.

TODO[Phase-2]: Implement semantic-aware document chunking that preserves
context boundaries for better LLM processing.
"""

from __future__ import annotations

from reason_reduce.ingestion.batch import Doc
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


def chunk_document(doc: Doc, max_chunk_tokens: int = 512) -> list[Doc]:
    """Split a document into semantically coherent chunks.

    TODO[Phase-2]: Replace naive splitting with SBERT-based boundary detection.

    Args:
        doc: Input document.
        max_chunk_tokens: Approximate max tokens per chunk.

    Returns:
        List of Doc chunks (child docs with parent reference in metadata).
    """
    words = doc.text.split()
    approx_tokens_per_word = 1.3
    words_per_chunk = int(max_chunk_tokens / approx_tokens_per_word)

    chunks: list[Doc] = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Doc(
                id=f"{doc.id}_chunk_{len(chunks)}",
                text=chunk_text,
                metadata={"parent_id": doc.id, "chunk_index": str(len(chunks))},
            )
        )

    return chunks
