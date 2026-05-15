"""Data ingestion: batch loading, streaming, chunking, and schema inference."""

from reason_reduce.ingestion.batch import Doc, load_documents

__all__ = ["load_documents", "Doc"]
