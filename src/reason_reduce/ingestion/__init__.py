"""Data ingestion: batch loading, streaming, chunking, and schema inference."""

from reason_reduce.ingestion.batch import load_documents, Doc

__all__ = ["load_documents", "Doc"]
