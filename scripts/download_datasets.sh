#!/bin/bash
# Download benchmark datasets for Reason-Reduce evaluation.
# Usage: bash scripts/download_datasets.sh

set -euo pipefail

DATASETS_DIR="benchmarks/datasets"

echo "=== Reason-Reduce Dataset Downloader ==="

# PubMed sample (10k abstracts)
echo "[1/3] PubMed sample..."
mkdir -p "$DATASETS_DIR/pubmed"
# TODO[Phase-5]: Add actual download from PubMed baseline
echo '{"id": "sample_1", "text": "Sample PubMed abstract for testing."}' > "$DATASETS_DIR/pubmed/sample.jsonl"
echo "  → Placeholder created. Replace with real download in Phase 5."

# SEC 10-K filings
echo "[2/3] SEC filings..."
mkdir -p "$DATASETS_DIR/sec"
echo '{"id": "sec_1", "text": "Sample SEC 10-K filing excerpt."}' > "$DATASETS_DIR/sec/sample.jsonl"
echo "  → Placeholder created."

# Common Crawl subset
echo "[3/3] Common Crawl subset..."
mkdir -p "$DATASETS_DIR/common_crawl"
echo '{"id": "cc_1", "text": "Sample Common Crawl document."}' > "$DATASETS_DIR/common_crawl/sample.jsonl"
echo "  → Placeholder created."

echo ""
echo "Done. Dataset SHA-256 hashes will be computed at load time."
