"""Semantic partitioning of documents for distributed processing.

Partitions documents into coherent groups using embedding similarity,
so that each Ray worker processes semantically related content.

Three strategies are implemented behind a common protocol:
    - AgglomerativePartitioner: greedy clustering on cosine similarity
    - SpectralPartitioner: spectral clustering on similarity graph Laplacian
    - HDBSCANPartitioner: HDBSCAN on UMAP-reduced embeddings

### Thesis Note
This is the first ablation target (Claim C1). Semantic partitioning
should preserve more intra-cluster coherence than byte-level or random
assignment, measurable via mean intra-cluster cosine similarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from reason_reduce.ingestion.batch import Doc
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_DIM = 768


@dataclass
class Partition:
    """A group of semantically related documents.

    Attributes:
        id: Partition identifier.
        docs: Documents in this partition.
        centroid: Mean embedding of partition members (L2-normalized).
        coherence_score: Mean intra-cluster cosine similarity.
    """

    id: int
    docs: list[Doc]
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(_EMBEDDING_DIM))
    coherence_score: float = 0.0


class PartitionerStrategy(ABC):
    """Protocol for partitioning strategies.

    Each strategy takes document embeddings and returns cluster assignments.
    """

    @abstractmethod
    def fit_predict(
        self, embeddings: np.ndarray, n_partitions: int, seed: int
    ) -> np.ndarray:
        """Assign documents to partitions based on embeddings.

        Args:
            embeddings: (n_docs, embedding_dim) array.
            n_partitions: Target number of partitions.
            seed: Random seed for reproducibility.

        Returns:
            Array of cluster labels (shape: n_docs).
        """
        ...


class AgglomerativePartitioner(PartitionerStrategy):
    """Greedy agglomerative clustering on cosine similarity.

    Uses average linkage on a precomputed cosine distance matrix.
    Works well for balanced cluster sizes and moderate n_docs.
    """

    def fit_predict(
        self, embeddings: np.ndarray, n_partitions: int, seed: int
    ) -> np.ndarray:
        distance_matrix = 1.0 - cosine_similarity(embeddings)
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

        clusterer = AgglomerativeClustering(
            n_clusters=n_partitions,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(distance_matrix)
        return labels


class SpectralPartitioner(PartitionerStrategy):
    """Spectral clustering on the SBERT cosine similarity graph.

    Constructs an affinity matrix from cosine similarity, then applies
    spectral clustering via the graph Laplacian. Better for non-convex
    cluster shapes than agglomerative.
    """

    def fit_predict(
        self, embeddings: np.ndarray, n_partitions: int, seed: int
    ) -> np.ndarray:
        affinity = cosine_similarity(embeddings)
        affinity = np.clip(affinity, 0.0, 1.0)
        np.fill_diagonal(affinity, 1.0)

        clusterer = SpectralClustering(
            n_clusters=n_partitions,
            affinity="precomputed",
            random_state=seed,
            assign_labels="kmeans",
        )
        labels = clusterer.fit_predict(affinity)
        return labels


class HDBSCANPartitioner(PartitionerStrategy):
    """HDBSCAN on UMAP-reduced embeddings.

    For irregular cluster shapes and when n_partitions is unknown.
    Falls back to agglomerative if HDBSCAN finds fewer clusters than requested.
    """

    def fit_predict(
        self, embeddings: np.ndarray, n_partitions: int, seed: int
    ) -> np.ndarray:
        try:
            from umap import UMAP
            from hdbscan import HDBSCAN

            reduced = UMAP(
                n_components=min(50, embeddings.shape[1]),
                random_state=seed,
                metric="cosine",
            ).fit_transform(embeddings)

            clusterer = HDBSCAN(
                min_cluster_size=max(2, len(embeddings) // (n_partitions * 2)),
                min_samples=2,
            )
            labels = clusterer.fit_predict(reduced)

            if len(set(labels) - {-1}) < n_partitions:
                logger.info("hdbscan_fallback", reason="too_few_clusters")
                return AgglomerativePartitioner().fit_predict(embeddings, n_partitions, seed)

            return labels

        except ImportError:
            logger.warning("hdbscan_unavailable", fallback="agglomerative")
            return AgglomerativePartitioner().fit_predict(embeddings, n_partitions, seed)


_STRATEGIES: dict[str, PartitionerStrategy] = {
    "semantic": AgglomerativePartitioner(),
    "spectral": SpectralPartitioner(),
    "hdbscan": HDBSCANPartitioner(),
}


def _embed_documents(docs: list[Doc]) -> np.ndarray:
    """Embed documents using Sentence-BERT.

    Falls back to TF-IDF hashing if sentence-transformers is unavailable
    (e.g., in CI without large model downloads).

    Args:
        docs: Documents to embed.

    Returns:
        (n_docs, embedding_dim) numpy array.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        texts = [doc.text[:512] for doc in docs]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings)

    except ImportError:
        logger.info("sbert_unavailable", fallback="tfidf_hash")
        return _tfidf_fallback(docs)


def _tfidf_fallback(docs: list[Doc]) -> np.ndarray:
    """Lightweight TF-IDF hash embedding fallback (no GPU, no large models)."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=_EMBEDDING_DIM, alternate_sign=False)
    texts = [doc.text[:512] for doc in docs]
    matrix = vectorizer.fit_transform(texts).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _compute_coherence(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Compute mean intra-cluster cosine similarity per cluster."""
    coherence: dict[int, float] = {}
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        if len(cluster_embeddings) < 2:
            coherence[label] = 1.0
            continue
        sim_matrix = cosine_similarity(cluster_embeddings)
        n = len(cluster_embeddings)
        upper_sum = (sim_matrix.sum() - n) / (n * (n - 1))
        coherence[label] = float(upper_sum)

    return coherence


def partition_documents(
    docs: list[Doc],
    n_partitions: int | None = None,
    strategy: str = "semantic",
    seed: int = 42,
) -> list[Partition]:
    """Partition documents into semantically coherent groups.

    Args:
        docs: Documents to partition.
        n_partitions: Number of partitions. None = auto (len(docs) // 100, min 2).
        strategy: Partitioning algorithm ("semantic", "spectral", "hdbscan", "random").
        seed: Random seed for reproducibility.

    Returns:
        List of Partition objects with centroids and coherence scores.
    """
    if not docs:
        return []

    if n_partitions is None:
        n_partitions = max(1, len(docs) // 100)

    n_partitions = min(n_partitions, len(docs))

    if n_partitions >= len(docs):
        return [
            Partition(id=i, docs=[doc], coherence_score=1.0)
            for i, doc in enumerate(docs)
        ]

    if strategy == "random":
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, n_partitions, size=len(docs))
        embeddings = None
    elif len(docs) < 4 or n_partitions < 2:
        labels = np.array([i % n_partitions for i in range(len(docs))])
        embeddings = None
    else:
        embeddings = _embed_documents(docs)
        partitioner = _STRATEGIES.get(strategy)
        if partitioner is None:
            logger.warning("unknown_strategy", strategy=strategy, fallback="semantic")
            partitioner = _STRATEGIES["semantic"]
        labels = partitioner.fit_predict(embeddings, n_partitions, seed)

    if embeddings is None:
        embeddings = _tfidf_fallback(docs)

    coherence_scores = _compute_coherence(embeddings, labels)

    partitions: list[Partition] = []
    unique_labels = sorted(set(labels) - {-1})
    for pid, label in enumerate(unique_labels):
        mask = labels == label
        partition_docs = [docs[i] for i in range(len(docs)) if mask[i]]
        partition_embeddings = embeddings[mask]

        centroid = partition_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        partitions.append(
            Partition(
                id=pid,
                docs=partition_docs,
                centroid=centroid,
                coherence_score=coherence_scores.get(label, 0.0),
            )
        )

    noise_mask = labels == -1
    if noise_mask.any():
        noise_docs = [docs[i] for i in range(len(docs)) if noise_mask[i]]
        if partitions:
            partitions[0].docs.extend(noise_docs)
        else:
            partitions.append(Partition(id=0, docs=noise_docs))

    mean_coherence = (
        np.mean([p.coherence_score for p in partitions]) if partitions else 0.0
    )

    logger.info(
        "partitioning_complete",
        n_docs=len(docs),
        n_partitions=len(partitions),
        strategy=strategy,
        mean_coherence=round(float(mean_coherence), 4),
    )
    return partitions
