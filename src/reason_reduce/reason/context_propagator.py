"""Cross-partition context propagation.

Provides partition-level context summaries to workers so they understand
the semantic neighborhood of their assigned data.

TODO[Phase-2]: Full implementation with embedding similarity lookup.
"""

from __future__ import annotations

from reason_reduce.reason.partitioner import Partition
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


def get_partition_context(partition: Partition, all_partitions: list[Partition]) -> str:
    """Generate a context preamble for a partition's workers.

    Summarizes what the partition contains and its relationship to
    other partitions, so the LLM has domain context.

    TODO[Phase-2]: Use small-LLM call to generate natural language summary.

    Args:
        partition: The target partition.
        all_partitions: All partitions for similarity computation.

    Returns:
        A context string to prepend to prompts.
    """
    n_docs = len(partition.docs)
    sample_texts = [doc.text[:100] for doc in partition.docs[:3]]
    summary = f"Processing partition {partition.id} with {n_docs} documents. "
    summary += f"Sample content: {'; '.join(sample_texts)}"
    return summary
