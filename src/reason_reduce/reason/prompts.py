"""Versioned Jinja2 prompt templates for reasoning tasks.

All prompts live here — never inline in worker code.
Each template has a version string and produces deterministic output
given the same inputs.
"""

from __future__ import annotations

from jinja2 import Template

TEMPLATES: dict[str, dict[str, Template]] = {
    "ner": {
        "v1.0": Template(
            """You are an expert named entity recognition system.

Document:
{{ document_text }}

{% if context %}
Context: This document is part of a collection related to: {{ context }}
{% endif %}

Extract all named entities from the document. For each entity, provide:
- entity_type: one of [PERSON, ORGANIZATION, LOCATION, DRUG, DISEASE, GENE, CHEMICAL]
- entity_value: the exact text span
- confidence: your certainty (0.0 to 1.0)

Respond ONLY with valid JSON:
{"key": "<entity_type>", "value": "<entity_value>", "confidence": <float>, "trace": "<your reasoning>"}
""".strip()
        ),
    },
    "summarization": {
        "v1.0": Template(
            """You are an expert summarization system.

Document:
{{ document_text }}

{% if context %}
Context: This document is part of a collection related to: {{ context }}
{% endif %}

Provide a concise summary (2-3 sentences) capturing the key information.

Respond ONLY with valid JSON:
{"key": "summary", "value": "<summary text>", "confidence": <float>, "trace": "<your reasoning>"}
""".strip()
        ),
    },
    "classification": {
        "v1.0": Template(
            """You are an expert document classification system.

Document:
{{ document_text }}

{% if context %}
Context: This document is part of a collection related to: {{ context }}
{% endif %}

Classify this document into one of these categories: {{ categories | join(', ') }}

Respond ONLY with valid JSON:
{"key": "category", "value": "<category>", "confidence": <float>, "trace": "<your reasoning>"}
""".strip()
        ),
    },
    "relation_extraction": {
        "v1.0": Template(
            """You are an expert relation extraction system.

Document:
{{ document_text }}

{% if context %}
Context: This document is part of a collection related to: {{ context }}
{% endif %}

Extract relationships between entities. For each relationship, provide:
- subject: the source entity
- relation: the relationship type
- object: the target entity

Respond ONLY with valid JSON:
{"key": "<relation_type>", "value": "<subject> -> <object>", "confidence": <float>, "trace": "<your reasoning>"}
""".strip()
        ),
    },
}


def render_prompt(
    task_type: str,
    document_text: str,
    version: str = "v1.0",
    context: str | None = None,
    **kwargs: str,
) -> str:
    """Render a prompt template for a given task.

    Args:
        task_type: One of "ner", "summarization", "classification", "relation_extraction".
        document_text: The document text to process.
        version: Template version string.
        context: Optional partition context (from context propagator).
        **kwargs: Additional template variables.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If task_type or version is not found.
    """
    if task_type not in TEMPLATES:
        msg = f"Unknown task type: {task_type}. Available: {list(TEMPLATES.keys())}"
        raise KeyError(msg)

    versions = TEMPLATES[task_type]
    if version not in versions:
        msg = f"Unknown version {version} for task {task_type}. Available: {list(versions.keys())}"
        raise KeyError(msg)

    template = versions[version]
    return template.render(
        document_text=document_text[:4000],
        context=context,
        **kwargs,
    )
