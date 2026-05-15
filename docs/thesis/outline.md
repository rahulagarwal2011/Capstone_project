# Thesis Outline

## Chapter 1: Introduction (5–7 pages)
- Problem statement: 80% of enterprise data is unstructured; GenAI processing is siloed from data infra
- Hypothesis: embedding LLM reasoning directly in the data plane yields cost/latency/scalability wins
- Contributions (3–5 bullet points)
- Thesis roadmap

## Chapter 2: Background & Related Work (10–12 pages)
- MapReduce → Spark lineage (Dean 2004; Zaharia 2016)
- LLM serving: vLLM, PagedAttention (Kwon 2023)
- LLM efficiency: LoRA (Hu 2022), quantization
- Probabilistic data processing (approximate query processing)
- RAG & LangChain — differentiation
- Position Reason-Reduce in this landscape

## Chapter 3: Reason-Reduce Architecture (12–15 pages)
- Formal model: define reason() and reason_reduce()
- System architecture (5 layers)
- Algorithm: semantic partitioner
- Algorithm: Dempster-Shafer combination
- Algorithm: cost-aware router
- Reproducibility contract

## Chapter 4: Implementation (8–10 pages)
- Tech stack rationale
- Worker lifecycle
- Failure modes and mitigations
- Engineering challenges

## Chapter 5: Evaluation (15–18 pages)
- Methodology
- Headline results table
- Ablation studies
- Scalability study
- Calibration analysis
- Negative results

## Chapter 6: Discussion (5–7 pages)
- When does Reason-Reduce win/lose?
- Threats to validity
- Generalization

## Chapter 7: Future Work & Conclusion (3–5 pages)
- Multimodal extension
- Streaming Reason-Reduce
- Cross-cluster federation
- Conclusion

## Appendices
- A: Reproducibility (configs, seeds, hardware, hashes)
- B: Prompt templates (all versions)
- C: Full ablation tables
- D: Code listings of key algorithms
