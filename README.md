# rag-scorecard

A lightweight, resilient, and production-ready evaluation engine to audit Retrieval-Augmented Generation (RAG) pipelines.

rag-scorecard balances ultra-fast, zero-cost deterministic token-overlap heuristics with deep semantic LLM-as-a-judge capabilities. The framework is architected to handle structural discrepancies, missing parameters, and sudden API rate limit or availability anomalies gracefully.

---

## Core Features

* **Smart Heuristics:** Token-overlap matching logic that filters out common stop words and handles punctuation variations dynamically without relying on external model endpoints.
* **Production-Grounded Math:** Strict mathematical alignment ensuring that failed or empty ground-truth parameters are fully captured as valid zero-score bounds rather than artificially shrinking the total evaluation denominator pool.
* **Multi-Provider Semantic Judges:** Native support for structured output evaluation utilizing both OpenAI and Gemini platforms.
* **Cascading Error Tolerant Architecture:** An embedded resilience layer prevents external network communication failures or 503/429 spikes from interrupting automated verification pipelines.

---

## Quick Start

### Installation
```bash
pip install rag-scorecard
