# rag-scorecard

A lightweight, resilient, and production-ready evaluation engine to audit Retrieval-Augmented Generation (RAG) pipelines. 

rag-scorecard balances ultra-fast, zero-cost deterministic token-overlap heuristics with deep semantic LLM-as-a-judge capabilities. The framework is architected to handle structural discrepancies, missing parameters, and sudden API rate limit anomalies gracefully without crashing execution pipelines.

---

## Core Features

* **Smart Heuristics**: Token-overlap matching logic that filters out common stop words and handles punctuation variations dynamically without relying on external model endpoints.
* **Production-Grounded Math**: Strict mathematical alignment ensuring that failed or empty ground-truth parameters are captured as valid zero-score bounds rather than artificially shrinking the total evaluation denominator pool.
* **Multi-Provider Semantic Judges**: Native support for structured output evaluation utilizing both OpenAI and Gemini platforms.
* **Cascading Error Tolerant Architecture**: An embedded resilience layer prevents external network communication failures or 503/429 spikes from interrupting automated verification pipelines.

---

## Quick Start

### Installation

```bash
pip install rag-scorecard
```

---

## Library Functions and Python Syntax Reference

The library exposes individual component modules alongside the main orchestration engine, allowing you to run micro-evaluations or full dataset sweeps.

### 1. AuditEngine.run()

The primary orchestration function used to execute an evaluation pipeline over a structured dataset file.

**Syntax:**

```python
from rag_scorecard import AuditEngine

results = AuditEngine.run(
    dataset_path="path/to/dataset.json",
    provider="gemini",            # Target LLM provider: 'gemini' or 'openai'
    model="gemini-3.5-flash",     # Model identifier string
    retrieval_threshold=0.7       # Minimum heuristic overlap cutoff (Optional)
)
```

### 2. HeuristicEvaluator.compute_retrieval_metrics()

A localized function to calculate token intersection metrics without triggering external network requests or LLM API calls.

**Syntax:**

```python
from rag_scorecard.evaluators import HeuristicEvaluator

heuristic_results = HeuristicEvaluator.compute_retrieval_metrics(
    query="What is dynamic memory allocation?",
    retrieved_contexts=[
        "Dynamic memory allocation allows programs to request memory from the heap at runtime.",
        "Static memory is allocated on the stack frame during compilation."
    ],
    ground_truth="Allocation of memory from the heap during runtime execution."
)

print(f"Calculated MRR: {heuristic_results['mrr']}")
print(f"Calculated Hit Rate: {heuristic_results['hit_rate']}")
```

### 3. SemanticJudge.evaluate_sample()

A modular utility function that evaluates a single text sample across the semantic parameters utilizing a selected LLM provider.

**Syntax:**

```python
from rag_scorecard.evaluators import SemanticJudge

judge_response = SemanticJudge.evaluate_sample(
    query="What does free() do in C?",
    context="The free function deallocates the memory block previously allocated by malloc.",
    response="It returns memory back to the system heap.",
    provider="gemini",
    model="gemini-3.5-flash"
)

print(f"Faithfulness Score: {judge_response.faithfulness.score}")
print(f"Reasoning Details: {judge_response.faithfulness.reasoning}")
```

---

## Evaluation Parameters

The framework evaluates model execution and retrieval performance against six strict parameters divided into two core execution layers:

### 1. Deterministic Heuristic Parameters

* **Mean Reciprocal Rank (MRR)**: Evaluates the position of the relevant text chunk within your retrieved context list. If the top-ranked chunk is the most accurate matching chunk, the score is 1.0. If the matching chunk is positioned further down, the score decreases proportionally (e.g., 0.5 for the second position).
* **Hit Rate**: Evaluates retrieval completeness by tracking the percentage of total queries where at least one correct contextual chunk was successfully recovered from the vector database within the top K results.

### 2. Semantic Judge Parameters (LLM-as-a-Judge)

* **Context Relevance**: Measures the information density of your retrieved text chunks. It analyzes whether the context contains strictly necessary data to resolve the query, penalizing verbose noise or redundant background text.
* **Faithfulness**: Audits the generated response for factual hallucinations. It verifies if every claim made in the final output is directly supported and grounded by the retrieved context.
* **Answer Relevance**: Evaluates whether the generated response directly satisfies the core intent of the user query, penalizing conversational drift or vague, non-committal statements.
* **Completeness**: Benchmarks the final generated response directly against the validated ground truth data to verify that no critical structural instructions, technical definitions, or necessary facts were omitted during generation.

---

## Understanding Output Results

When an evaluation sweep concludes, the engine outputs a unified JSON payload containing three distinct tracking metrics layers to help you iterate on your pipeline:

### 1. Dataset Summary & Judge Averages

Provides the macro health metrics of your production pipeline. Low heuristic scores mean you need to optimize your vector database chunking strategy, index parameters, or embedding model. Low semantic scores mean you need to refine system prompt engineering constraints or swap your generation model.

### 2. Performance Telemetry

Tracks system execution latency (`duration_ms`), cumulative input/output token usage (`tokens`), and estimated cost tracking metrics (`estimated_cost_usd`). This allows engineering teams to optimize the financial efficiency of their evaluation automation pipelines.

### 3. Detailed Logs

Per-sample analytical breakdown containing the localized score, pass/fail state validation boolean, and raw natural language reasoning strings generated directly by the semantic engine explaining exactly why a model scored poorly on a specific test sample.

---

## Data Structure Validation and Architecture

rag-scorecard enforces strict schema constraints using Pydantic v2 validation layers:

* **Strict Alignment Mode**: Structural anomalies or uncoercible score vectors trigger handled parsing warnings instead of corrupting execution run loops.
* **Token Intersection Analysis**: Stop words (such as 'the', 'a', 'an', 'is') are stripped prior to set-intersection evaluations to protect the integrity of the tracking metrics.
