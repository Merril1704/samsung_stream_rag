
# Streaming Live RAG

> **Cracked Code — Samsung PRISM Generative AI Hackathon 2026–27**  
> **Theme 04: Streaming Live RAG**

## Overview

**Streaming Live RAG** is a retrieval-augmented generation pipeline designed for conversational interactions where the user's intent develops while they are still speaking.

Traditional RAG systems typically wait for a complete user query before retrieving evidence. Our system treats an incoming utterance as a **stream of progressively useful information** and begins retrieval as soon as the intent becomes sufficiently stable.

The system is designed around four core capabilities:

1. **Early / speculative retrieval** — retrieve useful evidence before the utterance ends.
2. **Multi-intent decomposition** — split compound conversational requests into independent retrieval-ready sub-queries.
3. **Evidence fusion and grounded synthesis** — combine, rerank and verify evidence before producing an answer.
4. **Session-aware refinement** — incorporate late-arriving details without unnecessarily restarting the conversation.

The result is a RAG pipeline that aims to reduce perceived latency while preserving evidence traceability and answer quality.

---

## The Problem

In a conventional conversational RAG pipeline:

```text
User speaks
    ↓
Utterance ends
    ↓
Query construction
    ↓
Retrieval
    ↓
Generation
    ↓
Answer
```

This creates several problems:

- Useful retrieval work cannot begin until the user finishes speaking.
- A single natural utterance can contain multiple intents.
- Late-arriving details may require a complete retrieval restart.
- Retrieval decisions are often hidden from the user and difficult to inspect.
- Unsupported or contradictory evidence can lead to unreliable answers.

### Our approach

We change the interaction model to:

```text
User speaks
    ↓
Partial transcript
    ↓
Intent stability check
    ├── WAIT
    ├── SUPPRESS
    └── RETRIEVE
             ↓
      Speculative retrieval
             ↓
     User continues speaking
             ↓
      Multi-intent decomposition
             ↓
      Evidence fusion / reranking
             ↓
       Grounding / verification
             ↓
          Final answer
```

The key distinction is:

> **Conventional RAG:** retrieve after the user finishes.  
> **Streaming Live RAG:** retrieve when enough intent is available, while the user is still speaking.

---

## Key Features

### 1. Retrieval Controller

The controller decides whether the current transcript should:

- `WAIT` — intent is not stable enough for retrieval.
- `RETRIEVE` — sufficient intent has emerged to begin retrieval.
- `SUPPRESS` — retrieval is unnecessary, such as a reformatting request.

This prevents premature retrieval while still enabling early work.

### 2. Speculative / Early Retrieval

When the partial transcript becomes sufficiently stable, the system retrieves candidate evidence **before the final utterance is available**.

The retrieved chunks are retained so that they can be reused when the utterance ends.

Example:

```text
"I need to plan"
        ↓
WAIT

"a customer workshop in Pune for 30 people"
        ↓
SPECULATIVE RETRIEVAL

"and I need the cancellation policy"
        ↓
MULTI-INTENT DECOMPOSITION

"and catering options"
        ↓
FUSION + RERANKING

END UTTERANCE
        ↓
GROUNDED ANSWER
```

### 3. Multi-Intent Decomposition

Natural conversational requests frequently contain multiple questions.

For example:

```text
"I need a venue for 30 people,
and I need the cancellation policy,
and catering options."
```

The system can decompose this into retrieval-ready sub-queries:

```text
1. Venue capacity for 30 people
2. Cancellation policy
3. Catering options
```

The resulting evidence is then fused and reranked.

### 4. Evidence Fusion and Reranking

Evidence from multiple retrieval paths is combined using candidate fusion and reranking.

The implementation includes:

- dense/sparse retrieval support
- reciprocal-rank fusion (RRF)
- reranking
- duplicate handling
- contradiction detection
- evidence coverage checks

### 5. Claim-Level Grounding

The generated answer is checked against the retrieved corpus before rendering.

The verification layer is designed to detect:

- unsupported claims
- fabricated citation IDs
- insufficient evidence
- contradictory evidence
- citation/chunk mismatches

The pipeline is deliberately designed not to treat malformed verifier output as automatically supported.

### 6. Session-Aware Refinement

When a user adds information after an answer has already been generated, the system updates the relevant answer state instead of blindly restarting the entire conversation.

The answer state is versioned so that the evolution of claims can be tracked.

### 7. Observability and Telemetry

The system records pipeline events including:

- retrieval decisions
- timestamps
- retrieval latency
- answer versions
- citation information
- token estimates
- stage timings
- retrieval events

This makes the RAG process inspectable rather than treating it as a black box.

---

# System Architecture

```text
                     ┌─────────────────────────┐
                     │  Incremental Transcript  │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Retrieval Controller  │
                     │ WAIT / RETRIEVE /       │
                     │ SUPPRESS                │
                     └────────────┬────────────┘
                                  │
                         Retrieval triggered
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Multi-Intent           │
                     │  Decomposition          │
                     └────────────┬────────────┘
                                  │
                       Parallel sub-queries
                                  │
                                  ▼
              ┌────────────────────────────────────┐
              │       Corpus Retrieval              │
              │ Dense / Sparse Candidate Search     │
              └────────────────┬───────────────────┘
                               │
                               ▼
                     ┌─────────────────────────┐
                     │ Evidence Fusion &       │
                     │ Reranking               │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Session-Aware Answer     │
                     │ Refinement               │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Grounding / Verification│
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Grounded Answer +       │
                     │ Citations + Telemetry   │
                     └─────────────────────────┘
```

---

# Repository Structure

The implementation is organized around the following logical components:

```text
.
├── eval/
│   ├── benchmark.py
│   ├── benchmark_report.md
│   └── benchmark_results.json
│
├── tests/
│   ├── test_decomposer.py
│   ├── test_fusion.py
│   ├── test_grounding.py
│   ├── test_llm_layer.py
│   ├── test_session_4a.py
│   ├── test_session_4b.py
│   ├── test_telemetry.py
│   └── test_telemetry_e2e.py
│
├── simulate/
│   └── demo_streaming_rag.py
│
└── ...pipeline / retrieval / grounding / telemetry modules
```

> The exact repository tree may evolve as the implementation is packaged for submission. The commands below are the important reproducibility entry points.

---

# Installation

## Requirements

- Python 3.11+
- Git
- A configured LLM/API provider required by the implementation
- Python virtual environment recommended

### Windows

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd samsung_stream_rag

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd samsung_stream_rag

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Configure the required API credentials/environment variables used by the project before running LLM-backed evaluation.

**Do not commit API keys or secrets to GitHub.**

---

# Running the Tests

The project includes unit and end-to-end tests covering decomposition, fusion, grounding, session refinement, telemetry and pipeline behavior.

Run:

```bash
pytest tests/ -v
```

Our current implementation test run:

```text
73 passed, 2 deselected
```

The test suite completed successfully in approximately 111 seconds in the development environment.

---

# Running the Streaming Demo

The repository includes a streaming demonstration:

```bash
PYTHONUTF8=1 python -m simulate.demo_streaming_rag
```

On Windows Git Bash, the command used during development was:

```bash
PYTHONUTF8=1 .venv/Scripts/python -m simulate.demo_streaming_rag
```

The demonstration shows the controller processing incremental transcript chunks and triggering speculative retrieval before the final utterance.

A representative interaction is:

```text
Chunk 1:
"I need to cancel"

→ WAIT
→ Intent unstable
→ No retrieval

Chunk 2:
"a corporate event"

→ WAIT
→ Intent becoming stable

Chunk 3:
"booked for next week"

→ SPECULATIVE RETRIEVAL
→ User still speaking
→ Candidate evidence pre-warmed

Chunk 4:
"because of a force majeure closure."

→ END OF UTTERANCE
→ Reuse speculative evidence
→ Fusion
→ Grounding
→ Final answer
```

For the final hackathon presentation, the preferred interface is an interactive web UI where each partial speech chunk can be submitted manually so judges can observe the pipeline transition in real time.

---

# Benchmarking

The benchmark evaluates representative scenarios covering:

- single-intent queries
- multi-intent queries
- session refinement
- deliberate coverage gaps
- contradictory evidence

Run:

```bash
python -m eval.benchmark \
  --markdown eval/benchmark_report.md \
  --json eval/benchmark_results.json \
  --input-price 0.15 \
  --output-price 0.60
```

The benchmark used during development contained:

```text
13 scenarios
15 turns
```

The generated reports are:

```text
eval/benchmark_report.md
eval/benchmark_results.json
```

---

# Current Evaluation Results

The current benchmark produced the following results.

| Metric | Result |
|---|---:|
| Raw Retrieval Recall@10 | **100.0%** |
| Raw Retrieval Recall@5 | 92.3% |
| Raw Retrieval Recall@3 | 92.3% |
| Raw Retrieval Recall@1 | 67.9% |
| Fused Candidate Recall@10 | **100.0%** |
| Fused Candidate Recall@5 | 96.2% |
| Multi-Intent Sub-query Recall@10 | **100.0%** |
| Controller Action Accuracy | **100.0%** |
| Citation Validity | **100.0%** |
| Fabricated Citation Rate | **0.0%** |
| Mean Turn Latency | 80.531 s |
| Median Turn Latency | 84.647 s |
| Retrieval Latency | 0.009 s |
| Generator Latency | 3.088 s |
| Verifier Latency | 9.399 s |
| Response Latency Proxy | 3.098 s |
| Estimated Total API Cost | $0.0848 |

### Important limitation

The current benchmark pipeline operates synchronously with:

```text
stream=False
```

Therefore **true end-to-end streaming TTFT is not yet reported** by the benchmark.

The response-latency proxy should not be interpreted as measured end-to-end streaming latency.

Another current quality gap is citation groundedness, which was measured at **50.0%** in the benchmark configuration used.

These limitations are intentionally reported rather than hidden because the next engineering stage is to improve true streaming behavior and grounding quality.

---

# Engineering Safety

The pipeline includes explicit handling for failure cases.

### Fabricated citations

Unknown or invalid chunk IDs are rejected during verification.

### Unsupported claims

Claims without sufficient supporting evidence are not automatically marked as supported.

### Contradictory evidence

Contradictory sources are detected and retained for explicit handling rather than silently dropping one side.

### Query suppression

Requests that do not require retrieval can be handled without unnecessary retrieval calls.

### Context limits

Context-budget failures propagate through the pipeline rather than being silently converted into successful answers.

---

# Evaluation Test Coverage

The automated test suite includes coverage for:

- single-intent decomposition
- multi-intent decomposition
- over-fragmentation protection
- entity protection
- reciprocal-rank fusion
- reranking
- contradiction detection
- insufficient evidence
- fabricated citations
- unsupported claims
- citation cherry-picking
- LLM client behavior
- context-budget handling
- answer generation
- answer-ledger behavior
- session refinement
- retrieval deduplication
- answer-version supersession
- telemetry
- stage timing
- token estimation
- event logging
- end-to-end telemetry

This allows the project to validate individual pipeline components as well as complete conversational scenarios.

---

# Example: Why Streaming Retrieval Matters

Consider:

> "I need to plan a customer workshop in Pune for 30 people and I need the cancellation policy and catering options."

A conventional system may wait until the entire utterance is complete.

Streaming Live RAG can progressively identify:

```text
Partial intent
      ↓
Venue capacity
      ↓
Speculative retrieval
      ↓
Cancellation policy
      ↓
Catering options
      ↓
Multi-intent decomposition
      ↓
Evidence fusion
      ↓
Grounded response
```

The objective is not to generate prematurely.

The objective is:

> **Retrieve early, commit carefully.**

---

# Interactive Demo Concept

For the hackathon demonstration, the recommended UI exposes two layers:

### User-facing layer

A conversational chatbot interface:

```text
User:
"I need to plan a customer workshop in Pune..."

Assistant:
[streaming / grounded response]
```

### System-facing layer

A live pipeline panel:

```text
Transcript                 ✓
Controller                 ✓
Speculative Retrieval      ⚡
Multi-Intent Decomposition →
Evidence Fusion            →
Grounding / Verification   →
Final Answer               →
```

This makes the normally invisible retrieval process observable to the evaluator.

---

# Design Principles

### 1. Retrieve early

Use partial intent when it is stable enough to provide useful evidence.

### 2. Do not over-retrieve

The controller can wait or suppress retrieval when evidence is unlikely to be useful.

### 3. Do not over-generate

Final synthesis happens only when the conversational state is sufficiently stable.

### 4. Preserve evidence provenance

Every grounded claim should be traceable to corpus evidence.

### 5. Refine instead of restart

Late details should update the affected answer state where possible.

### 6. Measure everything important

Retrieval events, latency, versions, citations and token usage are captured through telemetry.

---

# Hackathon Alignment

This project addresses **Samsung PRISM Generative AI Hackathon — Theme 04: Streaming Live RAG**.

The implementation focuses on the theme requirements:

- retrieval before utterance completion
- natural multi-intent understanding
- evidence fusion and reranking
- session-scoped refinement
- grounded answers
- retrieval/answer observability

The project is designed as a working prototype rather than an ideation-only proposal.

---

# Team

## CRACKED CODE

| Member |
|---|
| **Pallavi Yadav** |
| **Merril Baiju** |
| **Abdur Rahuman** |
| **Gowdham B** |

---

# Reproducibility

For the final hackathon submission:

1. Clone the repository.
2. Create the Python environment.
3. Install dependencies.
4. Configure the required API credentials.
5. Run the automated tests.
6. Run the streaming demonstration.
7. Run the benchmark after the implementation is frozen.
8. Verify that all files referenced by the submission are present.
9. Create the required final release tag.

For Samsung PRISM submission, the required release tag is:

```text
PRISM_GENAI_HACKATHON_Y2026
```

The tagged commit should contain the final code, README, documentation, demo references and other artifacts submitted for evaluation.

---

# Project Status

**Prototype status:** Working

**Automated tests:** 73 selected tests passed in the recorded development run.

**Benchmark:** 13 scenarios / 15 turns completed.

**Primary strengths:**
- early retrieval logic
- controller accuracy
- retrieval recall
- multi-intent handling
- citation validity
- failure-safe verification
- session-aware refinement
- telemetry

**Current engineering focus:**
- true streaming inference and TTFT measurement
- improved citation groundedness
- verifier latency reduction
- polished interactive demonstration UI

---

# License / Usage

This repository was developed as a prototype for the **Samsung PRISM Generative AI Hackathon 2026–27**.

Refer to the repository and competition submission terms before redistributing project code or associated benchmark data.
