[README(1).md](https://github.com/user-attachments/files/32867499/README.1.md)

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

## Executive Summary

Traditional Retrieval-Augmented Generation (RAG) operates on a synchronous **"wait-then-retrieve"** bottleneck: a system remains idle until the user finishes typing or speaking, incurring a multi-second latency penalty before evidence search, reranking, and generation even begin. Furthermore, traditional pipelines fail when handling multi-intent clauses, contradictory policy documents, or late-arriving conversational refinements.

**Streaming Live RAG** re-architects conversational retrieval into an **in-flight predictive pipeline**. As human speech develops incrementally:

1. **Speculative Pre-fetching**: The intent stability controller monitors partial speech clauses, triggering background dense/sparse retrieval the moment semantic intent stabilizes—**while the user is still speaking**.
2. **Zero-Latency Handoff**: When speech concludes, the system reuses pre-warmed candidates directly from cache, bypassing search latency entirely (0.0 ms retrieval delay).
3. **Multi-Intent Decomposition**: Complex conjunctions are partitioned into atomic sub-queries, executed across corpus indices, and fused using Reciprocal Rank Fusion (RRF).
4. **Claim-Level Grounding**: Candidate responses are decomposed into atomic claims and verified for entailment against source corpus text, eliminating hallucinations (0% fabricated citations).
5. **Evolutionary Answer Ledger**: Follow-up refinements (e.g. natural disaster exemptions, budget thresholds) update existing ledger records ($v1 \to v2$) without restarting conversation context.

---

## System Architecture

```
                  USER SPEECH / INCREMENTAL TRANSCRIPT
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 1: Progressive Transcript Ingestion     │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 2: Intent Stability Controller          │
        │       - WAIT: Entity threshold < 4 tokens           │
        │       - PREFETCH: Speculative background search     │
        │       - SUPPRESS: Conversational / formatting cues  │
        └──────────────────────────┬──────────────────────────┘
                                   │
                     Intent Formed (Pre-fetch Trigger)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 3: Speculative Background Retrieval     │
        │       Pre-warms candidates in background cache       │
        └──────────────────────────┬──────────────────────────┘
                                   │
                      Speaker Ends Speech (is_final=True)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 4: Multi-Intent Query Decomposition     │
        │       Partitions compound queries into sub-queries  │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 5: Evidence Fusion & Contradiction      │
        │       - Reciprocal Rank Fusion (RRF)                │
        │       - Cross-document Contradiction Pre-screening  │
        │       - 0ms Zero-Latency Cache Handoff              │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 6: Claim Grounding & Verification       │
        │       - Strict NLI Entailment vs Source Chunks      │
        │       - Rejection of Fabricated Citations           │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       STAGE 7: Versioned Answer Ledger & Synthesis  │
        │       - Ledger Commit & Version Tracking (v1 -> v2) │
        │       - Final Answer with Clickable Corpus Citations│
        └─────────────────────────────────────────────────────┘
```

---

## Key Capabilities

### 1. Speculative In-Flight Retrieval

Rather than waiting for the end-of-speech delimiter, the system measures token stability and semantic completeness. If an utterance like *"I need to plan a customer workshop in Pune for 30 people..."* crosses the stability threshold, speculative retrieval fires in a non-blocking background thread. When the speaker finishes, evidence is already cached in memory.

### 2. Zero-Latency Cache Handoff

By computing candidates during conversational delivery, final synthesis begins immediately upon speech termination, achieving **0 ms retrieval delay** at turn end.

### 3. Multi-Intent Decomposition

Real-world queries often bundle disparate requirements. For example:
> *"I need a venue in Bangalore for 50 people and also need the standard catering packages."*

The decomposer splits this into:

- `Sub-query 1`: Bangalore venue capacity 50 attendees (`DOC_03_§1`)
- `Sub-query 2`: Standard catering packages (`DOC_05_§1`)

Evidence from each sub-query is retrieved in parallel and fused using rank-aware reciprocal algorithms.

### 4. Enterprise Answer Ledger & Refinement

When a user follows up with a condition or constraint (e.g. *"the cancellation was due to a government mandated closure"*), the pipeline checks semantic overlap against active ledger entries. Instead of wiping the session or answering blindly:

- The previous entry is marked for refinement.
- New claims are checked and appended.
- The entry is incremented ($v1 \to v2$) with full audit provenance.

### 5. Strict Entailment & Anti-Hallucination Shield

Every generated factual statement is decomposed into an atomic claim and verified against verbatim corpus passages. If a model hallucinates a non-existent document ID or unsupported fact, the verifier intercepts it, ensuring **100% citation validity** and **0% fabricated citations**.

---

## Interactive Demonstration Web UI

The project features a **pure conversational chatbot interface** accompanied by an **Inspect RAG Activity** observability side drawer tailored for hackathon evaluators.

<div align="center">
  <img src="https://via.placeholder.com/1000x500/121826/38bdf8?text=Streaming+Live+RAG+Interactive+Chatbot+Interface" alt="Live Demo Interface" width="100%">
</div>

### Features of the Web Interface

- **Natural Multi-Turn Chat**: Single clean input bar with Enter/Send controls, responsive bubbles, and smooth typewriter answer rendering.
- **Simulated Speech Streaming**: Progressive clause splitting emulates real-time voice transcripts behind the scenes.
- **In-Flight Status Indicators**: Subtle indicators (`🎙️ Listening...`, `⚡ Finding relevant information...`, `🔄 Updating context...`, `✓ Context ready`).
- **Collapsible RAG Activity Drawer**:
  - **Speculative Pre-fetch Status**: Displays cache hits, pre-warmed chunk IDs, and retrieval delay (0.0 ms).
  - **Multi-Intent Decomposition**: Shows decomposed sub-queries.
  - **Grounding Verification**: Interactive inspection of atomic claim checks and source chunk alignments.
  - **Answer Ledger**: Live view of versioned entries ($v1 \to v2$).

---

## Installation & Quickstart

### Prerequisites

- Python 3.11+
- Virtual environment (`venv` or `conda`)
- Groq API Key (or OpenAI / Ollama compatible endpoint)

### Setup

```bash
# Clone the repository
git clone https://github.com/Merril1704/samsung_stream_rag.git
cd samsung_stream_rag

# Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Primary LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Embeddings & Retrieval
EMBEDDING_PROVIDER=mock
BM25_K1=1.5
BM25_B=0.75
```

---

## Running the Application

### 1. Launch Interactive Chat Web Application

```bash
python -m demo.server
```

Open your browser at:

```
http://127.0.0.1:8000
```

> **Presenter Tip**: Try clicking any of the Quick Prompt chips on the welcome screen, or type natural follow-up questions to witness real-time versioned answer refinements!

### 2. Run the Benchmark Evaluation Suite

Execute the full held-out 15-turn evaluation across all 5 test categories:

```bash
python -m eval.benchmark --markdown eval/benchmark_report.md --json eval/benchmark_results.json
```

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

---

## Benchmark Evaluation Results

Evaluated against the held-out evaluation corpus consisting of 10 enterprise policy and directory documents in `corpus/raw/` across 13 complex scenarios (15 multi-turn dialogues):

| Metric | Result | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Raw Retrieval Recall@10** | **100.0%** | > 90.0% | PASS |
| **Raw Retrieval Recall@5** | **92.3%** | > 85.0% | PASS |
| **Fused Candidate Recall@10** | **100.0%** | > 95.0% | PASS |
| **Multi-Intent Sub-query Recall@10** | **100.0%** | > 90.0% | PASS |
| **Controller Action Accuracy** | **100.0%** | > 95.0% | PASS |
| **Citation Validity Rate** | **100.0%** | 100.0% | PASS |
| **Fabricated Citation Rate** | **0.0%** | 0.0% | PASS |
| **Cache Handoff Retrieval Delay** | **0.0 ms** | < 50.0 ms | OPTIMAL |

---

## Repository Structure

```
samsung_stream_rag/
├── corpus/
│   └── raw/                       # 10 Indexed Corporate Policy & Venue Documents
│       ├── doc_01_event_booking_policy.md
│       ├── doc_02_venue_directory_pune.md
│       ├── doc_03_venue_directory_bangalore.md
│       ├── doc_04_cancellation_refund_policy.md
│       ├── doc_05_catering_vendor_options.md
│       ├── doc_06_domestic_travel_reimbursement.md
│       ├── doc_07_international_travel_reimbursement.md
│       ├── doc_08_travel_booking_exceptions.md
│       ├── doc_09_expense_approval_workflow.md
│       └── doc_10_vendor_contract_terms.md
├── controller/                    # Intent Stability & Retrieval Gating
│   ├── rule_based.py              # Entity Threshold & Action Controller
│   ├── llm_factory.py             # Groq / OpenAI Provider Factory
│   └── types.py                   # Controller Action & Session Types
├── decomposer/                    # Multi-Intent Query Decomposition
│   └── decomposer.py              # Conjunction Analysis & Sub-query Generator
├── retrieval/                     # Hybrid Dense/Sparse Search Index
│   ├── bm25.py                    # Lexical BM25 Scoring
│   ├── dense.py                   # Semantic Vector Matching
│   └── indexer.py                 # Corpus Ingestion & Document Parser
├── fusion/                        # Candidate Fusion & Contradiction Defense
│   ├── fusion_pipeline.py         # Reciprocal Rank Fusion (RRF)
│   └── contradiction.py           # Pre-filtering Contradictory Clauses
├── grounding/                     # Entailment Verification Engine
│   ├── verifier.py                # Atomic Claim Decomposition & NLI Verification
│   └── types.py                   # ClaimVerification & Citation Types
├── session/                       # Session State & Answer Ledger
│   ├── orchestrator.py            # StreamRAGOrchestrator Pipeline Director
│   ├── ledger.py                  # AnswerLedger & Version Increment Engine
│   └── generator.py               # Grounded Synthesis Generator
├── eval/                          # Held-out Benchmark Engine
│   ├── scenarios.py               # 13 Multi-turn Held-out Evaluation Scenarios
│   ├── benchmark.py               # Comprehensive Metric Runner
│   └── benchmark_report.md        # Generated Benchmark Audit Report
├── demo/                          # Interactive Web Demonstration Server
│   ├── server.py                  # Threading HTTP & Server-Sent Events (SSE) Engine
│   └── static/
│       ├── index.html             # Conversational Assistant Interface
│       ├── style.css              # Dark Glassmorphism Styling System
│       └── app.js                 # EventSource Streaming Client
├── tests/                         # Full Pytest Test Suite
│   ├── test_interactive_demo.py   # Web UI & Streaming Pipeline Integration Tests
│   ├── test_refinement_pipeline.py# Multi-Turn Refinement & Versioning Tests
│   └── ...                        # Component Unit Tests
├── pytest.ini                     # Pytest Configuration
├── requirements.txt               # Locked Dependencies
└── README.md                      # Project Documentation
```

---

## Grounding and Safety Mechanisms

1. **Strict Provenance Verification**: Answers cite explicit corpus chunk designations (e.g. `[DOC_04_§3]`). Unreferenced claims are rejected.
2. **Cherry-Pick Detection**: If a query is answerable by multiple conflicting documents, the system triggers conflict resolution rather than presenting a partial truth.
3. **Coverage Gap Awareness**: When queries reference unindexed domains (e.g., personal car per-km mileage rates), the system explicitly flags the policy boundary rather than fabricating numerical estimates.
4. **Rate-Limit Resilience**: In high-load presentation environments, pre-validated verified entries provide sub-second responses without triggering downstream API backoff delays.

---

## Team & Contributors

### **Team CRACKED CODE**

| Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **Pallavi Yadav** | System Architecture & Eval | Scenario curation, benchmark metrics, evaluation design |
| **Merril Baiju** | Core Pipeline & Streaming RAG | Orchestrator, speculative retrieval, controller, web demo |
| **Abdur Rahuman** | Retrieval & Fusion Engine | BM25 indexing, Reciprocal Rank Fusion, contradiction screening |
| **Gowdham B** | Grounding & Ledger Modeling | Claim verification, Answer Ledger versioning, synthesis defense |

---

## Submission Release Tag

For the **Samsung PRISM Generative AI Hackathon 2026–27**, this repository is frozen and tagged at:

```bash
git tag PRISM_GENAI_HACKATHON_Y2026
git push origin PRISM_GENAI_HACKATHON_Y2026
```

---

## License

Developed under the Samsung PRISM Generative AI Hackathon 2026–27. All rights reserved.
