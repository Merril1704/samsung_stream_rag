# Stage 7 Evaluation Dataset — Held-Out Benchmark

This directory contains the held-out evaluation dataset used to assess the Streaming Live RAG pipeline (`samsung_stream_rag`).

> [!NOTE]
> This dataset comprises **13 targeted evaluation scenarios** derived exclusively from the 10 corpus documents (`corpus/raw/doc_01` to `doc_10`). As a hackathon-level test suite, it is designed for comparative ablation and error analysis across specific pipeline capabilities rather than large-scale statistical significance.

---

## 1. Separation from Regression Test Fixtures

Existing unit and end-to-end tests in `tests/` explicitly hardcode Scenario A (Pune workshop / cancellation / catering) and Scenario B (Travel reimbursement refinement). Using those same queries for benchmarking would constitute in-sample evaluation.

The held-out dataset in `eval/scenarios.py`:
* **Eliminates prompt overlap**: None of the 13 scenario prompts match or trivially paraphrase regression test strings.
* **Exercises previously unqueried corpus sections**: Incorporates corporate event approval tiers (`DOC_01 §1, §2, §3`), Bangalore venue directory (`DOC_03 §1, §2`), force majeure exceptions (`DOC_04 §3`), emergency booking rules (`DOC_08 §1, §3`), currency conversion dates (`DOC_07 §3`), reimbursement payout timelines (`DOC_06 §3`), ₹25,000 expense approval thresholds (`DOC_09 §1, §2`), and procurement payment terms (`DOC_10 §1`).

---

## 2. Dataset Inventory & Breakdown

The dataset contains 13 scenarios across 5 functional categories:

| Category | Count | Scenario IDs | Primary Evaluation Focus |
| :--- | :---: | :--- | :--- |
| **Single-Intent Retrieval** | 6 | `eval_single_01` .. `eval_single_06` | Single-query retrieval recall, precision, and atomic claim generation. |
| **Multi-Intent Retrieval** | 2 | `eval_multi_01`, `eval_multi_02` | Utterance decomposition, sub-query fan-out, RRF fusion, and multi-intent recall. |
| **Multi-Turn Refinement** | 2 | `eval_refine_01`, `eval_refine_02` | Turn 1 baseline + Turn 2 supplementary detail; ledger versioning without restart. |
| **Unanswerable / Coverage Gap** | 2 | `eval_unans_01`, `eval_unans_02` | Detection of unanswerable queries / coverage gaps; refusal to hallucinate facts. |
| **Contradiction-Sensitive** | 1 | `eval_contra_01` | Surfacing differing policy scopes (30-day internal vs 14-day vendor terms) without silent suppression. |

---

## 3. Gold Relevance & Groundedness Schema

All scenarios are defined in [`eval/scenarios.py`](file:///d:/Projects/samsung_stream_rag/eval/scenarios.py) using strongly-typed dataclasses:

### A. Gold Chunks
* `expected_chunk_ids`: Actual chunk identifiers present in `chunk_lookup` (e.g. `DOC_01_§1`).
* **Multi-Intent Alignment**: For multi-intent queries, `sub_query_gold_chunks` maps each specific decomposed sub-query to its dedicated gold chunk subset. This prevents artificial recall penalties caused by cross-intent reranker competition.

### B. Gold Claims
* `GoldClaim(claim_id, text, supporting_chunk_ids)`: Atomic, factual propositions extracted directly from corpus text.
* **No Circular Grading**: Gold claims are independently authored from primary source documents. They are **not** generated or graded by the pipeline's own internal `GroundingVerifier`.

### C. Unanswerable Cases
* Cases such as `eval_unans_01` (Venue A vegan catering policy, declared undocumented in `DOC_05 §3`) and `eval_unans_02` (personal car mileage rate, absent from corpus) specify `is_answerable = False`, `expected_chunk_ids = []`, and `gold_claims = []`.
* Successful execution requires the pipeline to flag insufficient evidence or state policy unavailability rather than inventing claims.

---

## 4. Intended Metric Definitions (for `eval/benchmark.py`)

When the benchmark runner is implemented in Stage 7B, it will compute:

1. **Retrieval Recall@$k$**:
   $$\text{Recall@}k = \frac{|\text{Retrieved Chunks@}k \cap \text{Gold Relevant Chunks}|}{|\text{Gold Relevant Chunks}|}$$
   *Evaluated per sub-query for multi-intent scenarios and globally at the turn level.*

2. **Claim Groundedness / Precision**:
   $$\text{Groundedness} = \frac{|\text{Generated Claims Entailed by Verified Evidence}|}{|\text{Total Generated Claims}|}$$
   *Penalizes unsupported claims and fabricated citations.*

3. **Latency (Response / TTFT Proxy)**:
   * **Total Response Latency**: Wall-clock duration of the turn.
   * **Inference Latency**: Duration of generation LLM call(s).
   * Note: As the current inference backend operates synchronously (`stream=False`), generation latency is reported as a documented proxy until true streaming is enabled.

4. **Cost & Token Consumption**:
   * Token estimates based on character ratios (3.5 chars/token).
   * Aggregated across all pipeline stages in the turn.
