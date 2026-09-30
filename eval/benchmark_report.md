# Streaming Live RAG — Benchmark Evaluation Report

**Generated**: 2026-09-29T20:00:30.901243+00:00  
**Total Scenarios Evaluated**: 13 (15 turns)  
**Token Count Mode**: estimated (3.5 chars / token)  

---

## 1. Executive Summary & Primary Theme 4 Metrics

### A. Retrieval Metrics
| Metric | Result | Target Benchmark Status | Notes |
| :--- | :---: | :---: | :--- |
| **Raw Retrieval Recall@10** | **100.0%** | ✅ Evaluated | Mean across answerable turns (excludes empty-gold) |
| **Raw Retrieval Recall@5** | 92.3% | ✅ Evaluated | Top-5 retrieval recall |
| **Raw Retrieval Recall@3** | 92.3% | ✅ Evaluated | Top-3 retrieval recall |
| **Raw Retrieval Recall@1** | 67.9% | ✅ Evaluated | Top-1 retrieval recall |
| **Fused Candidate Recall@10** | **100.0%** | ✅ Evaluated | Post-fusion/reranked candidate list @ 10 |
| **Fused Candidate Recall@5** | 96.2% | ✅ Evaluated | Post-fusion/reranked candidate list @ 5 |
| **Fused Candidate Recall@3** | 88.5% | ✅ Evaluated | Post-fusion/reranked candidate list @ 3 |
| **Fused Candidate Recall@1** | 64.1% | ✅ Evaluated | Post-fusion/reranked candidate list @ 1 |
| **Multi-Intent Sub-query Recall@10** | 100.0% | ✅ Evaluated | Independent per-sub-query recall |

### B. Routing Metrics
| Metric | Result | Target Benchmark Status | Notes |
| :--- | :---: | :---: | :--- |
| **Controller Action Accuracy** | **100.0%** | ✅ Evaluated | Correctly classified actions / total turns |

### C. Evidence Relevance & Groundedness
| Metric | Result | Target Benchmark Status | Notes |
| :--- | :---: | :---: | :--- |
| **Citation Groundedness** | **50.0%** | ✅ Evaluated | Evidence Relevance Rate: claims citing expected chunks |
| **Citation Validity** | 100.0% | ✅ Evaluated | Valid corpus chunks cited |
| **Fabricated Citation Rate** | 0.0% | ✅ Evaluated | Non-existent chunk IDs cited |

### D. Latency Breakdown (Synchronous Pipeline)
| Metric | Result | Target Benchmark Status | Notes |
| :--- | :---: | :---: | :--- |
| **Mean Turn Latency** | 80.531s | ✅ Timed | Monotonic perf_counter duration |
| **Median Turn Latency** | 84.647s | ✅ Timed | Turn latency median |
| **Retrieval Latency** | 0.009s | ✅ Timed | Index retrieval stage duration |
| **Generator Latency** | 3.088s | ✅ Timed | LLM generation stage duration |
| **Verifier Latency** | 9.399s | ✅ Timed | Verification stage duration |
| **Response Latency Proxy** | 3.098s | ✅ Modeled | Retrieval + Generator duration |
| **Streaming TTFT** | **N/A** | ℹ️ Design Note | Architecture operates synchronously (`stream=False`) |

### E. Token Usage & Modeled Cost
| Metric | Result | Target Benchmark Status | Notes |
| :--- | :---: | :---: | :--- |
| **Tokens per Turn** | 24973.2 | ✅ Estimated | Modeled at ~3.5 chars/token |
| **Total Tokens** | 374,598 | ✅ Estimated | Modeled across all turns |
| **Total Estimated API Cost** | $0.0848 | ✅ Modeled | Based on active PricingConfig |

> [!IMPORTANT]
> Note: Architecture operates synchronously (stream=False); generator_latency_s and response_latency_proxy_s represent non-streamed inference duration. True streaming TTFT = N/A.

---

## 2. Category Performance Breakdown

| Category | Scenarios | Turns | Controller Accuracy | Raw Recall@10 | Fused Recall@10 | Groundedness | Mean Latency (s) | Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `unanswerable` | 2 | 2 | 100.0% | N/A | N/A | 50.0% | 59.506s | 36,885 |
| `multi_intent` | 2 | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 67.047s | 43,125 |
| `contradiction` | 1 | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 64.724s | 22,859 |
| `single_intent` | 6 | 6 | 100.0% | 100.0% | 100.0% | 58.3% | 85.195s | 158,094 |
| `refinement` | 2 | 4 | 100.0% | 100.0% | 100.0% | 0.0% | 94.742s | 113,635 |

---

## 3. Detailed Scenario Results

| Scenario ID | Category | Raw Recall@10 | Fused Recall@10 | Groundedness | Latency (s) | Tokens | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `eval_single_01` | `single_intent` | 100.0% | 100.0% | 50.0% | 111.487s | 31,864 | ✅ PASS |
| `eval_single_02` | `single_intent` | 100.0% | 100.0% | 50.0% | 55.741s | 20,828 | ✅ PASS |
| `eval_single_03` | `single_intent` | 100.0% | 100.0% | 0.0% | 97.962s | 30,696 | ✅ PASS |
| `eval_single_04` | `single_intent` | 100.0% | 100.0% | 100.0% | 73.521s | 23,596 | ✅ PASS |
| `eval_single_05` | `single_intent` | 100.0% | 100.0% | 50.0% | 87.813s | 26,662 | ✅ PASS |
| `eval_single_06` | `single_intent` | 100.0% | 100.0% | 100.0% | 84.647s | 24,448 | ✅ PASS |
| `eval_multi_01` | `multi_intent` | 100.0% | 100.0% | 100.0% | 59.225s | 18,939 | ✅ PASS |
| `eval_multi_02` | `multi_intent` | 100.0% | 100.0% | 100.0% | 74.868s | 24,186 | ✅ PASS |
| `eval_refine_01` | `refinement` | 100.0% | 100.0% | 0.0% | 195.070s | 54,414 | ✅ PASS |
| `eval_refine_02` | `refinement` | 100.0% | 100.0% | 0.0% | 183.896s | 59,221 | ✅ PASS |
| `eval_unans_01` | `unanswerable` | N/A | N/A | 100.0% | 49.053s | 16,882 | ✅ PASS |
| `eval_unans_02` | `unanswerable` | N/A | N/A | 0.0% | 69.960s | 20,003 | ⚠️ REVIEW |
| `eval_contra_01` | `contradiction` | 100.0% | 100.0% | 100.0% | 64.724s | 22,859 | ✅ PASS |
