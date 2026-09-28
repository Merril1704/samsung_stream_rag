# Gold Manifest — Dev/Test Corpus Only

**Do not index this file.** It exists purely so the Stage 7 evaluation harness can
check retrieval/grounding/refinement outputs against a known answer key.
Application code (controller, decomposer, retriever, synthesizer) must never
reference these doc_ids or facts directly — that would violate the
no-hardcoding constraint and invalidate the ablation results.

## Scenario A — Multi-Intent (venue / cancellation / catering)
- Venue capacity (Pune, ~30 people) → DOC_02 §1, §2, §3
- Cancellation/refund terms → DOC_04 §1, §2
- Catering options → DOC_05 §1, §2
  - §3 is a **deliberate coverage gap** for Venue A — a correct system flags
    uncertainty here rather than inventing a policy.

## Scenario B — Late-arriving refinement (travel reimbursement)
- Initial query ("summarize the travel reimbursement rule") → DOC_06 (all sections)
- Refinement ("international, booking made after travel") →
  DOC_07 §1, §2; DOC_08 §2; DOC_09 §2
- Expected: Answer Version 2 keeps valid DOC_06 facts that don't conflict,
  adds the international/late-booking deltas, without a full session restart.

## Deliberate contradiction (evidence-fusion test)
- DOC_04 §1 (30-day internal cancellation window) vs.
  DOC_10 §2 (14-day vendor-specific clause, which self-acknowledges the
  discrepancy). Tests whether fusion/synthesis surfaces the conflict instead
  of silently picking one source.

## Deliberate coverage gap (uncertainty test)
- DOC_05 §3: Venue A's dietary/external-vendor policy is explicitly stated
  as undocumented. Any answer asserting a specific Venue A catering policy
  is a grounding failure.

## Distractors (precision test)
- DOC_03 (Bangalore venues) — must not surface for Pune-scoped queries.
- DOC_10 §1 (vendor payment terms) — irrelevant to cancellation/catering intents.

## Query suppression test
- Given any prior answer: "repeat that in bullet points" / "make it shorter"
  → controller must emit retrieval_required: false regardless of corpus content.
