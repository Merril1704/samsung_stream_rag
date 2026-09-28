"""Hybrid Controller — the production recommendation.

NOT one of the two required ablation arms (those are pure RuleBasedController
and pure ModelBasedController, run independently). This combines them: the
rule-based prefilter runs on every chunk at near-zero cost; the LLM is only
invoked when the rule-based confidence falls in the ambiguous band, bounding
LLM calls to the hard cases — required by the architectural-parsimony constraint.
"""
from .types import SessionState, RetrievalDecision
from .rule_based import RuleBasedController
from .model_based import ModelBasedController

AMBIGUOUS_CONFIDENCE_BAND = (0.45, 0.70)


class HybridController:
    mode = "hybrid"

    def __init__(self, rule_controller: RuleBasedController, model_controller: ModelBasedController):
        self.rule_controller = rule_controller
        self.model_controller = model_controller

    def decide(self, state: SessionState, new_chunk: str, timestamp_s: float) -> RetrievalDecision:
        rule_decision = self.rule_controller.decide(state, new_chunk, timestamp_s)
        if rule_decision.action == "SUPPRESS":
            return rule_decision

        low, high = AMBIGUOUS_CONFIDENCE_BAND
        if low <= rule_decision.confidence <= high:
            # State already has new_chunk appended by the rule controller above;
            # re-judge on the same state without appending twice.
            model_decision = self.model_controller.decide(state, "", timestamp_s)
            model_decision.reason = f"[escalated to model] {model_decision.reason}"
            return model_decision
        return rule_decision
