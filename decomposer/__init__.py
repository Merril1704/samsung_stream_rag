"""Stage 2 decomposer package.

Public surface mirrors controller/:
  Decomposer            – orchestrating class
  DecompositionResult   – result dataclass
  RuleBasedDecomposer   – rule-based layer (can be used standalone)
  LLMVerifier           – LLM verification layer (consulted for ambiguous cases)
"""
from .decomposer import Decomposer
from .types import DecompositionResult
from .rule_based import RuleBasedDecomposer
from .llm_verifier import LLMVerifier

__all__ = ["Decomposer", "DecompositionResult", "RuleBasedDecomposer", "LLMVerifier"]
