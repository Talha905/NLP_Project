"""
Fine-grained failure taxonomy classifier and mitigation advisor.
Provides research-oriented diagnostics and actionable engineering mitigations.
"""

from typing import Dict, Any, List
from .error_decomposer import EvaluationSample


class FailureTaxonomyAdvisor:
    """
    Analyzes evaluation failure modes and generates technical diagnostics
    and prioritized mitigations.
    """

    MITIGATION_GUIDE = {
        "MISSING_EVIDENCE": {
            "root_cause": "The retrieval module failed to rank gold evidence in the Top-K context.",
            "mitigations": [
                "Increase Top-K parameter from current setting to retrieve a broader candidate pool.",
                "Implement Hybrid Search (Dense Bi-encoder + BM25 Lexical) with Reciprocal Rank Fusion.",
                "Fine-tune embedding model on in-domain corpus pairs using contrastive loss (InfoNCE).",
                "Employ query expansion or HyDE (Hypothetical Document Embeddings)."
            ],
            "severity": "HIGH",
            "component": "RETRIEVAL"
        },
        "NOISE_DILUTION": {
            "root_cause": "Gold evidence was retrieved but surrounded by low-precision distractor chunks.",
            "mitigations": [
                "Introduce a cross-encoder reranker (e.g. bge-reranker-base) to filter out distractor chunks.",
                "Apply contextual compression or sentence window retrieval.",
                "Tune chunk size and reduce chunk overlap to minimize redundant text."
            ],
            "severity": "MEDIUM",
            "component": "RETRIEVAL"
        },
        "TRUNCATION_BOUNDARY": {
            "root_cause": "Evidence was split across chunk boundaries, fragmenting semantic meaning.",
            "mitigations": [
                "Increase chunk size (e.g., 400 -> 600 tokens) or chunk overlap (50 -> 100 tokens).",
                "Switch to semantic chunking or hierarchical parent-child document chunking."
            ],
            "severity": "MEDIUM",
            "component": "INGESTION"
        },
        "HALLUCINATION_UNSUPPORTED": {
            "root_cause": "The generator produced factual claims absent from the retrieved context.",
            "mitigations": [
                "Lower LLM temperature to 0.0 for deterministic decoding.",
                "Enforce strict grounded system prompts with negative constraints ('Cite only provided facts').",
                "Integrate post-generation NLI / self-reflection verification filters."
            ],
            "severity": "HIGH",
            "component": "GENERATION"
        },
        "CONTEXT_CONTRADICTION": {
            "root_cause": "The generator stated claims directly contradicting facts in the context.",
            "mitigations": [
                "Enforce citation-constrained prompting where every claim requires [Source X] attribution.",
                "Check for pretraining parametric memory bias overriding in-context evidence."
            ],
            "severity": "HIGH",
            "component": "GENERATION"
        },
        "REASONING_SYNTHESIS_ERROR": {
            "root_cause": "The LLM received required factual premises but failed multi-step deductive synthesis.",
            "mitigations": [
                "Adopt Chain-of-Thought (CoT) prompting or Step-Back Prompting.",
                "Decompose multi-hop questions into sequential single-hop sub-queries (Sub-Question Query Engine).",
                "Upgrade generator to a stronger reasoning model."
            ],
            "severity": "HIGH",
            "component": "GENERATION"
        },
        "INCOMPLETE_ANSWER": {
            "root_cause": "The answer addressed only a subset of the prompt constraints.",
            "mitigations": [
                "Include structured JSON output schemas demanding coverage of all entities.",
                "Increase max generation tokens (num_predict)."
            ],
            "severity": "LOW",
            "component": "GENERATION"
        },
        "CONSERVATIVE_REFUSAL": {
            "root_cause": "The LLM claimed lack of information even though context contained the answer.",
            "mitigations": [
                "Soften negative refusal constraints in the system prompt.",
                "Highlight extracted keywords in context prior to generation."
            ],
            "severity": "MEDIUM",
            "component": "GENERATION"
        }
    }

    @classmethod
    def get_mitigation(cls, failure_subtype: str) -> Dict[str, Any]:
        """Returns diagnostic details and engineering mitigations."""
        return cls.MITIGATION_GUIDE.get(
            failure_subtype,
            {
                "root_cause": "Unclassified error pattern.",
                "mitigations": ["Inspect sample query and retrieved chunks manually."],
                "severity": "LOW",
                "component": "GENERAL"
            }
        )

    @classmethod
    def generate_executive_summary(cls, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Generates an executive research overview of failure hotspots."""
        failures = [s for s in samples if s.quadrant in ["RETRIEVAL_FAILURE", "GENERATION_FAILURE"]]
        if not failures:
            return {
                "top_failure_subtype": "NONE",
                "primary_vulnerable_component": "NONE",
                "priority_recommendation": "System demonstrates high end-to-end fidelity across tested benchmarks."
            }

        counts: Dict[str, int] = {}
        for f in failures:
            counts[f.failure_subtype] = counts.get(f.failure_subtype, 0) + 1

        top_subtype = max(counts.keys(), key=lambda k: counts[k])
        guide = cls.get_mitigation(top_subtype)

        ret_count = sum(1 for f in failures if f.quadrant == "RETRIEVAL_FAILURE")
        gen_count = sum(1 for f in failures if f.quadrant == "GENERATION_FAILURE")

        primary_comp = "RETRIEVAL" if ret_count >= gen_count else "GENERATION"

        return {
            "top_failure_subtype": top_subtype,
            "top_failure_count": counts[top_subtype],
            "primary_vulnerable_component": primary_comp,
            "root_cause": guide["root_cause"],
            "priority_recommendation": guide["mitigations"][0],
            "all_mitigations": guide["mitigations"]
        }
