"""
Natural Language Inference (NLI) and Groundedness / Faithfulness analysis.
Measures whether the generated answer is supported by the retrieved context.
"""

import re
from typing import List, Dict, Any
from ..ingestion.chunker import TextChunk
from ..retrieval.embedder import DenseEmbedder
from ..config import FAITHFULNESS_THRESHOLD


class GroundingEvaluator:
    """
    Evaluates factual consistency, groundedness, and hallucinations
    between generated responses and retrieved context chunks.
    """

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder

    def evaluate_groundedness(
        self,
        answer: str,
        retrieved_chunks: List[TextChunk]
    ) -> Dict[str, Any]:
        """
        Assesses if the answer statements are grounded in retrieved chunks.
        Returns:
            - faithfulness_score (0.0 to 1.0)
            - is_faithful (bool)
            - unsupported_claims (List[str])
            - context_coverage (float)
        """
        if not retrieved_chunks or not answer.strip():
            return {
                "faithfulness_score": 0.0,
                "is_faithful": False,
                "unsupported_claims": [answer] if answer.strip() else [],
                "context_coverage": 0.0,
                "contradiction_detected": False
            }

        # Combine all chunk text
        full_context = " ".join([c.text for c in retrieved_chunks]).lower()

        # Split answer into sentence claims
        claims = [c.strip() for c in re.split(r"[.!?]\s+", answer) if len(c.strip()) > 10]
        if not claims:
            claims = [answer.strip()]

        supported_count = 0
        unsupported_claims = []

        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "in", "to", "for", "that", "this", "with", "as", "by"}

        for claim in claims:
            claim_lower = claim.lower()
            claim_tokens = [w for w in re.findall(r"\b\w+\b", claim_lower) if w not in stop_words and len(w) > 2]

            if not claim_tokens:
                supported_count += 1
                continue

            # Token recall in context
            present_tokens = [t for t in claim_tokens if t in full_context]
            token_recall = len(present_tokens) / len(claim_tokens)

            # Semantic similarity of claim to best context chunk
            max_sim = 0.0
            for chunk in retrieved_chunks:
                sim = self.embedder.embed_query(claim)
                c_vec = self.embedder.embed_query(chunk.text)
                dot = float(sim.dot(c_vec) / (np_norm(sim) * np_norm(c_vec) + 1e-9))
                if dot > max_sim:
                    max_sim = dot

            # A claim is supported if token recall is high or semantic similarity is high
            is_supported = (token_recall >= 0.50) or (max_sim >= 0.58)

            if is_supported:
                supported_count += 1
            else:
                unsupported_claims.append(claim)

        faithfulness = supported_count / max(1, len(claims))
        is_faithful = faithfulness >= FAITHFULNESS_THRESHOLD

        # Contradiction heuristic (e.g. presence of opposite antonyms or negation mismatches)
        contradiction_detected = False
        negations = ["not", "never", "no", "cannot", "neither", "nor"]
        has_negation_ans = any(f" {n} " in f" {answer.lower()} " for n in negations)
        has_negation_ctx = any(f" {n} " in f" {full_context} " for n in negations)
        if has_negation_ans and not has_negation_ctx and faithfulness < 0.5:
            contradiction_detected = True

        return {
            "faithfulness_score": float(faithfulness),
            "is_faithful": bool(is_faithful),
            "unsupported_claims": unsupported_claims,
            "context_coverage": float(faithfulness),
            "contradiction_detected": contradiction_detected
        }


def np_norm(v):
    import numpy as np
    n = np.linalg.norm(v)
    return max(1e-9, float(n))
