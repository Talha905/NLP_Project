"""
Cross-Encoder Reranker for Second-Stage Retrieval in RAG.
Reranks first-stage candidate chunks using deep cross-attention between query and context.
"""

import sys
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None
if "onnx" not in sys.modules:
    sys.modules["onnx"] = None

import os
import math
from typing import List, Tuple, Optional
from ..ingestion.chunker import TextChunk
from ..config import DEFAULT_RERANKER_MODEL


class CrossEncoderReranker:
    """
    Second-stage Cross-Encoder reranker.
    Computes cross-attention between query and candidate passages to refine ranking.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, local_files_only: bool = True):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads CrossEncoder model safely with offline enforcement."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)
            print(f"[RERANKER] [LOAD] Loaded Cross-Encoder '{self.model_name}' (offline mode: {self.local_files_only})")
        except Exception as e:
            # Attempt load without local_files_only if initial cache missing
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                print(f"[RERANKER] [LOAD] Loaded Cross-Encoder '{self.model_name}' (online fallback)")
            except Exception as e2:
                print(f"[RERANKER] [WARNING] Could not load CrossEncoder '{self.model_name}': {e2}. Falling back to 1st-stage scores.")
                self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[TextChunk, float]],
        top_k: int = 3
    ) -> List[Tuple[TextChunk, float]]:
        """
        Reranks candidate (TextChunk, score) pairs using cross-encoder cross-attention.
        Returns top_k (TextChunk, reranked_score) sorted descending by score.
        """
        if not candidates:
            return []

        if len(candidates) <= 1 or self.model is None:
            return candidates[:top_k]

        # Prepare pairs: query + text (using parent_text if available for full context)
        pairs = []
        for chunk, _ in candidates:
            doc_context = chunk.parent_text if chunk.parent_text else chunk.text
            pairs.append((query, doc_context))

        try:
            raw_scores = self.model.predict(pairs)
            # Convert raw logits to sigmoid probabilities [0, 1] for clean scoring
            sigmoid_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

            scored_candidates = []
            for (chunk, _), score in zip(candidates, sigmoid_scores):
                scored_candidates.append((chunk, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            selected = scored_candidates[:top_k]

            best_score = selected[0][1] if selected else 0.0
            print(
                f"[RERANKER] [2ND_STAGE] Reranking {len(candidates)} candidates for query '{query[:45]}...' "
                f"-> Top-{len(selected)} selected (Best CE sigmoid score: {best_score:.4f})"
            )
            return selected

        except Exception as e:
            print(f"[RERANKER] [ERROR] Reranking failed: {e}. Falling back to 1st-stage order.")
            return candidates[:top_k]
