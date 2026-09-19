"""
Hybrid retriever combining Dense semantic search and BM25 lexical search
using Reciprocal Rank Fusion (RRF) or linear score combination.
"""

from typing import List, Tuple, Dict, Literal
import numpy as np
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .embedder import DenseEmbedder
from ..ingestion.chunker import TextChunk
from ..config import DEFAULT_RRF_K, DEFAULT_HYBRID_ALPHA


class HybridRetriever:
    """
    Hybrid search engine unifying dense embeddings and BM25 sparse matching.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedder: DenseEmbedder,
        rrf_k: int = DEFAULT_RRF_K,
        alpha: float = DEFAULT_HYBRID_ALPHA
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        strategy: Literal["hybrid", "dense", "sparse"] = "hybrid",
        fusion_method: Literal["rrf", "linear"] = "rrf"
    ) -> List[Tuple[TextChunk, float]]:
        """
        Retrieves top_k chunks using specified strategy.
        Returns list of (TextChunk, fused_score).
        """
        if strategy == "dense":
            q_vec = self.embedder.embed_query(query)
            return self.vector_store.search(q_vec, top_k=top_k)

        if strategy == "sparse":
            return self.bm25_retriever.search(query, top_k=top_k)

        # Hybrid strategy: retrieve candidate pool from both
        candidate_k = max(top_k * 3, 15)
        q_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(q_vec, top_k=candidate_k)
        sparse_results = self.bm25_retriever.search(query, top_k=candidate_k)

        if fusion_method == "rrf":
            return self._fuse_rrf(dense_results, sparse_results, top_k=top_k)
        else:
            return self._fuse_linear(dense_results, sparse_results, top_k=top_k)

    def _fuse_rrf(
        self,
        dense_results: List[Tuple[TextChunk, float]],
        sparse_results: List[Tuple[TextChunk, float]],
        top_k: int
    ) -> List[Tuple[TextChunk, float]]:
        """Reciprocal Rank Fusion."""
        chunk_map: Dict[str, TextChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # Dense ranks
        for rank, (chunk, _) in enumerate(dense_results):
            chunk_map[chunk.chunk_id] = chunk
            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + self.alpha * score

        # Sparse ranks
        for rank, (chunk, _) in enumerate(sparse_results):
            chunk_map[chunk.chunk_id] = chunk
            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 - self.alpha) * score

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [(chunk_map[cid], float(rrf_scores[cid])) for cid in sorted_ids]

    def _fuse_linear(
        self,
        dense_results: List[Tuple[TextChunk, float]],
        sparse_results: List[Tuple[TextChunk, float]],
        top_k: int
    ) -> List[Tuple[TextChunk, float]]:
        """Normalized min-max linear score combination."""
        chunk_map: Dict[str, TextChunk] = {}
        dense_dict: Dict[str, float] = {}
        sparse_dict: Dict[str, float] = {}

        for chunk, score in dense_results:
            chunk_map[chunk.chunk_id] = chunk
            dense_dict[chunk.chunk_id] = score

        for chunk, score in sparse_results:
            chunk_map[chunk.chunk_id] = chunk
            sparse_dict[chunk.chunk_id] = score

        def min_max_norm(score_dict: Dict[str, float]) -> Dict[str, float]:
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 1.0 for k in score_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

        norm_dense = min_max_norm(dense_dict)
        norm_sparse = min_max_norm(sparse_dict)

        combined: Dict[str, float] = {}
        for cid, chunk in chunk_map.items():
            d_score = norm_dense.get(cid, 0.0)
            s_score = norm_sparse.get(cid, 0.0)
            combined[cid] = self.alpha * d_score + (1.0 - self.alpha) * s_score

        sorted_ids = sorted(combined.keys(), key=lambda cid: combined[cid], reverse=True)[:top_k]
        return [(chunk_map[cid], float(combined[cid])) for cid in sorted_ids]
