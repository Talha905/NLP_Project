"""
Vector Store implementation using FAISS with pure NumPy fallback.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from ..ingestion.chunker import TextChunk


class VectorStore:
    """
    Vector index supporting FAISS IndexFlatIP with a pure NumPy fallback.
    """

    def __init__(self, dimension: int = 384, use_faiss: bool = True):
        self.dimension = dimension
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.use_faiss = use_faiss
        self._init_index()

    def _init_index(self):
        if self.use_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                return
            except Exception:
                self.faiss_index = None
        self.use_faiss = False

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray):
        """Adds chunks and their normalized dense embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")

        if len(chunks) == 0:
            return

        # Ensure float32 and normalized
        emb = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        emb = emb / norms

        start_idx = len(self.chunks)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(emb)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Searches for top_k most similar chunks.
        Returns list of (TextChunk, similarity_score).
        """
        if len(self.chunks) == 0 or self.embeddings is None:
            return []

        top_k = min(top_k, len(self.chunks))

        # Normalize query vector
        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            q = q / q_norm

        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            return results

        # NumPy Cosine Similarity fallback
        scores = np.dot(self.embeddings, q.T).flatten()
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def save(self, storage_dir: str | Path):
        """Persists the vector store to disk."""
        path = Path(storage_dir)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "chunks": [c.model_dump() for c in self.chunks],
            "embeddings": self.embeddings
        }
        with open(path / "vector_store.pkl", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, storage_dir: str | Path) -> "VectorStore":
        """Loads a vector store from disk."""
        path = Path(storage_dir)
        file_path = path / "vector_store.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Vector store file not found: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        store = cls(dimension=data["dimension"])
        chunks = [TextChunk(**cd) for cd in data["chunks"]]
        if data["embeddings"] is not None and len(chunks) > 0:
            store.add_chunks(chunks, data["embeddings"])
        return store
