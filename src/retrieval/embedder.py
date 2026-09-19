"""
Embedding module supporting SentenceTransformers with PyTorch mean-pooling fallback.
Optimized for local CPU execution with zero API cost.
"""

from typing import List
import numpy as np
import torch
from ..config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM


class DenseEmbedder:
    """
    Computes dense vector representations for queries and document chunks.
    Tries SentenceTransformers first; falls back to raw PyTorch + Transformers mean pooling;
    and has a deterministic n-gram vectorizer fallback for completely offline environments.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIM
        self.backend = None
        self._load_model()

    def _load_model(self):
        # 1. Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.backend = "sentence-transformers"
            self.dimension = self.model.get_sentence_embedding_dimension()
            return
        except Exception as e:
            # print(f"SentenceTransformer not loaded ({e}), falling back to transformers...")
            pass

        # 2. Try raw transformers + PyTorch
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.raw_model = AutoModel.from_pretrained(self.model_name)
            self.raw_model.eval()
            self.backend = "transformers-pytorch"
            self.dimension = self.raw_model.config.hidden_size
            return
        except Exception as e:
            # print(f"Transformers model not loaded ({e}), using deterministic hashing fallback...")
            pass

        # 3. Deterministic n-gram projection fallback (offline fallback)
        self.backend = "deterministic-projection"
        self.dimension = EMBEDDING_DIM

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of text strings into a 2D float32 numpy array [N, D]."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # SentenceTransformers path
        if self.backend == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.astype(np.float32)

        # Raw PyTorch Transformers path
        if self.backend == "transformers-pytorch":
            embeddings_list = []
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.raw_model(**encoded)
                    token_embeddings = outputs[0]  # First element has hidden states
                    input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask
                    # L2 normalize
                    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                    embeddings_list.append(normalized.cpu().numpy())
            return np.vstack(embeddings_list).astype(np.float32)

        # Deterministic projection path (stable subword n-gram hashing)
        import hashlib
        import re

        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = re.findall(r"\b\w+\b", text.lower())
            for w in words:
                # Word hash
                w_h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dimension
                vec[w_h] += 2.0
                # Subword n-grams (3 to 5 chars) to capture morphological variants
                padded = f"<{w}>"
                for n in (3, 4, 5):
                    for i in range(len(padded) - n + 1):
                        sub = padded[i:i + n]
                        sub_h = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16) % self.dimension
                        vec[sub_h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 numpy array [D]."""
        res = self.embed_texts([query])
        return res[0]
