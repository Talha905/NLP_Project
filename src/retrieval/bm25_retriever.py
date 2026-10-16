"""
BM25 Lexical Retriever supporting rank_bm25 and a built-in Python BM25 Okapi implementation.
"""

import math
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional
from ..ingestion.chunker import TextChunk


class BM25Retriever:
    """
    BM25 Okapi lexical retriever for keyword and exact-match search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []
        self.use_rank_bm25 = False
        self.bm25_engine = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace & alphanumeric tokenizer."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        return [w for w in clean.split() if len(w) > 1]

    def index_chunks(self, chunks: List[TextChunk]):
        """Indexes text chunks for BM25 retrieval."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            return

        tokenized_corpus = [self.tokenize(c.text) for c in chunks]

        # Try using rank_bm25 if available
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_engine = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            self.use_rank_bm25 = True
            return
        except Exception:
            self.use_rank_bm25 = False

        # Pure Python BM25 Okapi fallback
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]

        self.doc_freqs = {}
        for tf in self.term_freqs:
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def search(self, query: str, top_k: int = 3, doc_id: Optional[str] = None) -> List[Tuple[TextChunk, float]]:
        """Searches for top_k chunks by BM25 score, optionally scoped to a single document."""
        if not self.chunks:
            return []

        # Determine target candidate indices
        valid_indices = list(range(len(self.chunks)))
        if doc_id:
            from pathlib import Path
            valid_indices = [
                i for i, c in enumerate(self.chunks)
                if c.doc_id == doc_id or c.source_filename == doc_id or Path(c.source_filename).name == Path(doc_id).name
            ]
            if not valid_indices:
                print(f"[BM25] [FILTER] No chunks match doc_id='{doc_id}'")
                return []
            top_k = min(top_k, len(valid_indices))
        else:
            top_k = min(top_k, len(self.chunks))

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [(self.chunks[i], 0.0) for i in valid_indices[:top_k]]

        # rank_bm25 path
        if self.use_rank_bm25 and self.bm25_engine is not None:
            all_scores = self.bm25_engine.get_scores(query_tokens)
            sorted_indices = sorted(valid_indices, key=lambda i: all_scores[i], reverse=True)[:top_k]
            res = [(self.chunks[i], float(all_scores[i])) for i in sorted_indices]
            if doc_id:
                print(f"[BM25] [FILTER] Scoped to '{doc_id}': retrieved {len(res)} chunks")
            return res

        # Pure Python BM25 calculation
        all_scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for i in valid_indices:
                tf = self.term_freqs[i].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lens[i]
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    all_scores[i] += idf * (num / denom)

        sorted_indices = sorted(valid_indices, key=lambda i: all_scores[i], reverse=True)[:top_k]
        res = [(self.chunks[i], float(all_scores[i])) for i in sorted_indices]
        if doc_id:
            print(f"[BM25] [FILTER] Scoped to '{doc_id}': retrieved {len(res)} chunks")
        return res
