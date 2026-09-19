"""
End-to-end integration test running benchmark evaluation through the full RAG pipeline.
"""

import json
import unittest
from pathlib import Path
from src.config import SAMPLE_DOCS_DIR, BENCHMARKS_DIR
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import RecursiveChunker
from src.retrieval.embedder import DenseEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LocalLLMClient
from src.evaluation.error_decomposer import ErrorDecomposer
from src.evaluation.ablation_engine import AblationEngine


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        docs = DocumentLoader.load_directory(SAMPLE_DOCS_DIR)
        chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)
        embedder = DenseEmbedder()

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        v_store = VectorStore(dimension=embedder.dimension)
        v_store.add_chunks(chunks, embeddings)

        bm25 = BM25Retriever()
        bm25.index_chunks(chunks)

        hybrid = HybridRetriever(v_store, bm25, embedder)
        llm = LocalLLMClient()
        decomposer = ErrorDecomposer(embedder)
        cls.ablation = AblationEngine(hybrid, llm, decomposer)

        bench_path = BENCHMARKS_DIR / "quick_test_benchmark_10.json"
        with open(bench_path, "r", encoding="utf-8") as f:
            cls.benchmark_data = json.load(f)

    def test_run_quick_benchmark(self):
        res = self.ablation.run_benchmark_evaluation(
            benchmark_data=self.benchmark_data,
            top_k=3,
            strategy="hybrid",
            force_fallback=True
        )
        self.assertEqual(res.total_samples, 10)
        self.assertGreaterEqual(res.success_count, 1)
        self.assertEqual(len(res.samples), 10)
        # Verify valid attribution
        if res.total_error_rate > 0:
            self.assertAlmostEqual(res.retrieval_attribution + res.generation_attribution, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
