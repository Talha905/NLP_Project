"""
Unit tests for Model Capability Analysis under constant retrieval conditions.
"""

import unittest
import json
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


class TestModelCapabilityAnalysis(unittest.TestCase):

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
            cls.benchmark_data = json.load(f)[:5]  # Quick 5 items

    def test_compare_models_constant_retrieval(self):
        # We test with two model identifiers using local deterministic generator
        models = ["qwen2.5:3b", "llama3.2:latest"]
        res = self.ablation.compare_models(
            benchmark_data=self.benchmark_data,
            models=models,
            top_k=3,
            strategy="hybrid",
            force_fallback=True
        )

        self.assertIn("summary_df", res)
        self.assertIn("subtype_df", res)
        self.assertIn("model_results", res)
        self.assertIn("discordant_cases", res)

        summary_df = res["summary_df"]
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(summary_df["Model"].tolist(), models)

        # Retrieval Failure Rate MUST be identical because retrieval is held constant
        ret_rate_m1 = summary_df.loc[summary_df["Model"] == models[0], "Retrieval_Failure_Rate"].values[0]
        ret_rate_m2 = summary_df.loc[summary_df["Model"] == models[1], "Retrieval_Failure_Rate"].values[0]
        self.assertAlmostEqual(ret_rate_m1, ret_rate_m2, places=4)

        # Check model_samples dictionary
        self.assertEqual(len(res["model_samples"][models[0]]), len(self.benchmark_data))
        self.assertEqual(len(res["model_samples"][models[1]]), len(self.benchmark_data))

        # Ensure identical retrieved chunks were provided to both models for each query
        for s1, s2 in zip(res["model_samples"][models[0]], res["model_samples"][models[1]]):
            self.assertEqual(s1.retrieved_chunk_ids, s2.retrieved_chunk_ids)


if __name__ == "__main__":
    unittest.main()
