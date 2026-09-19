"""
Unit tests for document loading, chunking, and dense/sparse/hybrid retrieval.
"""

import unittest
from pathlib import Path
from src.config import SAMPLE_DOCS_DIR
from src.ingestion.document_loader import DocumentLoader, Document
from src.ingestion.chunker import RecursiveChunker, TextChunk
from src.retrieval.embedder import DenseEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


class TestRetrievalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.docs = DocumentLoader.load_directory(SAMPLE_DOCS_DIR)
        cls.chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
        cls.chunks = cls.chunker.chunk_documents(cls.docs)
        cls.embedder = DenseEmbedder()

    def test_document_loader(self):
        self.assertGreater(len(self.docs), 0, "Should load sample documents")
        for doc in self.docs:
            self.assertGreater(len(doc.content), 50)
            self.assertTrue(doc.doc_id)

    def test_chunker(self):
        self.assertGreater(len(self.chunks), 0, "Should produce text chunks")
        for c in self.chunks:
            self.assertIsInstance(c, TextChunk)
            self.assertGreater(len(c.text), 10)
            self.assertGreaterEqual(c.token_count, 1)

    def test_dense_vector_store(self):
        texts = [c.text for c in self.chunks[:5]]
        embeddings = self.embedder.embed_texts(texts)
        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.shape[1], self.embedder.dimension)

        v_store = VectorStore(dimension=self.embedder.dimension)
        v_store.add_chunks(self.chunks[:5], embeddings)
        q_vec = self.embedder.embed_query("quantum computing qubits")
        results = v_store.search(q_vec, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0][0], TextChunk)
        self.assertIsInstance(results[0][1], float)

    def test_bm25_retriever(self):
        bm25 = BM25Retriever()
        bm25.index_chunks(self.chunks)
        results = bm25.search("Transformer attention mechanism", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0][0], TextChunk)

    def test_hybrid_retriever(self):
        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        v_store = VectorStore(dimension=self.embedder.dimension)
        v_store.add_chunks(self.chunks, embeddings)

        bm25 = BM25Retriever()
        bm25.index_chunks(self.chunks)

        hybrid = HybridRetriever(v_store, bm25, self.embedder)
        results = hybrid.retrieve("Shor's algorithm prime factorization", top_k=3, strategy="hybrid")
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
