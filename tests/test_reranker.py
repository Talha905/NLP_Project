import unittest
from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.chunker import TextChunk


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = CrossEncoderReranker()
        self.chunk1 = TextChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="The transformer architecture relies entirely on attention mechanisms to draw global dependencies.",
            start_char=0,
            end_char=90,
            token_count=15
        )
        self.chunk2 = TextChunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            text="Apples and bananas are nutritious fruits that contain dietary fiber and essential vitamins.",
            start_char=91,
            end_char=180,
            token_count=14
        )

    def test_reranker_relevance_ordering(self):
        query = "How does attention work in transformer architectures?"
        candidates = [(self.chunk2, 0.9), (self.chunk1, 0.5)]  # Intentionally wrong 1st-stage order
        reranked = self.reranker.rerank(query, candidates, top_k=2)

        self.assertEqual(len(reranked), 2)
        # Cross encoder should place the transformer chunk first
        self.assertEqual(reranked[0][0].chunk_id, "chunk_1")
        self.assertGreater(reranked[0][1], reranked[1][1])


if __name__ == "__main__":
    unittest.main()
