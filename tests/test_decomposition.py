"""
Unit tests for mathematical error decomposition and failure taxonomy classification.
"""

import unittest
from src.retrieval.embedder import DenseEmbedder
from src.ingestion.chunker import TextChunk
from src.evaluation.metrics import calculate_token_f1, calculate_rouge_l, calculate_semantic_similarity
from src.evaluation.error_decomposer import ErrorDecomposer
from src.evaluation.failure_classifier import FailureTaxonomyAdvisor


class TestErrorDecomposition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder = DenseEmbedder()
        cls.decomposer = ErrorDecomposer(cls.embedder)

    def test_metrics(self):
        f1 = calculate_token_f1("The cat sat on the mat", "The cat was on the mat")
        self.assertGreater(f1, 0.5)

        rouge = calculate_rouge_l("The cat sat on the mat", "The cat sat on the mat")
        self.assertAlmostEqual(rouge, 1.0, places=2)

        sim = calculate_semantic_similarity("quantum computing", "quantum computation", self.embedder)
        self.assertGreater(sim, 0.5)

    def test_success_quadrant(self):
        meta = {
            "id": "t1",
            "question": "What is the formula for attention?",
            "gold_answer": "softmax((Q * K^T) / sqrt(d_k)) * V",
            "gold_evidence": "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            "question_type": "factoid",
            "difficulty": "easy",
            "domain": "nlp"
        }
        chunks = [
            TextChunk(
                chunk_id="c1", doc_id="d1",
                text="The formula is Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
                start_char=0, end_char=50, token_count=12
            )
        ]
        gen_out = {
            "answer": "The formula for attention is softmax((Q * K^T) / sqrt(d_k)) * V",
            "generation_time_ms": 10,
            "model": "test"
        }
        sample = self.decomposer.evaluate_sample(meta, chunks, gen_out)
        self.assertEqual(sample.quadrant, "SUCCESS")
        self.assertTrue(sample.retrieval_success)
        self.assertTrue(sample.generation_success)

    def test_generation_failure_quadrant(self):
        meta = {
            "id": "t2",
            "question": "What is the formula for attention?",
            "gold_answer": "softmax((Q * K^T) / sqrt(d_k)) * V",
            "gold_evidence": "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            "question_type": "factoid",
            "difficulty": "easy",
            "domain": "nlp"
        }
        # Evidence IS retrieved
        chunks = [
            TextChunk(
                chunk_id="c1", doc_id="d1",
                text="The formula is Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
                start_char=0, end_char=50, token_count=12
            )
        ]
        # But generator hallucinated or failed
        gen_out = {
            "answer": "The formula for attention is sigmoid(W * x + b) * tanh(c)",
            "generation_time_ms": 10,
            "model": "test"
        }
        sample = self.decomposer.evaluate_sample(meta, chunks, gen_out)
        self.assertEqual(sample.quadrant, "GENERATION_FAILURE")
        self.assertTrue(sample.retrieval_success)
        self.assertFalse(sample.generation_success)

    def test_retrieval_failure_quadrant(self):
        meta = {
            "id": "t3",
            "question": "What is the formula for attention?",
            "gold_answer": "softmax((Q * K^T) / sqrt(d_k)) * V",
            "gold_evidence": "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            "question_type": "factoid",
            "difficulty": "easy",
            "domain": "nlp"
        }
        # Unrelated chunk retrieved (Evidence is missing)
        chunks = [
            TextChunk(
                chunk_id="c_unrelated", doc_id="d_other",
                text="Photosynthesis occurs in plant chloroplasts using sunlight to convert CO2 into glucose.",
                start_char=0, end_char=50, token_count=12
            )
        ]
        # Generator couldn't answer correctly
        gen_out = {
            "answer": "The context does not contain sufficient evidence to answer this question.",
            "generation_time_ms": 10,
            "model": "test"
        }
        sample = self.decomposer.evaluate_sample(meta, chunks, gen_out)
        self.assertEqual(sample.quadrant, "RETRIEVAL_FAILURE")
        self.assertFalse(sample.retrieval_success)
        self.assertFalse(sample.generation_success)

    def test_mathematical_decomposition_sum(self):
        # Create a mock set of samples and verify alpha_R + alpha_G = 1.0
        meta = {
            "id": "t", "question": "q", "gold_answer": "a", "gold_evidence": "e",
            "question_type": "f", "difficulty": "e", "domain": "d"
        }
        c_good = [TextChunk(chunk_id="1", doc_id="d", text="e", start_char=0, end_char=1, token_count=1)]
        c_bad = [TextChunk(chunk_id="2", doc_id="d", text="xyz entirely unrelated text", start_char=0, end_char=1, token_count=1)]

        s1 = self.decomposer.evaluate_sample(meta, c_good, {"answer": "a", "generation_time_ms": 1, "model": "m"}) # Success
        s2 = self.decomposer.evaluate_sample(meta, c_good, {"answer": "wrong hallucinated", "generation_time_ms": 1, "model": "m"}) # Gen fail
        s3 = self.decomposer.evaluate_sample(meta, c_bad, {"answer": "wrong refused", "generation_time_ms": 1, "model": "m"}) # Ret fail

        res = self.decomposer.aggregate_results([s1, s2, s3])
        self.assertAlmostEqual(res.retrieval_attribution + res.generation_attribution, 1.0, places=4)
        self.assertGreater(res.total_error_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
