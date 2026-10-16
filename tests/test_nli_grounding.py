import unittest
from src.evaluation.nli_grounding import GroundingEvaluator
from src.ingestion.chunker import TextChunk


class TestNLIGroundingEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = GroundingEvaluator()
        self.chunks = [
            TextChunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                text="The multi-head attention mechanism in Transformer models computes attention weights in parallel across different representation subspaces.",
                start_char=0,
                end_char=130,
                token_count=20
            )
        ]

    def test_entailed_claim(self):
        answer = "Multi-head attention calculates attention weights in parallel across distinct representation subspaces."
        res = self.evaluator.evaluate_groundedness(answer, self.chunks)
        self.assertTrue(res["is_faithful"])
        self.assertGreaterEqual(res["faithfulness_score"], 0.60)
        self.assertEqual(len(res["unsupported_claims"]), 0)
        self.assertFalse(res["contradiction_detected"])

    def test_contradicted_claim(self):
        answer = "The multi-head attention mechanism strictly executes attention calculations sequentially without any parallel subspaces."
        res = self.evaluator.evaluate_groundedness(answer, self.chunks)
        self.assertTrue(res["contradiction_detected"] or len(res["unsupported_claims"]) > 0)
        self.assertLess(res["faithfulness_score"], 0.50)

    def test_hallucinated_claim(self):
        answer = "Transformers were invented in ancient Egypt during the reign of Pharaoh Ramses."
        res = self.evaluator.evaluate_groundedness(answer, self.chunks)
        self.assertFalse(res["is_faithful"])
        self.assertIn(answer, res["unsupported_claims"])


if __name__ == "__main__":
    unittest.main()
