"""
Mathematical Error Decomposition Engine.
Decomposes total RAG failure into Retrieval Failures, Generation Failures,
and Parametric Memory / Spurious Recall.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .metrics import evaluate_retrieval, evaluate_generation
from .nli_grounding import GroundingEvaluator
from ..ingestion.chunker import TextChunk
from ..retrieval.embedder import DenseEmbedder


class EvaluationSample(BaseModel):
    """Encapsulates a single evaluated query instance with full diagnostics."""
    sample_id: str
    question: str
    gold_answer: str
    gold_evidence: str
    question_type: str
    difficulty: str
    domain: str

    # Retrieval results
    retrieved_chunk_ids: List[str]
    retrieval_success: bool
    retrieval_similarity: float
    retrieval_hit_rank: int
    retrieval_precision: float

    # Generation results
    generated_answer: str
    generation_success: bool
    token_f1: float
    rouge_l: float
    semantic_similarity: float
    is_refusal: bool
    generation_time_ms: int

    # Grounding & Faithfulness
    faithfulness_score: float
    is_faithful: bool
    contradiction_detected: bool

    # Error quadrant & taxonomy
    quadrant: str            # SUCCESS, GENERATION_FAILURE, RETRIEVAL_FAILURE, PARAMETRIC_SUCCESS
    primary_failure_cause: str
    failure_subtype: str
    diagnostic_explanation: str


class DecompositionResult(BaseModel):
    """Aggregate statistical breakdown of benchmark evaluation."""
    total_samples: int
    success_count: int
    generation_failure_count: int
    retrieval_failure_count: int
    parametric_success_count: int

    # Probabilities
    total_error_rate: float
    retrieval_attribution: float    # Alpha_R: proportion of error due to retrieval
    generation_attribution: float   # Alpha_G: proportion of error due to generation
    parametric_recall_rate: float   # Proportion of queries answered correctly without retrieval

    # Averages
    mean_token_f1: float
    mean_rouge_l: float
    mean_semantic_similarity: float
    mean_faithfulness: float
    mean_retrieval_precision: float
    mean_generation_time_ms: float

    # Subtype distribution
    subtype_breakdown: Dict[str, int]
    samples: List[EvaluationSample]


class ErrorDecomposer:
    """
    Orchestrates evaluation across retrieval and generation,
    mathematically partitions errors, and assigns diagnostic causes.
    """

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder
        self.grounding_evaluator = GroundingEvaluator(embedder)

    def evaluate_sample(
        self,
        sample_meta: Dict[str, Any],
        retrieved_chunks: List[TextChunk],
        generation_output: Dict[str, Any]
    ) -> EvaluationSample:
        """Evaluates a single end-to-end RAG question."""
        sample_id = str(sample_meta.get("id", "unknown"))
        question = sample_meta.get("question", "")
        gold_answer = sample_meta.get("gold_answer", "")
        gold_evidence = sample_meta.get("gold_evidence", "")
        question_type = sample_meta.get("question_type", "factoid")
        difficulty = sample_meta.get("difficulty", "medium")
        domain = sample_meta.get("domain", "general")

        generated_answer = generation_output.get("answer", "")
        gen_time = generation_output.get("generation_time_ms", 0)

        # 1. Retrieval Verification
        r_eval = evaluate_retrieval(retrieved_chunks, gold_evidence, self.embedder)
        retrieval_success = r_eval["retrieval_success"]

        # 2. Generation Verification
        g_eval = evaluate_generation(generated_answer, gold_answer, self.embedder)
        generation_success = g_eval["generation_success"]

        # 3. Grounding / Faithfulness
        grounding = self.grounding_evaluator.evaluate_groundedness(generated_answer, retrieved_chunks)

        # 4. Assign 4-Quadrant Decomposition
        if retrieval_success and generation_success:
            quadrant = "SUCCESS"
            primary_cause = "NONE"
            subtype = "NONE"
            explanation = "Evidence was retrieved in top-K and the LLM synthesized an accurate, grounded answer."
        elif retrieval_success and not generation_success:
            quadrant = "GENERATION_FAILURE"
            primary_cause = "GENERATION_FAILURE"
            # Subtype classification
            if g_eval["is_refusal"]:
                subtype = "CONSERVATIVE_REFUSAL"
                explanation = "Evidence was present in the retrieved context, but the LLM exhibited over-conservative refusal ('I don't know')."
            elif grounding["contradiction_detected"]:
                subtype = "CONTEXT_CONTRADICTION"
                explanation = "Evidence was present, but the LLM generated statements contradicting facts in the context."
            elif not grounding["is_faithful"]:
                subtype = "HALLUCINATION_UNSUPPORTED"
                explanation = "Evidence was present, but the LLM hallucinated ungrounded claims not supported by the context."
            elif question_type in ["multi-hop", "reasoning"]:
                subtype = "REASONING_SYNTHESIS_ERROR"
                explanation = "Evidence chunks were present, but the model failed to synthesize multi-hop logic correctly."
            else:
                subtype = "INCOMPLETE_ANSWER"
                explanation = "Evidence was present, but the answer was incomplete or low fidelity."
        elif not retrieval_success and not generation_success:
            quadrant = "RETRIEVAL_FAILURE"
            primary_cause = "RETRIEVAL_FAILURE"
            # Check retrieval subtype
            if r_eval.get("is_unanswerable", False):
                if g_eval["is_refusal"]:
                    quadrant = "SUCCESS"
                    primary_cause = "NONE"
                    subtype = "UNANSWERABLE_CORRECT_REFUSAL"
                    explanation = "Adversarial/unanswerable query: evidence does not exist and LLM appropriately refused."
                else:
                    subtype = "HALLUCINATION_UNSUPPORTED"
                    explanation = "Adversarial query: evidence does not exist in knowledge base, but LLM fabricated an answer."
            elif r_eval["max_similarity"] < 0.35:
                subtype = "MISSING_EVIDENCE"
                explanation = "Top-K chunks contained no semantic overlap with the required ground-truth evidence."
            elif r_eval["precision_at_k"] < 0.3:
                subtype = "NOISE_DILUTION"
                explanation = "Distractor chunks pushed evidence beyond ranking threshold or diluted retrieval fidelity."
            else:
                subtype = "TRUNCATION_BOUNDARY"
                explanation = "Partial context retrieved, but key factual terms were cut off across chunk boundaries."
        else:
            # R=0, G=1: Parametric / Spurious Success
            quadrant = "PARAMETRIC_SUCCESS"
            primary_cause = "PARAMETRIC_MEMORY_LEAKAGE"
            subtype = "PARAMETRIC_RECALL"
            explanation = "Evidence was missing from retrieved context, but the LLM answered correctly using pre-trained parametric memory."

        return EvaluationSample(
            sample_id=sample_id,
            question=question,
            gold_answer=gold_answer,
            gold_evidence=gold_evidence,
            question_type=question_type,
            difficulty=difficulty,
            domain=domain,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
            retrieval_success=retrieval_success,
            retrieval_similarity=r_eval["max_similarity"],
            retrieval_hit_rank=r_eval["hit_rank"],
            retrieval_precision=r_eval["precision_at_k"],
            generated_answer=generated_answer,
            generation_success=generation_success,
            token_f1=g_eval["token_f1"],
            rouge_l=g_eval["rouge_l"],
            semantic_similarity=g_eval["semantic_similarity"],
            is_refusal=g_eval["is_refusal"],
            generation_time_ms=gen_time,
            faithfulness_score=grounding["faithfulness_score"],
            is_faithful=grounding["is_faithful"],
            contradiction_detected=grounding["contradiction_detected"],
            quadrant=quadrant,
            primary_failure_cause=primary_cause,
            failure_subtype=subtype,
            diagnostic_explanation=explanation
        )

    def aggregate_results(self, samples: List[EvaluationSample]) -> DecompositionResult:
        """Computes statistical decomposition metrics from evaluated samples."""
        total = len(samples)
        if total == 0:
            return DecompositionResult(
                total_samples=0,
                success_count=0,
                generation_failure_count=0,
                retrieval_failure_count=0,
                parametric_success_count=0,
                total_error_rate=0.0,
                retrieval_attribution=0.0,
                generation_attribution=0.0,
                parametric_recall_rate=0.0,
                mean_token_f1=0.0,
                mean_rouge_l=0.0,
                mean_semantic_similarity=0.0,
                mean_faithfulness=0.0,
                mean_retrieval_precision=0.0,
                mean_generation_time_ms=0.0,
                subtype_breakdown={},
                samples=[]
            )

        successes = sum(1 for s in samples if s.quadrant == "SUCCESS")
        gen_fails = sum(1 for s in samples if s.quadrant == "GENERATION_FAILURE")
        ret_fails = sum(1 for s in samples if s.quadrant == "RETRIEVAL_FAILURE")
        param_succ = sum(1 for s in samples if s.quadrant == "PARAMETRIC_SUCCESS")

        total_errors = gen_fails + ret_fails
        total_error_rate = total_errors / total

        if total_errors > 0:
            alpha_r = ret_fails / total_errors
            alpha_g = gen_fails / total_errors
        else:
            alpha_r = 0.0
            alpha_g = 0.0

        param_rate = param_succ / total

        # Subtype counts
        subtypes: Dict[str, int] = {}
        for s in samples:
            if s.failure_subtype != "NONE":
                subtypes[s.failure_subtype] = subtypes.get(s.failure_subtype, 0) + 1

        return DecompositionResult(
            total_samples=total,
            success_count=successes,
            generation_failure_count=gen_fails,
            retrieval_failure_count=ret_fails,
            parametric_success_count=param_succ,
            total_error_rate=float(total_error_rate),
            retrieval_attribution=float(alpha_r),
            generation_attribution=float(alpha_g),
            parametric_recall_rate=float(param_rate),
            mean_token_f1=float(sum(s.token_f1 for s in samples) / total),
            mean_rouge_l=float(sum(s.rouge_l for s in samples) / total),
            mean_semantic_similarity=float(sum(s.semantic_similarity for s in samples) / total),
            mean_faithfulness=float(sum(s.faithfulness_score for s in samples) / total),
            mean_retrieval_precision=float(sum(s.retrieval_precision for s in samples) / total),
            mean_generation_time_ms=float(sum(s.generation_time_ms for s in samples) / total),
            subtype_breakdown=subtypes,
            samples=samples
        )
