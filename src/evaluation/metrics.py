"""
Core NLP and Information Retrieval evaluation metrics.
Includes Token F1, ROUGE-L, Semantic Cosine Similarity, Recall@K, and Precision@K.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from ..ingestion.chunker import TextChunk
from ..retrieval.embedder import DenseEmbedder
from ..config import RETRIEVAL_SIMILARITY_THRESHOLD, ANSWER_SIMILARITY_THRESHOLD, TOKEN_F1_THRESHOLD


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def calculate_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = collections_counter_intersection(pred_tokens, gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def collections_counter_intersection(list1: List[str], list2: List[str]) -> Dict[str, int]:
    """Helper to count token overlap."""
    from collections import Counter
    c1 = Counter(list1)
    c2 = Counter(list2)
    return {k: min(c1[k], c2[k]) for k in c1 if k in c2}


def calculate_rouge_l(prediction: str, ground_truth: str) -> float:
    """
    Computes ROUGE-L score (Longest Common Subsequence).
    Uses rouge_score if available; falls back to pure Python LCS.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        pass

    # Pure Python LCS fallback
    s1 = normalize_text(prediction).split()
    s2 = normalize_text(ground_truth).split()
    if not s1 or not s2:
        return 1.0 if s1 == s2 else 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0:
        return 0.0
    return float((2 * prec * rec) / (prec + rec))


def calculate_semantic_similarity(text1: str, text2: str, embedder: DenseEmbedder) -> float:
    """Computes cosine similarity between two texts using the DenseEmbedder."""
    if not text1.strip() or not text2.strip():
        return 0.0

    v1 = embedder.embed_query(text1)
    v2 = embedder.embed_query(text2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return max(0.0, min(1.0, sim))


def evaluate_retrieval(
    retrieved_chunks: List[TextChunk],
    gold_evidence: str,
    embedder: DenseEmbedder,
    similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates whether the gold evidence was successfully retrieved.
    Checks:
    1. Lexical inclusion / phrase overlap
    2. Highest semantic similarity to gold evidence
    3. Rank of first matching chunk
    """
    if not retrieved_chunks:
        return {
            "retrieval_success": False,
            "max_similarity": 0.0,
            "hit_rank": -1,
            "hit_chunk_id": None,
            "precision_at_k": 0.0,
            "overlap_ratio": 0.0
        }

    # Special handling for adversarial / unanswerable questions
    if "UNANSWERABLE" in gold_evidence.upper():
        return {
            "retrieval_success": False,  # Gold evidence genuinely does not exist in corpus
            "max_similarity": 0.0,
            "hit_rank": -1,
            "hit_chunk_id": None,
            "precision_at_k": 0.0,
            "overlap_ratio": 0.0,
            "is_unanswerable": True
        }

    gold_norm = normalize_text(gold_evidence)
    gold_tokens = set(gold_norm.split())

    max_sim = 0.0
    hit_rank = -1
    hit_chunk_id = None
    relevant_count = 0

    gold_vec = embedder.embed_query(gold_evidence)
    gold_vec_norm = np.linalg.norm(gold_vec)

    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_norm = normalize_text(chunk.text)
        chunk_tokens = set(chunk_norm.split())

        # Exact substring or high token intersection
        token_overlap = len(gold_tokens.intersection(chunk_tokens)) / max(1, len(gold_tokens))
        exact_sub = gold_norm in chunk_norm or chunk_norm in gold_norm

        # Semantic cosine similarity
        c_vec = embedder.embed_query(chunk.text)
        c_norm = np.linalg.norm(c_vec)
        cos_sim = 0.0
        if gold_vec_norm > 0 and c_norm > 0:
            cos_sim = float(np.dot(gold_vec, c_vec) / (gold_vec_norm * c_norm))

        # Combined match logic
        is_hit = exact_sub or (token_overlap >= 0.45) or (cos_sim >= similarity_threshold)

        if cos_sim > max_sim:
            max_sim = cos_sim

        if is_hit:
            relevant_count += 1
            if hit_rank == -1:
                hit_rank = rank
                hit_chunk_id = chunk.chunk_id

    precision = relevant_count / len(retrieved_chunks)
    retrieval_success = (hit_rank != -1)

    return {
        "retrieval_success": retrieval_success,
        "max_similarity": float(max_sim),
        "hit_rank": hit_rank,
        "hit_chunk_id": hit_chunk_id,
        "precision_at_k": float(precision),
        "relevant_chunks_count": relevant_count
    }


def evaluate_generation(
    prediction: str,
    gold_answer: str,
    embedder: DenseEmbedder,
    answer_sim_threshold: float = ANSWER_SIMILARITY_THRESHOLD,
    f1_threshold: float = TOKEN_F1_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates whether generated answer is correct compared to gold answer.
    Combines Token F1, ROUGE-L, and Semantic Cosine Similarity.
    """
    # Check for refusal responses
    refusal_keywords = [
        "does not contain sufficient evidence",
        "not contain sufficient evidence",
        "insufficient evidence",
        "unanswerable",
        "not mentioned in the context",
        "context lacks",
        "cannot be answered"
    ]
    is_refusal = any(kw in prediction.lower() for kw in refusal_keywords)

    # Check if gold answer itself is a refusal (e.g. for adversarial query)
    gold_is_unanswerable = any(kw in gold_answer.lower() for kw in ["unanswerable", "not present", "not mentioned"])

    if gold_is_unanswerable:
        # If ground truth was unanswerable, refusing is a SUCCESS
        is_correct = is_refusal
        return {
            "generation_success": is_correct,
            "token_f1": 1.0 if is_correct else 0.0,
            "rouge_l": 1.0 if is_correct else 0.0,
            "semantic_similarity": 1.0 if is_correct else 0.0,
            "composite_score": 1.0 if is_correct else 0.0,
            "is_refusal": is_refusal,
            "is_unanswerable_gold": True
        }

    if is_refusal:
        # LLM refused to answer when answer was expected
        return {
            "generation_success": False,
            "token_f1": 0.0,
            "rouge_l": 0.0,
            "semantic_similarity": 0.0,
            "composite_score": 0.0,
            "is_refusal": True,
            "is_unanswerable_gold": False
        }

    token_f1 = calculate_token_f1(prediction, gold_answer)
    rouge_l = calculate_rouge_l(prediction, gold_answer)
    semantic_sim = calculate_semantic_similarity(prediction, gold_answer, embedder)

    # Composite correctness: weighted combination
    composite = 0.4 * semantic_sim + 0.3 * token_f1 + 0.3 * rouge_l
    generation_success = (semantic_sim >= answer_sim_threshold) or (token_f1 >= f1_threshold) or (composite >= 0.50)

    return {
        "generation_success": bool(generation_success),
        "token_f1": float(token_f1),
        "rouge_l": float(rouge_l),
        "semantic_similarity": float(semantic_sim),
        "composite_score": float(composite),
        "is_refusal": False,
        "is_unanswerable_gold": False
    }
