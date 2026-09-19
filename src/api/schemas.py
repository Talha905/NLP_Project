"""
FastAPI Pydantic Schemas for RAG and Error Decomposition.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    strategy: str = "hybrid"  # hybrid, dense, sparse
    prompt_style: str = "grounded"  # grounded, cot
    model_name: Optional[str] = None
    force_fallback: bool = False


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_filename: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    model: str
    generation_time_ms: int
    is_fallback: bool
    retrieved_chunks: List[RetrievedChunkSchema]


class SingleEvaluationRequest(BaseModel):
    question: str
    gold_answer: str
    gold_evidence: str
    top_k: int = 3
    strategy: str = "hybrid"
    question_type: str = "factoid"
    difficulty: str = "medium"
    force_fallback: bool = False


class BenchmarkEvaluationRequest(BaseModel):
    benchmark_name: str = "quick_10"  # quick_10, full_50, or custom
    custom_benchmark: Optional[List[Dict[str, Any]]] = None
    top_k: int = 3
    strategy: str = "hybrid"
    prompt_style: str = "grounded"
    force_fallback: bool = True


class IndexStatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    dimension: int
    backend: str
    is_indexed: bool


class ModelComparisonRequest(BaseModel):
    benchmark_name: str = "quick_10"
    custom_benchmark: Optional[List[Dict[str, Any]]] = None
    models: List[str] = ["qwen2.5:3b", "llama3.2:latest"]
    top_k: int = 3
    strategy: str = "hybrid"
    prompt_style: str = "grounded"
    force_fallback: bool = False

