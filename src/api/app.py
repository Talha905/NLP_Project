"""
FastAPI REST application for RAG Error Decomposition & Failure Analysis.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ..config import SAMPLE_DOCS_DIR, BENCHMARKS_DIR, INDEX_DIR, DEFAULT_TOP_K
from ..ingestion.document_loader import DocumentLoader
from ..ingestion.chunker import RecursiveChunker
from ..retrieval.embedder import DenseEmbedder
from ..retrieval.vector_store import VectorStore
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generation.llm_client import LocalLLMClient
from ..evaluation.error_decomposer import ErrorDecomposer
from ..evaluation.ablation_engine import AblationEngine
from .schemas import (
    QueryRequest, QueryResponse, RetrievedChunkSchema,
    SingleEvaluationRequest, BenchmarkEvaluationRequest, IndexStatusResponse,
    ModelComparisonRequest
)

# Global runtime state
state: Dict[str, Any] = {
    "embedder": None,
    "vector_store": None,
    "bm25_retriever": None,
    "hybrid_retriever": None,
    "llm_client": None,
    "decomposer": None,
    "ablation_engine": None,
    "chunks": [],
    "is_indexed": False
}


def build_system_index(doc_dir: Path = SAMPLE_DOCS_DIR):
    """Loads documents, chunks, computes embeddings, and builds index."""
    embedder = state["embedder"] or DenseEmbedder()
    state["embedder"] = embedder

    docs = DocumentLoader.load_directory(doc_dir)
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_documents(docs)

    if not chunks:
        return False

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    v_store = VectorStore(dimension=embedder.dimension)
    v_store.add_chunks(chunks, embeddings)

    bm25 = BM25Retriever()
    bm25.index_chunks(chunks)

    hybrid = HybridRetriever(v_store, bm25, embedder)
    llm = LocalLLMClient()
    decomposer = ErrorDecomposer(embedder)
    ablation = AblationEngine(hybrid, llm, decomposer)

    state.update({
        "vector_store": v_store,
        "bm25_retriever": bm25,
        "hybrid_retriever": hybrid,
        "llm_client": llm,
        "decomposer": decomposer,
        "ablation_engine": ablation,
        "chunks": chunks,
        "is_indexed": True
    })
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize embedder and build default index
    try:
        build_system_index()
    except Exception as e:
        print(f"Warning during startup indexing: {e}")
    yield


app = FastAPI(
    title="RAG Error Decomposition and Failure Analysis API",
    version="1.0.0",
    description="API for decomposing RAG failures into Retrieval vs Generation surfaces.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "is_indexed": state["is_indexed"],
        "total_chunks": len(state["chunks"]),
        "ollama_active": state["llm_client"].is_ollama_available if state["llm_client"] else False
    }


@app.get("/index/status", response_model=IndexStatusResponse)
def index_status():
    dim = state["embedder"].dimension if state["embedder"] else 384
    backend = state["embedder"].backend if state["embedder"] else "none"
    return IndexStatusResponse(
        total_documents=len(set(c.doc_id for c in state["chunks"])),
        total_chunks=len(state["chunks"]),
        dimension=dim,
        backend=str(backend),
        is_indexed=state["is_indexed"]
    )


@app.post("/index/build")
def trigger_index_build():
    success = build_system_index()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to build index from sample documents.")
    return {"message": "Index built successfully", "chunks": len(state["chunks"])}


@app.post("/rag/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not state["is_indexed"]:
        build_system_index()

    retriever: HybridRetriever = state["hybrid_retriever"]
    llm: LocalLLMClient = state["llm_client"]

    # Retrieve
    scored_chunks = retriever.retrieve(
        req.query, top_k=req.top_k, strategy=req.strategy
    )
    chunks_only = [c for c, _ in scored_chunks]

    # Generate
    gen_result = llm.generate(
        query=req.query,
        retrieved_chunks=chunks_only,
        prompt_style=req.prompt_style,
        force_fallback=req.force_fallback
    )

    retrieved_schemas = [
        RetrievedChunkSchema(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            score=float(score),
            source_filename=c.source_filename
        )
        for c, score in scored_chunks
    ]

    return QueryResponse(
        query=req.query,
        answer=gen_result["answer"],
        model=gen_result["model"],
        generation_time_ms=gen_result["generation_time_ms"],
        is_fallback=gen_result["is_fallback"],
        retrieved_chunks=retrieved_schemas
    )


@app.post("/evaluate/single")
def evaluate_single(req: SingleEvaluationRequest):
    if not state["is_indexed"]:
        build_system_index()

    retriever: HybridRetriever = state["hybrid_retriever"]
    llm: LocalLLMClient = state["llm_client"]
    decomposer: ErrorDecomposer = state["decomposer"]

    # 1. Retrieve
    scored_chunks = retriever.retrieve(req.question, top_k=req.top_k, strategy=req.strategy)
    chunks = [c for c, _ in scored_chunks]

    # 2. Generate
    gen_out = llm.generate(req.question, chunks, force_fallback=req.force_fallback)

    # 3. Evaluate & Decompose
    meta = {
        "id": "single_test",
        "question": req.question,
        "gold_answer": req.gold_answer,
        "gold_evidence": req.gold_evidence,
        "question_type": req.question_type,
        "difficulty": req.difficulty,
        "domain": "custom"
    }
    sample = decomposer.evaluate_sample(meta, chunks, gen_out)
    return sample.model_dump()


@app.post("/evaluate/benchmark")
def evaluate_benchmark(req: BenchmarkEvaluationRequest):
    if not state["is_indexed"]:
        build_system_index()

    ablation: AblationEngine = state["ablation_engine"]

    # Load benchmark
    if req.custom_benchmark:
        benchmark_data = req.custom_benchmark
    elif req.benchmark_name == "full_50":
        p = BENCHMARKS_DIR / "rag_failure_benchmark_50.json"
        with open(p, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
    else:
        p = BENCHMARKS_DIR / "quick_test_benchmark_10.json"
        with open(p, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    result = ablation.run_benchmark_evaluation(
        benchmark_data=benchmark_data,
        top_k=req.top_k,
        strategy=req.strategy,
        prompt_style=req.prompt_style,
        force_fallback=req.force_fallback
    )
    return result.model_dump()


@app.post("/ablation/sweep_top_k")
def sweep_top_k(req: BenchmarkEvaluationRequest):
    if not state["is_indexed"]:
        build_system_index()

    ablation: AblationEngine = state["ablation_engine"]
    p = BENCHMARKS_DIR / ("quick_test_benchmark_10.json" if req.benchmark_name == "quick_10" else "rag_failure_benchmark_50.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = ablation.sweep_top_k(data, k_values=[1, 2, 3, 5, 8], force_fallback=req.force_fallback)
    return df.to_dict(orient="records")


@app.post("/ablation/sweep_strategies")
def sweep_strategies(req: BenchmarkEvaluationRequest):
    if not state["is_indexed"]:
        build_system_index()

    ablation: AblationEngine = state["ablation_engine"]
    p = BENCHMARKS_DIR / ("quick_test_benchmark_10.json" if req.benchmark_name == "quick_10" else "rag_failure_benchmark_50.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = ablation.sweep_retrieval_strategies(data, top_k=req.top_k, force_fallback=req.force_fallback)
    return df.to_dict(orient="records")


@app.post("/ablation/compare_models")
def compare_models_endpoint(req: ModelComparisonRequest):
    if not state["is_indexed"]:
        build_system_index()

    ablation: AblationEngine = state["ablation_engine"]
    if req.custom_benchmark:
        data = req.custom_benchmark
    elif req.benchmark_name == "full_50":
        p = BENCHMARKS_DIR / "rag_failure_benchmark_50.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        p = BENCHMARKS_DIR / "quick_test_benchmark_10.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

    result = ablation.compare_models(
        benchmark_data=data,
        models=req.models,
        top_k=req.top_k,
        strategy=req.strategy,
        prompt_style=req.prompt_style,
        force_fallback=req.force_fallback
    )

    return {
        "summary": result["summary_df"].to_dict(orient="records"),
        "subtypes": result["subtype_df"].to_dict(orient="records"),
        "discordant_cases": result["discordant_cases"]
    }

