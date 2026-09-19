"""
Core RAG Engine: Clean, decoupled, production-ready RAG application
with automatic real-time self-diagnostics for live queries.
"""

from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import time
from pydantic import BaseModel, Field

from ..config import SAMPLE_DOCS_DIR, DEFAULT_TOP_K, DEFAULT_EMBEDDING_MODEL
from ..ingestion.document_loader import DocumentLoader, Document
from ..ingestion.chunker import RecursiveChunker, TextChunk
from ..retrieval.embedder import DenseEmbedder
from ..retrieval.vector_store import VectorStore
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generation.llm_client import LocalLLMClient
from ..evaluation.nli_grounding import GroundingEvaluator
from ..evaluation.failure_classifier import FailureTaxonomyAdvisor


class LiveDiagnostic(BaseModel):
    """Real-time health and failure diagnosis for any live query (no gold label required)."""
    status: str             # HEALTHY, RETRIEVAL_WARNING, HALLUCINATION_WARNING, REFUSAL_WARNING
    badge_text: str         # e.g. "Grounded & Supported", "Low Context Relevance", "Potential Hallucination"
    color: str              # Hex color code (#10B981, #F59E0B, #EF4444)
    explanation: str        # Plain-English diagnosis
    mitigation: str         # Actionable advice
    faithfulness_score: float = 1.0
    max_retrieval_sim: float = 0.0
    unsupported_claims: List[str] = []


class RAGResponse(BaseModel):
    """Encapsulates a generated RAG answer, source chunks, and diagnostic check."""
    query: str
    answer: str
    model: str
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: int
    diagnostic: LiveDiagnostic


class RAGApp:
    """
    Unified, clean RAG Application.
    Handles document ingestion, semantic/hybrid search, streaming LLM generation,
    and automatic live self-diagnostics.
    """

    def __init__(
        self,
        doc_dir: Optional[Path | str] = None,
        model_name: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        strategy: str = "hybrid"
    ):
        self.doc_dir = Path(doc_dir) if doc_dir else SAMPLE_DOCS_DIR
        self.top_k = top_k
        self.strategy = strategy

        # Core Components
        self.embedder = DenseEmbedder()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.bm25 = BM25Retriever()
        self.llm = LocalLLMClient(model_name=model_name or "qwen2.5:3b")
        self.grounding_eval = GroundingEvaluator(self.embedder)
        self.retriever = HybridRetriever(self.vector_store, self.bm25, self.embedder)

        self.documents: List[Document] = []
        self.chunks: List[TextChunk] = []
        self.is_indexed = False

        # Build initial index from doc_dir
        self.index_directory(self.doc_dir)

    def index_directory(self, dir_path: Path | str, chunk_size: int = 400, chunk_overlap: int = 50):
        """Ingests and indexes all documents from a directory."""
        path = Path(dir_path)
        if not path.exists():
            return

        docs = DocumentLoader.load_directory(path)
        if not docs:
            return

        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(docs)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.vector_store.add_chunks(chunks, embeddings)

        self.bm25 = BM25Retriever()
        self.bm25.index_chunks(chunks)

        self.retriever = HybridRetriever(self.vector_store, self.bm25, self.embedder)
        self.documents = docs
        self.chunks = chunks
        self.is_indexed = True

    def index_uploaded_file(self, file_path: Path | str):
        """Indexes a newly uploaded single file."""
        doc = DocumentLoader.load_file(file_path)
        if not doc:
            return False

        chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
        new_chunks = chunker.chunk_documents([doc])
        if not new_chunks:
            return False

        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.add_chunks(new_chunks, embeddings)
        self.chunks.extend(new_chunks)
        self.bm25.index_chunks(self.chunks)
        self.documents.append(doc)
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None, strategy: Optional[str] = None):
        """Retrieves top_k context chunks with similarity scores."""
        k = top_k or self.top_k
        strat = strategy or self.strategy
        return self.retriever.retrieve(query, top_k=k, strategy=strat)

    def diagnose(
        self,
        query: str,
        retrieved_chunks: List[TextChunk],
        answer: str
    ) -> LiveDiagnostic:
        """
        Automatic live self-diagnostic on any query without requiring gold labels.
        Evaluates retrieval relevance, grounding faithfulness, and refusal patterns.
        """
        if not retrieved_chunks:
            return LiveDiagnostic(
                status="RETRIEVAL_WARNING",
                badge_text="No Relevant Context Found",
                color="#DC2626",
                explanation="The system retrieved zero text chunks matching your question.",
                mitigation="Ensure documents are uploaded and contain content related to this topic."
            )

        # 1. Retrieval confidence check (max cosine similarity between query and retrieved chunks)
        q_vec = self.embedder.embed_query(query)
        max_sim = 0.0
        for chunk in retrieved_chunks:
            c_vec = self.embedder.embed_query(chunk.text)
            dot = float(q_vec.dot(c_vec) / (np_norm(q_vec) * np_norm(c_vec) + 1e-9))
            if dot > max_sim:
                max_sim = dot

        # Check for model refusal
        refusal_keywords = [
            "does not contain sufficient evidence", "insufficient evidence",
            "not mentioned in the context", "context lacks", "cannot be answered"
        ]
        is_refusal = any(kw in answer.lower() for kw in refusal_keywords)

        # 2. Grounding / Hallucination check
        grounding = self.grounding_eval.evaluate_groundedness(answer, retrieved_chunks)
        faithfulness = grounding["faithfulness_score"]
        unsupported = grounding.get("unsupported_claims", [])

        # Diagnostics tree:
        if is_refusal:
            if max_sim >= 0.55:
                return LiveDiagnostic(
                    status="REFUSAL_WARNING",
                    badge_text="Over-Conservative Refusal",
                    color="#F59E0B",
                    explanation="The retrieved documents contain relevant evidence, but the model conservatively refused to answer.",
                    mitigation="Soften prompt instructions or rephrase your question more directly.",
                    faithfulness_score=1.0,
                    max_retrieval_sim=max_sim
                )
            else:
                return LiveDiagnostic(
                    status="HEALTHY",
                    badge_text="Appropriate Refusal (Information Missing)",
                    color="#10B981",
                    explanation="The model correctly recognized that the documents lack information to answer this question.",
                    mitigation="No mitigation needed; the model prevented a hallucination.",
                    faithfulness_score=1.0,
                    max_retrieval_sim=max_sim
                )

        if max_sim < 0.38:
            return LiveDiagnostic(
                status="RETRIEVAL_WARNING",
                badge_text="Low Relevance Context",
                color="#F59E0B",
                explanation="The retrieved text chunks have low semantic similarity to your query. The answer may be based on distractor information.",
                mitigation="Try using Hybrid Search or verify if the topic exists in the knowledge base.",
                faithfulness_score=faithfulness,
                max_retrieval_sim=max_sim,
                unsupported_claims=unsupported
            )

        if not grounding["is_faithful"] or faithfulness < 0.60:
            return LiveDiagnostic(
                status="HALLUCINATION_WARNING",
                badge_text="Potential Hallucination Detected",
                color="#DC2626",
                explanation=f"{len(unsupported)} claim(s) in the generated answer were not supported by the retrieved document chunks.",
                mitigation="Lower model temperature to 0.0 or prompt with 'Answer strictly using only cited context'.",
                faithfulness_score=faithfulness,
                max_retrieval_sim=max_sim,
                unsupported_claims=unsupported
            )

        # Passed all checks
        return LiveDiagnostic(
            status="HEALTHY",
            badge_text="Grounded & Supported",
            color="#10B981",
            explanation="The answer is fully grounded in the retrieved document context with high factual consistency.",
            mitigation="No failure detected. Answer satisfies grounding and relevance thresholds.",
            faithfulness_score=faithfulness,
            max_retrieval_sim=max_sim
        )

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        prompt_style: str = "grounded",
        model_name: Optional[str] = None
    ) -> RAGResponse:
        """Executes full RAG query and returns answer, sources, and live self-diagnosis."""
        start_t = time.time()
        scored_chunks = self.retrieve(query, top_k=top_k, strategy=strategy)
        chunks = [c for c, _ in scored_chunks]

        gen_out = self.llm.generate(
            query=query,
            retrieved_chunks=chunks,
            prompt_style=prompt_style,
            model_name=model_name
        )

        latency = int((time.time() - start_t) * 1000)
        answer = gen_out["answer"]
        diagnostic = self.diagnose(query, chunks, answer)

        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "score": float(score),
                "source": c.source_filename or c.doc_id
            }
            for c, score in scored_chunks
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            model=gen_out["model"],
            retrieved_chunks=chunk_dicts,
            latency_ms=latency,
            diagnostic=diagnostic
        )

    def stream_ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        prompt_style: str = "grounded",
        model_name: Optional[str] = None
    ) -> tuple[Generator[str, None, None], List[Dict[str, Any]], List[TextChunk]]:
        """Streaming version returning (token_generator, chunk_dicts, chunk_objects)."""
        scored_chunks = self.retrieve(query, top_k=top_k, strategy=strategy)
        chunks = [c for c, _ in scored_chunks]

        token_gen = self.llm.generate_stream(
            query=query,
            retrieved_chunks=chunks,
            prompt_style=prompt_style,
            model_name=model_name
        )

        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "score": float(score),
                "source": c.source_filename or c.doc_id
            }
            for c, score in scored_chunks
        ]
        return token_gen, chunk_dicts, chunks


def np_norm(v):
    import numpy as np
    n = np.linalg.norm(v)
    return max(1e-9, float(n))
