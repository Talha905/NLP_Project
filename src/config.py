"""
Global configuration for RAG Retrieval-vs-Generation Error Decomposition & Failure Analysis.
Configured for free, local execution without paid APIs or dedicated GPU requirements.
"""

import os

# Strict Offline Enforcement: Prevent any online network calls/pings to Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
INDEX_DIR = BASE_DIR / "storage" / "indices"

# Create directories if needed
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval Defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_ALPHA = 0.5  # Weight for dense (1-alpha for BM25)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_RERANKER = True
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ALTERNATIVE_OLLAMA_MODEL = "llama3.2:latest"

# Empirically Calibrated Evaluation Thresholds
# Grounded via 50-sample validation sweep (see THRESHOLD_DERIVATION.md and scripts/derive_thresholds.py)
RETRIEVAL_SIMILARITY_THRESHOLD = 0.72  # tau_R: Cosine similarity threshold for gold evidence presence
ANSWER_SIMILARITY_THRESHOLD = 0.78     # tau_S: Semantic cosine similarity threshold for answer correctness
TOKEN_F1_THRESHOLD = 0.50              # tau_A: Token-level F1 threshold for lexical alignment
FAITHFULNESS_THRESHOLD = 0.60          # tau_F: Sentence NLI entailment threshold

# Failure Taxonomy Categories
FAILURE_CATEGORIES = {
    # High-level outcome quadrants
    "SUCCESS": "End-to-End Success (R=1, G=1)",
    "GENERATION_FAILURE": "Generation Failure (R=1, G=0)",
    "RETRIEVAL_FAILURE": "Retrieval Failure (R=0, G=0)",
    "PARAMETRIC_SUCCESS": "Parametric / Spurious Recall (R=0, G=1)",
    
    # Granular Retrieval Subtypes
    "MISSING_EVIDENCE": "Retrieval: Missing Gold Evidence in Top-K",
    "NOISE_DILUTION": "Retrieval: Evidence Diluted by Distractor Chunks",
    "TRUNCATION_BOUNDARY": "Retrieval: Critical Content Cutoff Across Chunk Boundary",
    
    # Granular Generation Subtypes
    "HALLUCINATION_UNSUPPORTED": "Generation: Unsupported Claims / Hallucination",
    "CONTEXT_CONTRADICTION": "Generation: Direct Contradiction of Context",
    "REASONING_SYNTHESIS_ERROR": "Generation: Synthesis or Deductive Reasoning Error",
    "INCOMPLETE_ANSWER": "Generation: Incomplete / Partial Coverage",
    "CONSERVATIVE_REFUSAL": "Generation: Over-conservative Refusal (Evidence was present)",
    "UNANSWERABLE_CORRECT_REFUSAL": "Generation: Appropriate Refusal for Unanswerable Query"
}
