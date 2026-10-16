# 🚀 How to Run & Complete Operational Guide

This document provides a comprehensive, step-by-step operational manual for running and utilizing **every single module, interface, and analysis capability** within the RAG Retrieval-vs-Generation Error Decomposition & Failure Analysis framework.

Each section includes exact CLI commands, user interface instructions, and an **under-the-hood explanation** of what algorithms and data transformations occur behind the scenes.

---

## 📑 Table of Contents

1. [⚡ Quick Start & Environment Setup](#-quick-start--environment-setup)
2. [🦙 Ollama Local Neural LLM Setup (Optional)](#-ollama-local-neural-llm-setup-optional)
3. [💬 Workflow 1: Core RAG Assistant (Daily Chatbot & Live Diagnostics)](#-workflow-1-core-rag-assistant-daily-chatbot--live-diagnostics)
   * [1. Uploading & Ingesting Documents (PDFs, Tables, Text)](#1-uploading--ingesting-documents-pdfs-tables-text)
   * [2. Search Scoping: Focused vs. Global](#2-search-scoping-focused-vs-global)
   * [3. RAG Tuning Profiles & Advanced Sliders](#3-rag-tuning-profiles--advanced-sliders)
   * [4. Live Streaming Generation & Zero-Ground-Truth Diagnostics](#4-live-streaming-generation--zero-ground-truth-diagnostics)
   * [5. Source Inspector & One-Click Failure Studio Transfer](#5-source-inspector--one-click-failure-studio-transfer)
4. [🔬 Workflow 2: Failure Analysis Studio (Research Lab)](#-workflow-2-failure-analysis-studio-research-lab)
   * [Tab 1: 📊 Benchmark & 4-Quadrant Error Decomposition](#tab-1--benchmark--4-quadrant-error-decomposition)
   * [Tab 2: 🌊 Sankey & Waterfall Error Flow Analysis](#tab-2--sankey--waterfall-error-flow-analysis)
   * [Tab 3: 📈 Top-K Parameter Sweep & Context Trade-offs](#tab-3--top-k-parameter-sweep--context-trade-offs)
   * [Tab 4: 🔀 Search Strategy Ablation (Dense vs. Sparse vs. Hybrid)](#tab-4--search-strategy-ablation-dense-vs-sparse-vs-hybrid)
   * [Tab 5: 🤖 Cross-Model Capability Comparison](#tab-5--cross-model-capability-comparison)
   * [Tab 6: 🔍 Live Interactive Sample Inspector](#tab-6--live-interactive-sample-inspector)
5. [🧪 Workflow 3: Running Automated Test Suites](#-workflow-3-running-automated-test-suites)
6. [🌐 Workflow 4: Running the Production FastAPI REST Server](#-workflow-4-running-the-production-fastapi-rest-server)
7. [🐍 Workflow 5: Programmatic Python Usage (`RAGApp`)](#-workflow-5-programmatic-python-usage-ragapp)
8. [🧠 Under the Hood: The 7-Stage Architectural Pipeline](#-under-the-hood-the-7-stage-architectural-pipeline)
9. [❓ Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## ⚡ Quick Start & Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/Talha905/NLP_Project.git
cd NLP_Project
```

### 2. Create and activate a virtual environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays a script policy error, execute: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
* **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Offline Guarantee**: Pre-trained HuggingFace models (`sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/nli-deberta-v3-small`) are cached in `~/.cache/huggingface/`. The pipeline sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, ensuring **100% offline execution** with zero external API calls or network delays.

---

## 🦙 Ollama Local Neural LLM Setup (Optional)

The application functions completely out of the box **with or without Ollama**. If Ollama is offline, the system automatically falls back to an **Intelligent Local Grounded Extractor**.

If you wish to run actual local neural language models on your CPU or GPU:

1. **Install Ollama**: Download from [ollama.com](https://ollama.com).
2. **Pull lightweight local models**:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull llama3.2:latest
   ```
3. **Start Ollama service**:
   ```bash
   ollama serve
   ```
4. The dashboard automatically detects Ollama at `http://localhost:11434` and dynamically switches to the neural generative pipeline.

---

## 💬 Workflow 1: Core RAG Assistant (Daily Chatbot & Live Diagnostics)

To launch the primary web application:
```bash
python run_app.py
```
*(Or alternatively: `streamlit run app/dashboard.py`)*

Open your browser at **`http://localhost:8501`**. Select **💬 Core RAG Assistant** in the top left sidebar navigation.

```
+------------------------------------------------------------------------------------+
| 💬 Core RAG Assistant                                                             |
| [📁 Upload Document]   [RAG Profile: Balanced ▾]   [⚙️ Advanced Settings]         |
| [Search Scope: 🎯 Focused (Selected Doc) | 🌐 Global (All Docs)]                  |
| Active Target Document: [CMPN SEM-7 Div-C SCD-2 Marks.pdf ▾]                      |
|------------------------------------------------------------------------------------|
| Chat Area:                                                                         |
| User: "what is the Roll No of Shashwat"                                            |
| Assistant: "The Roll No of Shashwat Patil is 23102C0048."                         |
| [● Grounded & Supported — Answer is fully grounded in retrieved document context]  |
| [> 📚 Retrieved Source Chunks (Score: 0.9826)]                                     |
+------------------------------------------------------------------------------------+
```

### 1. Uploading & Ingesting Documents (PDFs, Tables, Text)
1. Click **`📁 Upload Custom Document (PDF / TXT)`** in the top control bar.
2. Choose any file from your computer (e.g. an academic marksheet, assignment PDF, textbook chapter, or report).
3. Click **`Index Uploaded Document`**.

#### 🔍 What is happening behind the scenes?
* **PDF Layout Extraction**: The document loader activates `pypdf` with `extraction_mode="layout"`. Instead of collapsing whitespace into an illegible stream, it preserves column alignments and whitespace structure.
* **Token Boundary Separation**: Regular expressions cleanly separate joined indices and roll numbers (e.g. `3223102C0048` $\rightarrow$ `32 23102C0048`).
* **Table Header Detection**: `RecursiveChunker._detect_table_header()` scans the first 50 lines. If tabular markers (`Roll No`, `Sr. No.`, `Marks`, `Attendance`) are found, it captures the column header.
* **Hierarchical Chunking**:
  * **Parent Chunks** (~1200 chars) are constructed. If a parent chunk does not contain the table header, the detected column schema is automatically prepended (`[TABLE COLUMNS REFERENCE: ...]`), ensuring every row retains its meaning.
  * **Child Chunks** (~250 chars) are generated for dense indexing and BM25 search.
* **Dual Indexing**: Child vectors are embedded into the FAISS/NumPy Vector Store, and tokens are indexed into the BM25 Okapi inverted index.
* **Auto-Focus**: The uploaded document is automatically set as the active target document in `st.session_state["selected_doc"]`.

---

### 2. Search Scoping: Focused vs. Global
* **🎯 Focused (Selected Doc)**: Queries retrieve chunks **only** from the selected document in the dropdown.
  * *Why use this?* Perfect for specific documents (e.g. asking for a student's marks from a specific marksheet) to prevent distractor chunks from other files diluting retrieval.
* **🌐 Global (All Docs)**: Queries search across the entire multi-document corpus simultaneously.
  * *Why use this?* Ideal for open-domain questions or comparative analysis across multiple uploaded papers.

#### 🔍 What is happening behind the scenes?
In `VectorStore.search` and `BM25Retriever.search`, when `doc_id` is supplied, chunks from non-matching documents are filtered out before scoring. RRF and Cross-Encoder reranking operate strictly on the targeted document's candidate pool.

---

### 3. RAG Tuning Profiles & Advanced Sliders
Under the top control bar, you can choose pre-configured profiles or customize hyperparameters:
* **⚡ Fast**: Top-K = 2, Dense (semantic vector) search, Grounded direct prompt. Optimized for real-time speed.
* **⚖️ Balanced** *(Default)*: Top-K = 3, Hybrid search (Dense + BM25 + RRF), Grounded prompt. Balanced accuracy and speed.
* **🎯 Deep Research**: Top-K = 5, Hybrid search, Chain-of-Thought (CoT) step-by-step reasoning prompt.
* **⚙️ Advanced Settings (Popover)**:
  * **Context Top-K**: Slider from 1 to 8 chunks.
  * **Search Strategy**: Select `hybrid`, `dense`, or `sparse`.
  * **Prompt Strategy**: Select `grounded` or `cot`.
  * **Force Fast Synthesizer**: Checkbox to bypass Ollama and use local extractive synthesis.

---

### 4. Live Streaming Generation & Zero-Ground-Truth Diagnostics
Type your question in the chat input (e.g., *"What is the Roll No of Shashwat?"* or *"What is the Total ISA of Shashwat?"*).

1. The assistant streams the answer word by word in real time.
2. Immediately below the generated answer, an **Automatic Live Diagnostic Badge** appears.

| Diagnostic Badge | Color | Meaning | Root Cause Detected |
| :--- | :---: | :--- | :--- |
| **● Grounded & Supported** | 🟢 Green | Answer is strictly entailed by the context ($F \ge 0.70$). | Healthy execution; no failure. |
| **● Low Relevance Context** | 🟡 Yellow | Retrieved chunks have low semantic similarity to query ($S_{ret} < 0.60$). | Retrieval failure; danger of distractor information. |
| **● Potential Hallucination** | 🔴 Red | Answer makes claims not supported by the retrieved text ($F < 0.60$). | Generation hallucination; model hallucinated facts. |
| **● Direct Context Contradiction** | 🔴 Red | Model output contradicts explicit facts in the context. | Severe generation failure; prompt constraint violated. |
| **● Appropriate Refusal** | 🟢 Green | Model correctly refused an unanswerable query. | Expected behavior on unanswerable/out-of-domain query. |

#### 🔍 What is happening behind the scenes?
Unlike standard benchmarks that require gold reference answers, our **Live Diagnostic Engine** works on completely novel user queries:
1. It splits the generated answer into individual sentences.
2. It runs sentence-level Natural Language Inference (NLI) using `cross-encoder/nli-deberta-v3-small` against the retrieved chunks.
3. If any sentence generates a high contradiction probability ($P_{contra} > 0.35$) or fails entailment ($P_{entail} < 0.30$), it flags unsupported claims and computes an exact **Faithfulness Score ($F$)**.

---

### 5. Source Inspector & One-Click Failure Studio Transfer
* Click **`📚 Retrieved Source Chunks`** below any assistant message to see the exact parent-child chunks provided to the LLM, complete with source file names and relevance scores.
* If a warning badge appears (e.g. *Low Relevance Context* or *Potential Hallucination*), a button appears:
  **`🧪 Inspect in Failure Analysis Studio ➔`**
  Clicking this button automatically navigates you to the research lab with your exact query pre-loaded into the inspector for deep-dive forensic analysis.

---

## 🔬 Workflow 2: Failure Analysis Studio (Research Lab)

Select **🔬 Failure Analysis Studio** in the left sidebar to access the quantitative evaluation lab.

---

### Tab 1: 📊 Benchmark & 4-Quadrant Error Decomposition

```
                    R = 0 (Retrieval Fail)       R = 1 (Retrieval Pass)
               +-----------------------------+-----------------------------+
               |                             |                             |
  G = 1        |        QUADRANT 3:          |        QUADRANT 1:          |
(Generation    |       HALLUCINATION         |          SUCCESS            |
   Pass)       |   (Correct answer by luck;  |   (Retrieved context good;  |
               |     context irrelevant)     |    answer fully grounded)   |
               |                             |                             |
               +-----------------------------+-----------------------------+
               |                             |                             |
  G = 0        |        QUADRANT 2:          |        QUADRANT 4:          |
(Generation    |      RETRIEVAL FAILURE      |     REASONING FAILURE       |
   Fail)       |    (Wrong context led to    |   (Correct context given;   |
               |       incorrect answer)     |    LLM failed to reason)    |
               |                             |                             |
               +-----------------------------+-----------------------------+
```

#### Steps to Use:
1. In the sidebar, select your benchmark: **`Quick Test (10 Queries)`** or **`Full Research Benchmark (50 Queries)`**.
2. Click **`🚀 Run Complete Benchmark Evaluation`**.
3. View the **Interactive 2D Scatter Plot**:
   * **X-Axis**: Retrieval Quality Score $S_{ret}$ (Threshold $\tau_R = 0.72$).
   * **Y-Axis**: Generation Accuracy Score $S_{ans}$ (Threshold $\tau_A = 0.50$).
   * Each dot represents a query, color-coded by its failure quadrant.
4. Hover over any dot to inspect the question, retrieved chunks, and generated response.
5. Review the summary metrics: System Accuracy, Retrieval Precision, Generation Faithfulness, and Quadrant distributions.

---

### Tab 2: 🌊 Sankey & Waterfall Error Flow Analysis

#### Steps to Use:
1. Navigate to the **`🌊 Sankey & Waterfall Flow`** subtab.
2. The **Sankey Diagram** visualizes how queries traverse the RAG pipeline:
   * `Total Queries` $\rightarrow$ `Retrieval Stage (Pass vs Fail)` $\rightarrow$ `Generation Stage (Success, Hallucination, Reasoning Failure, Compound Failure)`.
3. The **Waterfall Chart** displays net attribution:
   * Baseline (100%) $\rightarrow$ Deductions from Retrieval Dropoffs $\rightarrow$ Deductions from LLM Reasoning Dropoffs $\rightarrow$ Net Pipeline Yield.

---

### Tab 3: 📈 Top-K Parameter Sweep & Context Trade-offs

#### Steps to Use:
1. Navigate to **`📈 Top-K Parameter Sweep`**.
2. Select your test dataset and click **`Run Top-K Sweep (K ∈ [1, 2, 3, 5, 8])`**.
3. View the dual-axis trade-off graph:
   * **Blue Line**: Retrieval Recall (increases as $K$ increases).
   * **Red Line**: Generation Faithfulness (decreases at high $K$ due to **context dilution / distractor noise**).
4. Discover the mathematical sweet spot ($K^* = 3$), which maximizes context relevance while minimizing distractor-induced hallucination.

---

### Tab 4: 🔀 Search Strategy Ablation (Dense vs. Sparse vs. Hybrid)

#### Steps to Use:
1. Navigate to **`🔀 Strategy Ablation`**.
2. Click **`Compare Retrieval Strategies`**.
3. The engine runs the benchmark across:
   * **Dense Only**: Bi-encoder cosine similarity (`all-MiniLM-L6-v2`).
   * **Sparse Only**: BM25 Okapi lexical term matching.
   * **Hybrid (RRF)**: Reciprocal Rank Fusion combination ($k_{rrf} = 60$).
4. The resulting bar chart and table show how Hybrid retrieval prevents out-of-vocabulary (OOV) entity dropouts while maintaining semantic understanding.

---

### Tab 5: 🤖 Cross-Model Capability Comparison

#### Steps to Use:
1. Navigate to **`🤖 Model Capability Analysis`**.
2. Choose two models available in Ollama (e.g. `qwen2.5:3b` vs `llama3.2:latest`, or local synthesizer).
3. Click **`Run Cross-Model Comparison`**.
4. Both models are evaluated on the **exact same retrieved context chunks**.
5. The view displays **Discordant Cases**: queries where Retrieval succeeded ($R=1$), but Model A succeeded while Model B failed, isolating **model parameter capacity** from retrieval errors.

---

### Tab 6: 🔍 Live Interactive Sample Inspector

#### Steps to Use:
1. Navigate to **`🔍 Sample Inspector`**.
2. Select any query from the dropdown or type a custom question with reference context.
3. Click **`Inspect Query Decomposition`**.
4. The system presents:
   * Extracted entities and keyword matches.
   * Cosine similarity and BM25 rank scores.
   * Sentence-by-sentence NLI entailment probabilities.
   * Formal Failure Taxonomy classification (e.g. *Type 1: Out-of-Vocabulary Lexical Mismatch*, *Type 3: Distractor Context Dilution*, *Type 5: Negation Inversion*).

---

## 🧪 Workflow 3: Running Automated Test Suites

To verify pipeline integrity, run the complete unit and integration test suite:

```bash
python run_app.py --test
```
*(Or alternatively: `python -m unittest discover -s tests -p "test_*.py"`)*

### Test Coverage Breakdown:
* `test_ingestion.py`: Verifies PDF layout extraction, regex token boundaries, table header preservation, and parent-child hierarchical chunk linkage.
* `test_retrieval.py`: Tests Dense Bi-Encoder embeddings, BM25 indexing, FAISS cosine search, document-scoping filters, and RRF fusion.
* `test_reranker.py`: Validates Cross-Encoder second-stage reranking and candidate scoring.
* `test_nli_grounding.py`: Checks sentence-level DeBERTa NLI faithfulness scoring and contradiction detection.
* `test_decomposition.py`: Validates 4-Quadrant classification logic and threshold adherence ($\tau_R = 0.72, \tau_A = 0.50$).
* `test_api.py`: Validates FastAPI REST endpoints, JSON schemas, and response status codes.

---

## 🌐 Workflow 4: Running the Production FastAPI REST Server

To serve the RAG pipeline as a headless microservice:

```bash
python run_app.py --api
```
*(Or alternatively: `uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload`)*

Interactive OpenAPI Swagger documentation will be available at: **`http://127.0.0.1:8000/docs`**.

### Key REST Endpoints & Example Requests:

#### 1. Ask a Question (`POST /rag/query`)
```bash
curl -X POST "http://127.0.0.1:8000/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the Roll No of Shashwat?",
       "top_k": 3,
       "strategy": "hybrid",
       "prompt_style": "grounded"
     }'
```

#### 2. Evaluate a Single Query (`POST /evaluate/single`)
```bash
curl -X POST "http://127.0.0.1:8000/evaluate/single" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is self-attention?",
       "gold_answer": "A mechanism relating different positions of a single sequence.",
       "gold_evidence": "Self-attention connects different positions of a sequence.",
       "top_k": 3,
       "strategy": "hybrid"
     }'
```

#### 3. Run Benchmark Suite (`POST /evaluate/benchmark`)
```bash
curl -X POST "http://127.0.0.1:8000/evaluate/benchmark" \
     -H "Content-Type: application/json" \
     -d '{
       "benchmark_name": "quick_10",
       "top_k": 3,
       "strategy": "hybrid"
     }'
```

---

## 🐍 Workflow 5: Programmatic Python Usage (`RAGApp`)

You can instantiate and interact with the RAG pipeline directly in Python scripts or Jupyter notebooks:

```python
from src.core.rag_engine import RAGApp

# 1. Initialize RAG application (indexes data/sample_docs automatically)
app = RAGApp()

# 2. Ingest an uploaded file on the fly
app.index_uploaded_file("data/sample_docs/CMPN SEM-7 Div-C SCD-2 Marks.pdf")

# 3. Query with document scoping and live diagnostics
response = app.ask(
    query="what is the Roll No of Shashwat",
    doc_id="CMPN SEM-7 Div-C SCD-2 Marks.pdf",
    top_k=3,
    strategy="hybrid"
)

# 4. Inspect results
print("Answer:", response.answer)
print("Diagnostic Status:", response.diagnostic.status)
print("Diagnostic Badge:", response.diagnostic.badge_text)
print("Faithfulness Score:", response.diagnostic.faithfulness_score)

# 5. Streaming response generator
token_stream, sources, chunks = app.stream_ask(
    query="what is the Total ISA of shashwat",
    doc_id="CMPN SEM-7 Div-C SCD-2 Marks.pdf"
)
for token in token_stream:
    print(token, end="", flush=True)
print()
```

---

## 🧠 Under the Hood: The 7-Stage Architectural Pipeline

```
[Raw Document] 
      │
      ▼
1. INGESTION ──────────► Layout Preservation (pypdf layout mode) + Token Boundaries
      │
      ▼
2. CHUNKING ───────────► Hierarchical Parent (~1200c) & Child (~250c) + Table Header Retention
      │
      ▼
3. RETRIEVAL ──────────► Hybrid Search: Dense Vectors (all-MiniLM) + Sparse Lexical (BM25)
      │
      ▼
4. FUSION & RERANK ────► Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranker (ms-marco)
      │
      ▼
5. CONTEXT EXPANSION ──► Retrieved child chunks expand to full parent section context
      │
      ▼
6. GENERATION ─────────► Grounded Prompt + Local Neural LLM (Ollama) / Local Synthesizer
      │
      ▼
7. DIAGNOSTICS ────────► Sentence-Level Cross-Encoder NLI (DeBERTa) Faithfulness Verification
```

### Detailed Stage Breakdown:

1. **Ingestion & Layout Extraction (`document_loader.py`)**:
   Reads text and PDF files. For PDFs, extracts characters with coordinate-aware physical layout mode (`extraction_mode="layout"`). Splits glued alphanumeric tokens via regex (`re.sub(r'\b(\d{1,2})(23102[A-Za-z0-9]+)\b', r'\1 \2', text)`).
2. **Hierarchical Parent-Child Chunking (`chunker.py`)**:
   Detects table headers (`Sr. No.`, `Roll No.`, `Student Name`, `Marks`). Prepares large parent context chunks (~1200 chars) injected with the table header reference, and small child chunks (~250 chars) for granular vector search.
3. **Dual Embedding & Inverted Indexing (`embedder.py`, `bm25_retriever.py`, `vector_store.py`)**:
   Embeds child chunks into 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2` with PyTorch mean-pooling. Builds an inverted token frequency index with BM25 Okapi parameters ($k_1 = 1.5, b = 0.75$).
4. **Hybrid Search & 2nd-Stage Reranking (`hybrid_retriever.py`, `reranker.py`)**:
   Retrieves candidates from both Dense and Sparse indices, fusing ranks using Reciprocal Rank Fusion:
   $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k_{\text{rrf}} + r_m(d)}$$
   Top candidates are then passed through `cross-encoder/ms-marco-MiniLM-L-6-v2` to compute deep query-passage cross-attention scores.
5. **Context Expansion (`prompt_templates.py`)**:
   When child chunks win retrieval, `PromptBuilder.format_context_chunks()` replaces the tiny child chunk with its full expanded parent section, eliminating truncated sentence boundaries.
6. **Constrained Generation (`llm_client.py`)**:
   Constructs a strictly grounded prompt with instructions to locate tabular entity rows and extract fields. Queries local Ollama instance (`qwen2.5:3b` or `llama3.2`) with low temperature ($T = 0.1$) for deterministic output.
7. **Zero-Ground-Truth NLI Verification (`nli_grounding.py`, `rag_engine.py`)**:
   Splits the generated answer into individual propositions. Computes pairwise premise-hypothesis entailment and contradiction scores against the retrieved context using `cross-encoder/nli-deberta-v3-small`. Flags any unevidenced claims and assigns real-time diagnostic badges.

---

## ❓ Troubleshooting & FAQs

### 1. "Ollama: Offline (Fallback Mode Active)"
* **Normal Behavior**: If you haven't installed or started Ollama, this is expected. The app runs fully in fallback mode using its local extractive synthesizer.
* **To Fix**: If Ollama is installed, run `ollama serve` in a terminal and confirm `http://localhost:11434` loads in your browser, then refresh Streamlit.

### 2. Tabular marksheets returning "The provided context does not contain sufficient evidence..."
* Make sure you select **🎯 Focused (Selected Doc)** and choose the uploaded marksheet in the dropdown.
* Ensure you ask for specific student names or fields present in the file (e.g. *"What is the Roll No of Shashwat"* or *"What is the Total ISA of Shashwat"*).

### 3. Port 8501 is already in use
* Launch Streamlit on an alternate port:
  ```bash
  python -m streamlit run app/dashboard.py --server.port 8502
  ```

### 4. PowerShell script execution disabled error
* Run this command in PowerShell before activating your virtual environment:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```
