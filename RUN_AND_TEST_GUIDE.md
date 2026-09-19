# Complete Guide: Running the Project and Testing RAG Failures

This guide provides step-by-step instructions on **how to run the project** and **how to systematically test and reproduce every RAG failure type** (both retrieval and generation failures) using the web dashboard, the REST API, and standalone Python scripts.

---

## 1. Prerequisites and Installation

### System Requirements
* **Operating System:** Windows, macOS, or Linux.
* **Python Version:** Python 3.10 or 3.11.
* **Hardware:** Standard laptop CPU (No dedicated GPU required).
* **Cost:** 100% free and open-source (no paid APIs or subscriptions).

### Installation Steps
Open PowerShell or your terminal in the project directory (`NLP_Project`):

```powershell
# 1. (Optional) Activate your Python virtual environment if you use one
# python -m venv venv
# .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

### Local LLM Setup (Optional)
The project works immediately out of the box with zero external setup using its built-in fast local grounded synthesizer. 
If you want to run actual local neural generation:
1. Download and run [Ollama](https://ollama.com).
2. Pull a lightweight model:
   ```bash
   ollama run qwen2.5:3b
   # or
   ollama run llama3.2:latest
   ```
The application will automatically detect Ollama running at `http://localhost:11434`.

---

## 2. How to Run the Project

The project provides three runtime interfaces managed via `run_app.py`:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Interactive Dashboard** | `python run_app.py` | Launches the Streamlit UI at `http://localhost:8501` |
| **FastAPI REST Server** | `python run_app.py --api` | Launches the backend at `http://127.0.0.1:8000` (Docs at `/docs`) |
| **Automated Test Suite** | `python run_app.py --test` | Runs all 11 unit and integration tests |

---

## 3. How to Test Each Specific RAG Failure Mode

The framework categorizes errors into a **4-Quadrant Outcome Matrix** and fine-grained subtypes:
- **Retrieval Failures ($R=0, G=0$):** `MISSING_EVIDENCE`, `NOISE_DILUTION`, `TRUNCATION_BOUNDARY`
- **Generation Failures ($R=1, G=0$):** `CONSERVATIVE_REFUSAL`, `HALLUCINATION_UNSUPPORTED`, `CONTEXT_CONTRADICTION`, `REASONING_SYNTHESIS_ERROR`, `INCOMPLETE_ANSWER`
- **Parametric Recall ($R=0, G=1$):** `PARAMETRIC_RECALL`
- **Insufficient Evidence ($R=0, G=1$ for unanswerable):** `UNANSWERABLE_CORRECT_REFUSAL`

Below are recipes to trigger and verify each failure mode.

---

### Test 1: Testing Retrieval Failure (`MISSING_EVIDENCE`)

* **What it tests:** Top-$K$ chunks contain zero semantic or lexical overlap with required ground-truth evidence (`max_similarity < 0.35`).
* **Method 1 (via Streamlit Dashboard):**
  1. Open **Tab 1: Live RAG & Inspector**.
  2. Set **Top-K Context Chunks** to `1`.
  3. Set **Retrieval Method** to `Dense`.
  4. Ask a question outside the retrieved chunk's focus or an out-of-domain question:
     > *"What is the exact battery capacity and warp drive core temperature of Apollo 11?"*
  5. Click **Run RAG Query & Diagnostic**.
  6. **Observation:** The similarity score of Chunk 1 is very low (< 0.35). If evaluated, the system tags this as `MISSING_EVIDENCE` under `RETRIEVAL_FAILURE`.
* **Method 2 (via Standalone Script):**
  ```bash
  python -c "
  from src.evaluation.error_decomposer import ErrorDecomposer
  from src.retrieval.embedder import DenseEmbedder
  from src.ingestion.chunker import TextChunk

  e = DenseEmbedder()
  d = ErrorDecomposer(e)
  meta = {'id': 't1', 'question': 'What is quantum entanglement?', 'gold_answer': 'Entanglement is non-local correlation', 'gold_evidence': 'Quantum entanglement is a non-local correlation between two or more qubits', 'question_type': 'factoid', 'difficulty': 'easy'}
  unrelated_chunk = [TextChunk(chunk_id='c1', doc_id='d1', text='Climate change causes sea level rise from melting glaciers.', start_char=0, end_char=50, token_count=10)]
  sample = d.evaluate_sample(meta, unrelated_chunk, {'answer': 'The context lacks info.', 'generation_time_ms': 5, 'model': 'test'})
  print('Quadrant:', sample.quadrant)
  print('Subtype:', sample.failure_subtype)
  print('Explanation:', sample.diagnostic_explanation)
  "
  ```
  **Expected Output:**
  ```
  Quadrant: RETRIEVAL_FAILURE
  Subtype: MISSING_EVIDENCE
  Explanation: Top-K chunks contained no semantic overlap with the required ground-truth evidence.
  ```

---

### Test 2: Testing Retrieval Failure (`NOISE_DILUTION`)

* **What it tests:** Relevant evidence is buried under irrelevant distractor chunks (`precision_at_k < 0.30`).
* **Method (via Streamlit Dashboard):**
  1. Open **Tab 4: Ablation & Sensitivity**.
  2. Click **Run Top-K Sweep (K = 1, 2, 3, 5, 8)**.
  3. Inspect the **Sensitivity Chart**:
     - At $K=1$, retrieval failure is higher due to missing evidence.
     - At $K=8$, note how `Mean_Retrieval_Precision` drops from ~0.80 down to ~0.25 as distractor noise chunks are pulled into context.
  4. In **Tab 5: Sample Deep Dive**, filter by subtype `NOISE_DILUTION` to inspect queries where evidence was diluted.

---

### Test 3: Testing Generation Failure (`CONSERVATIVE_REFUSAL`)

* **What it tests:** Evidence **is present** in retrieved context ($R=1$), but the LLM states "I don't know" or "The context does not contain sufficient evidence".
* **Method (via Standalone Script):**
  ```bash
  python -c "
  from src.evaluation.error_decomposer import ErrorDecomposer
  from src.retrieval.embedder import DenseEmbedder
  from src.ingestion.chunker import TextChunk

  e = DenseEmbedder()
  d = ErrorDecomposer(e)
  meta = {'id': 't2', 'question': 'What is the attention formula?', 'gold_answer': 'softmax(QK^T/sqrt(d_k))V', 'gold_evidence': 'Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V', 'question_type': 'factoid', 'difficulty': 'easy'}
  good_chunk = [TextChunk(chunk_id='c1', doc_id='d1', text='Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V is the formula.', start_char=0, end_char=50, token_count=12)]
  # LLM received evidence but conservatively refused
  sample = d.evaluate_sample(meta, good_chunk, {'answer': 'The provided context does not contain sufficient evidence to answer this question.', 'generation_time_ms': 5, 'model': 'test'})
  print('Quadrant:', sample.quadrant)
  print('Subtype:', sample.failure_subtype)
  print('Explanation:', sample.diagnostic_explanation)
  "
  ```
  **Expected Output:**
  ```
  Quadrant: GENERATION_FAILURE
  Subtype: CONSERVATIVE_REFUSAL
  Explanation: Evidence was present in the retrieved context, but the LLM exhibited over-conservative refusal ('I don't know').
  ```

---

### Test 4: Testing Generation Failure (`HALLUCINATION_UNSUPPORTED`)

* **What it tests:** Evidence is retrieved, but the generator makes claims not supported by the context (`faithfulness_score < 0.60`).
* **Method (via Standalone Script):**
  ```bash
  python -c "
  from src.evaluation.error_decomposer import ErrorDecomposer
  from src.retrieval.embedder import DenseEmbedder
  from src.ingestion.chunker import TextChunk

  e = DenseEmbedder()
  d = ErrorDecomposer(e)
  meta = {'id': 't3', 'question': 'What are transmon qubits?', 'gold_answer': 'Superconducting qubits using Josephson junctions', 'gold_evidence': 'Employ transmon qubits constructed from Josephson junctions operating below 15 mK', 'question_type': 'factoid', 'difficulty': 'easy'}
  chunk = [TextChunk(chunk_id='c1', doc_id='d1', text='Employ transmon qubits constructed from Josephson junctions operating below 15 mK.', start_char=0, end_char=60, token_count=12)]
  # Hallucinated response
  sample = d.evaluate_sample(meta, chunk, {'answer': 'Transmon qubits are alien crystal hyperdrive cores operating on antimatter plasma.', 'generation_time_ms': 10, 'model': 'test'})
  print('Quadrant:', sample.quadrant)
  print('Subtype:', sample.failure_subtype)
  print('Faithfulness Score:', sample.faithfulness_score)
  print('Explanation:', sample.diagnostic_explanation)
  "
  ```
  **Expected Output:**
  ```
  Quadrant: GENERATION_FAILURE
  Subtype: HALLUCINATION_UNSUPPORTED
  Faithfulness Score: 0.0
  Explanation: Evidence was present, but the LLM hallucinated ungrounded claims not supported by the context.
  ```

---

### Test 5: Testing Generation Failure (`CONTEXT_CONTRADICTION`)

* **What it tests:** The LLM output directly negates facts stated in the retrieved context.
* **Method (via Standalone Script):**
  ```bash
  python -c "
  from src.evaluation.error_decomposer import ErrorDecomposer
  from src.retrieval.embedder import DenseEmbedder
  from src.ingestion.chunker import TextChunk

  e = DenseEmbedder()
  d = ErrorDecomposer(e)
  meta = {'id': 't4', 'question': 'Does CO2 contribute to radiative forcing?', 'gold_answer': 'Yes, CO2 contributes approximately 66%', 'gold_evidence': 'Carbon dioxide contributes approximately 66% of total radiative forcing', 'question_type': 'factoid', 'difficulty': 'easy'}
  chunk = [TextChunk(chunk_id='c1', doc_id='d1', text='Carbon dioxide contributes approximately 66% of total radiative forcing from long-lived greenhouse gases.', start_char=0, end_char=80, token_count=15)]
  # Direct negation contradiction
  sample = d.evaluate_sample(meta, chunk, {'answer': 'Carbon dioxide does not contribute to radiative forcing and never causes warming.', 'generation_time_ms': 10, 'model': 'test'})
  print('Quadrant:', sample.quadrant)
  print('Subtype:', sample.failure_subtype)
  print('Contradiction Flag:', sample.contradiction_detected)
  print('Explanation:', sample.diagnostic_explanation)
  "
  ```
  **Expected Output:**
  ```
  Quadrant: GENERATION_FAILURE
  Subtype: CONTEXT_CONTRADICTION
  Contradiction Flag: True
  Explanation: Evidence was present, but the LLM generated statements contradicting facts in the context.
  ```

---

### Test 6: Testing Insufficient Evidence (`UNANSWERABLE_CORRECT_REFUSAL`)

* **What it tests:** The question asks for information not present in the knowledge base, and the model correctly refuses to answer.
* **Method (via Streamlit Dashboard):**
  1. Open **Tab 2: Benchmark Evaluation**.
  2. Select **Quick Test (10 Questions)** and click **Start Benchmark Evaluation**.
  3. Open **Tab 5: Sample Deep Dive**.
  4. Select Filter by Quadrant: **`SUCCESS`**.
  5. Locate Question 10:
     > *"What was the exact battery capacity and warp drive core temperature of the Apollo 11 lunar module?"*
  6. **Observation:**
     - Ground-truth evidence is labeled `UNANSWERABLE_ADVERSARIAL_QUERY`.
     - Retrieval is $R=0$ (evidence not found).
     - Model output: *"The provided context does not contain sufficient evidence..."*
     - Final Quadrant: **`SUCCESS`** with Subtype **`UNANSWERABLE_CORRECT_REFUSAL`**.

---

### Test 7: Testing Spurious Parametric Recall (`PARAMETRIC_RECALL`)

* **What it tests:** Evidence was missing from Top-$K$ ($R=0$), but the LLM recalled the fact from pretraining memory ($G=1$).
* **Method (via Standalone Script):**
  ```bash
  python -c "
  from src.evaluation.error_decomposer import ErrorDecomposer
  from src.retrieval.embedder import DenseEmbedder
  from src.ingestion.chunker import TextChunk

  e = DenseEmbedder()
  d = ErrorDecomposer(e)
  meta = {'id': 't6', 'question': 'Who wrote Attention Is All You Need?', 'gold_answer': 'Vaswani et al.', 'gold_evidence': 'Vaswani et al. in 2017 in Attention Is All You Need', 'question_type': 'factoid', 'difficulty': 'easy'}
  # Retriever failed (returned unrelated climate chunk)
  bad_chunk = [TextChunk(chunk_id='c1', doc_id='d1', text='Oceans absorb over 90% of excess heat in the climate system.', start_char=0, end_char=50, token_count=12)]
  # LLM correctly guessed from pretraining weights
  sample = d.evaluate_sample(meta, bad_chunk, {'answer': 'Vaswani et al. in 2017.', 'generation_time_ms': 10, 'model': 'test'})
  print('Quadrant:', sample.quadrant)
  print('Retrieval Success:', sample.retrieval_success)
  print('Generation Success:', sample.generation_success)
  print('Explanation:', sample.diagnostic_explanation)
  "
  ```
  **Expected Output:**
  ```
  Quadrant: PARAMETRIC_SUCCESS
  Retrieval Success: False
  Generation Success: True
  Explanation: Evidence was missing from retrieved context, but the LLM answered correctly using pre-trained parametric memory.
  ```

---

## 4. How to Run Batch Benchmark Sweeps & Ablations

### Running Batch Evaluation in the UI
1. Launch dashboard: `python run_app.py`.
2. Go to **Tab 2: Benchmark Evaluation**.
3. Choose:
   - `Quick Test (10 Questions)` for a fast 5-second sanity check.
   - `Full Research Benchmark (50 Questions)` for comprehensive evaluation.
4. Set **Top-K** (e.g. 3) and **Strategy** (`hybrid`).
5. Click **⚡ Start Benchmark Evaluation**.
6. View the KPI cards:
   - **Total Error Rate ($E_{\text{total}}$):** Overall proportion of erroneous responses.
   - **Retrieval Attribution ($\alpha_R$):** Proportion of errors originating in the retriever.
   - **Generation Attribution ($\alpha_G$):** Proportion of errors originating in the generator.

### Visualizing the Error Breakdown
* Navigate to **Tab 3: Error Decomposition**:
  - **Sankey Diagram:** Follow the flow of queries from All Queries $\to$ Retrieval Outcome $\to$ Generation Outcome $\to$ Subtypes.
  - **Waterfall Chart:** Decomposes Total Error Rate into the percentage lost to Retrieval vs Generation.
  - **2×2 Confusion Matrix:** Visualizes the distribution across the 4 quadrants ($Q_1, Q_2, Q_3, Q_4$).
  - **Taxonomy Bar Chart:** Ranks the most frequent failure causes.

### Running Ablation Studies
* Navigate to **Tab 4: Ablation & Sensitivity**:
  1. Click **Run Top-K Sweep**: Generates line curves showing how increasing $K$ from 1 to 8 decreases retrieval error while increasing context distraction.
  2. Click **Run Strategy Comparison**: Compares **Dense** vs. **Sparse (BM25)** vs. **Hybrid (RRF)** to show how Hybrid search reduces keyword retrieval misses.
  3. **🤖 Model Capability Comparison (Holding Retrieval Constant)**:
     - In Section 4 of Tab 4, select two or more models (e.g. `qwen2.5:3b`, `llama3.2:latest`, and `local-fallback-extractor`).
     - Click **🚀 Run Model Capability Comparison**.
     - **Observation:** The system retrieves and freezes $\mathcal{C}_K$ for each benchmark query. Both models receive identical context chunks.
     - **Inspect:**
       - The **Comparative Metrics Table**: Compares Success Rate, Generation Failure Rate, F1, and Faithfulness.
       - The **Differential Failure Inspector**: Directly displays questions where one model succeeded while the other failed, highlighting the exact answers and root-cause failure subtype!

### Running Model Capability Comparison via Python CLI
You can also run the multi-model comparison headless from the terminal:
```bash
python -c "
import json
from src.retrieval.embedder import DenseEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LocalLLMClient
from src.evaluation.error_decomposer import ErrorDecomposer
from src.evaluation.ablation_engine import AblationEngine
from src.config import BENCHMARKS_DIR, SAMPLE_DOCS_DIR
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import RecursiveChunker

docs = DocumentLoader.load_directory(SAMPLE_DOCS_DIR)
chunks = RecursiveChunker().chunk_documents(docs)
embedder = DenseEmbedder()
v_store = VectorStore(dimension=embedder.dimension)
v_store.add_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
bm25 = BM25Retriever()
bm25.index_chunks(chunks)
hybrid = HybridRetriever(v_store, bm25, embedder)
llm = LocalLLMClient()
decomposer = ErrorDecomposer(embedder)
ablation = AblationEngine(hybrid, llm, decomposer)

with open(BENCHMARKS_DIR / 'quick_test_benchmark_10.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)[:5]

res = ablation.compare_models(bench, models=['qwen2.5:3b', 'llama3.2:latest'], top_k=3)
print(res['summary_df'][['Model', 'Success_Rate', 'Generation_Failure_Rate', 'Mean_Token_F1', 'Mean_Faithfulness']])
"
```

### Exporting Evaluation Data
* In **Tab 5: Sample Deep Dive**, click **📥 Download Full Evaluation Results (CSV)** to export all query outputs, similarity scores, faithfulness ratings, and diagnostic explanations for external spreadsheet analysis.

---

## 5. Running the REST API

To integrate or test programmatically with curl or Python:
1. Start the API:
   ```bash
   python run_app.py --api
   ```
2. Open interactive Swagger docs: `http://127.0.0.1:8000/docs`.
3. Test a live query:
   ```bash
   curl -X POST "http://127.0.0.1:8000/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Shor algorithm complexity?", "top_k": 3, "strategy": "hybrid"}'
   ```
4. Test benchmark evaluation:
   ```bash
   curl -X POST "http://127.0.0.1:8000/evaluate/benchmark" \
     -H "Content-Type: application/json" \
     -d '{"benchmark_name": "quick_10", "top_k": 3, "strategy": "hybrid"}'
   ```
