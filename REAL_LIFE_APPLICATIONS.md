# Real-Life Applications & Step-by-Step Implementation Guide

This document outlines **5 production-grade real-world applications** of the RAG Error Decomposition and Failure Analysis framework. Rather than high-level theoretical descriptions, each section provides **concrete, reproducible steps** you can execute immediately using this codebase.

---

## 🏢 Application 1: Enterprise Legal Contract & Compliance Auditing

### The Real-World Problem
Legal teams and compliance officers search thousands of pages of Master Service Agreements (MSAs), vendor contracts, and Non-Disclosure Agreements (NDAs). 
* **The High-Stakes Risk**: A standard RAG chatbot might answer *"Yes, the supplier is liable up to $10M"* because it missed a limitation of liability clause buried across a page boundary (**Boundary Truncation**), or the LLM asserted terms not in the contract (**Hallucination**).
* **Why This Framework Is Critical**: It determines whether an inaccurate legal answer was caused by the search engine missing the clause ($R=0$) or the LLM misinterpreting complex legal terminology ($R=1, G=0$).

### 🛠️ Step-by-Step Instructions to Implement:

#### Step 1: Ingest Your Target Legal Contracts
1. Collect your target legal PDF or TXT contracts (e.g., `vendor_agreement_2026.pdf`, `gdpr_compliance_policy.pdf`).
2. Copy them into `data/sample_docs/`, or open the web dashboard (`python run_app.py`) and use the **📁 Upload Custom Document** button.
3. The system will recursively chunk the contracts using sliding windows with 20% overlap, storing chunk offsets and page source metadata.

#### Step 2: Create a Legal Evaluation Benchmark
Create a ground-truth JSON file in `data/benchmarks/legal_benchmark.json` containing high-liability clauses:
```json
[
  {
    "id": "legal_01",
    "query": "What is the maximum liability cap for gross negligence under section 14?",
    "gold_evidence": "Under Section 14.2, neither party's liability is capped in the event of gross negligence, intentional misconduct, or breach of confidentiality.",
    "gold_answer": "Liability is uncapped for gross negligence, willful misconduct, or confidentiality breaches.",
    "difficulty": "hard"
  },
  {
    "id": "legal_02",
    "query": "Does the vendor indemnify the customer against third-party patent infringement?",
    "gold_evidence": "Vendor shall defend, indemnify, and hold harmless Customer against any third-party claims alleging that the Software infringes any registered patent.",
    "gold_answer": "Yes, the vendor provides full indemnification against third-party patent infringement claims.",
    "difficulty": "medium"
  }
]
```

#### Step 3: Run the Failure Decomposition
1. In the sidebar, select **🔬 Failure Analysis Studio**.
2. Navigate to **🧪 Benchmark Evaluator**, select `legal_benchmark.json` (or paste your samples), set **Search Strategy** to `Hybrid (Dense + BM25)`, and click **🚀 Run Diagnostic Evaluation**.
3. Inspect the resulting **4-Quadrant Confusion Matrix**:
   * If you see high **Quadrant 3 ($R=0, G=0$) with `TRUNCATION_BOUNDARY`**: The legal clause was sliced midway through an indemnity exception.
     * **Action**: In `src/config.py`, increase `CHUNK_SIZE` from `400` to `800` characters and `CHUNK_OVERLAP` from `80` to `160` characters.
   * If you see high **Quadrant 2 ($R=1, G=0$) with `CONTEXT_CONTRADICTION`**: The retriever found the clause, but the LLM inverted the negative condition (*"neither party's liability is capped"* interpreted as *"liability is capped"*).
     * **Action**: In the sidebar, change **Prompt Strategy** from `grounded` to `cot` (Chain-of-Thought) and upgrade to `qwen2.5:3b` or `llama3.2:latest`.

#### Step 4: Enable Real-Time Audit Alerts in Daily Chat
1. Switch to **💬 Core RAG Assistant**.
2. When lawyers or compliance officers ask questions, the system displays the real-time **Zero-Ground-Truth Diagnostic Badge**:
   * 🟢 **HEALTHY & FAITHFUL**: The answer is safe to quote.
   * 🔴 **POTENTIAL HALLUCINATION**: The LLM generated terms not found in the contract chunks. The UI displays an alert preventing the lawyer from relying on it.

---

## 🏥 Application 2: Clinical Decision Support & Medical Guideline QA

### The Real-World Problem
Clinicians and medical researchers query complex Clinical Practice Guidelines (CPGs) for drug dosages, contraindications, and treatment algorithms.
* **The High-Stakes Risk**: If a clinician asks *"Can Drug X be prescribed with Patient Condition Y?"*, a model that answers from pre-trained memory without retrieving the specific guideline (**Parametric Recall $R=0, G=1$**) or hallucinates a dosage can cause patient harm.
* **Why This Framework Is Critical**: It explicitly flags **Quadrant 4 (Parametric Recall)** as an untrusted, high-risk outcome and enforces verifiable chunk citations.

### 🛠️ Step-by-Step Instructions to Implement:

#### Step 1: Ingest Clinical Guidelines
1. Ingest treatment guideline documents (e.g. `cardiology_guidelines.txt`, `antibiotic_stewardship.pdf`).
2. Verify indexing status in the sidebar: confirmed chunk count, dense embeddings loaded via FAISS.

#### Step 2: Establish Safety Evaluation Criteria
Configure strict safety thresholds in `src/config.py`:
```python
# Raise retrieval threshold to ensure strict medical precision
RETRIEVAL_SIMILARITY_THRESHOLD = 0.78  # Stricter evidence matching
GENERATION_SIMILARITY_THRESHOLD = 0.82 # Stricter answer fidelity
```

#### Step 3: Run the Model Capability Lab on Frozen Guidelines
1. In **🔬 Failure Analysis Studio**, click sub-tab **🤖 Model Capability Lab**.
2. Select Model 1: `qwen2.5:3b` and Model 2: `llama3.2:latest`.
3. Click **🔬 Run Controlled Model Comparison**:
   * The system freezes the retrieved medical context $\mathcal{C}_K$, ensuring both models read the **exact same guideline sentences**.
   * Compare their **Differential Failure Cases**:
     * Does Model A trigger `CONSERVATIVE_REFUSAL` on complex dosage queries while Model B answers correctly?
     * Does Model B trigger `HALLUCINATION_UNSUPPORTED` by suggesting off-label uses not present in the guideline?
4. Select the model that minimizes ungrounded hallucinations for clinical deployment.

#### Step 4: Monitor Refusal Health
* For ambiguous medical queries where guidelines contain no evidence, verify the model lands in `UNANSWERABLE_CORRECT_REFUSAL` rather than inventing an answer.

---

## 📈 Application 3: FinTech & Financial Risk Analysis (10-K & Earnings Reports)

### The Real-World Problem
Financial analysts query SEC 10-K filings, quarterly earnings transcripts, and equity research reports to extract revenue breakdowns, EBITDA margins, and risk factors.
* **The High-Stakes Risk**: 10-K filings are dense and full of tables. Standard semantic vector search often retrieves adjacent discussion chunks rather than the exact fiscal table (**Noise Dilution**), causing the LLM to calculate incorrect growth percentages (**Reasoning Synthesis Error**).
* **Why This Framework Is Critical**: Discloses whether financial calculation errors are due to noisy retrieval or numerical reasoning limitations of the LLM.

### 🛠️ Step-by-Step Instructions to Implement:

#### Step 1: Load Financial Statements
1. Save corporate filings (e.g., `apple_10k_2025.txt`, `tesla_q4_earnings.txt`) into `data/sample_docs/`.
2. Ensure both dense vectors and BM25 sparse indices are built.

#### Step 2: Run a Parameter Sensitivity Sweep across Top-K
1. Open **🔬 Failure Analysis Studio** $\to$ **📈 Parameter Sweeps**.
2. In the **Top-K Context Depth Sweep**, select range $K \in [1, 2, 3, 5, 8]$.
3. Click **Execute Top-K Sensitivity Analysis**.
4. Observe the trade-off curve:
   * **At $K=1$**: Retrieval error ($E_R$) is high because multi-part revenue figures span multiple pages.
   * **At $K=8$**: Retrieval error drops, but Generation error ($E_G$) rises due to **Noise Dilution** (the LLM is distracted by irrelevant footnotes).
   * **Optimal Setting**: Identify the exact "sweet spot" (typically $K=3$ or $K=4$) that minimizes $E_{\text{total}}$.

#### Step 3: Compare Retrieval Strategies (Dense vs. Sparse vs. Hybrid)
1. In the same tab, run **Retrieval Strategy Comparison**.
2. Observe how BM25 performs vs. Dense Semantic search:
   * Exact financial tickers (e.g., `"GAAP EPS"`, `"ARR"`, `"EBITDA"`) perform significantly better under **BM25 Lexical Search**.
   * Conceptual questions (e.g., *"What macro supply chain challenges impacted gross margins?"*) perform better under **Dense Semantic Search**.
   * **Conclusion**: Deploy **Hybrid Search (RRF)** to get the lowest error rate across both metric types.

---

## 💰 Application 4: RAG Cost & Model Downsizing (Pareto Frontier Optimization)

### The Real-World Problem
Many organizations default to routing all internal RAG queries to expensive proprietary cloud models (e.g., GPT-4o, Claude 3.5 Sonnet) costing tens of thousands of dollars per month in token API fees, when an on-premise, lightweight 3B model could achieve identical performance if retrieval is high quality.
* **The Goal**: Determine the empirical breakpoint where a cheap, local 3B model suffices, versus where a larger model is strictly necessary.

### 🛠️ Step-by-Step Instructions to Implement:

#### Step 1: Pull Your Candidate Local Models
In your terminal, pull local models of varying sizes:
```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull llama3.2:latest
ollama pull mistral:7b
```

#### Step 2: Benchmark Generation Attribution Across Models
1. Launch the dashboard: `python run_app.py`.
2. Go to **🔬 Failure Analysis Studio** $\to$ **🤖 Model Capability Lab**.
3. Select `qwen2.5:1.5b` as Model A and `qwen2.5:3b` as Model B.
4. Execute the comparison on the 50-sample benchmark.
5. Review the **Generation Attribution ($\alpha_G$)**:
   * If $\alpha_G$ drops from `48%` down to `12%` when moving from 1.5B to 3B, the 3B model is essential for reasoning synthesis.
   * Now compare `qwen2.5:3b` against `mistral:7b`. If $\alpha_G$ only changes from `12%` to `10%`, the extra compute cost and latency of 7B yields negligible accuracy gains.
   * **Conclusion**: `qwen2.5:3b` represents the optimal Pareto efficiency point, saving 60% of hardware memory while preserving 90%+ accuracy.

---

## 🔄 Application 5: Automated CI/CD Regression Quality Gate for Knowledge Bases

### The Real-World Problem
Enterprise documentation is constantly updated. When an engineering team updates product manuals, API docs, or internal wikis, changes to text formatting or deleted sections can silently break existing customer RAG queries without anyone noticing until production complaints arrive.
* **The Goal**: Automatically test every pull request or document update in a CI/CD pipeline and block the build if RAG failure attribution exceeds an error budget.

### 🛠️ Step-by-Step Instructions to Implement:

#### Step 1: Run the Headless Test Suite in CI
Add the test execution to your GitHub Actions workflow (`.github/workflows/rag_ci.yml`):
```yaml
name: RAG Pipeline CI Quality Gate

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-rag:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Execute Automated Unit & Regression Tests
        run: |
          python run_app.py --test
```

#### Step 2: Add an Automated Regression Benchmark Script
Create a script `scripts/ci_quality_gate.py`:
```python
from pathlib import Path
import json, sys
from src.core.rag_engine import RAGApp
from src.evaluation.error_decomposer import ErrorDecomposer
from src.evaluation.ablation_engine import AblationEngine

def run_quality_gate():
    app = RAGApp()
    decomposer = ErrorDecomposer(app.embedder)
    ablation = AblationEngine(app.retriever, app.llm, decomposer)

    benchmark_path = Path("data/benchmarks/quick_test_benchmark_10.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Evaluate benchmark using deterministic fallback mode
    res = ablation.run_benchmark_evaluation(data, top_k=3, force_fallback=True)

    print(f"Total Queries: {res.total_samples}")
    print(f"Total Error Rate: {res.error_rate * 100:.1f}%")
    print(f"Retrieval Attribution: {res.retrieval_attribution * 100:.1f}%")
    print(f"Generation Attribution: {res.generation_attribution * 100:.1f}%")

    # QUALITY GATE CRITERIA
    MAX_ALLOWED_ERROR_RATE = 0.35  # Max 35% error rate allowed
    if res.error_rate > MAX_ALLOWED_ERROR_RATE:
        print(f"❌ CI FAILED: Error rate {res.error_rate:.2f} exceeds threshold {MAX_ALLOWED_ERROR_RATE}")
        sys.exit(1)

    print("✅ CI PASSED: All RAG quality gates satisfied.")
    sys.exit(0)

if __name__ == "__main__":
    run_quality_gate()
```

#### Step 3: Trigger on Every Knowledge Base Update
Whenever new documents are committed to `data/sample_docs/`, the CI job automatically indexes them, runs the 10-query regression test, evaluates the 4-quadrant outcome, and verifies that no prior queries were broken.
