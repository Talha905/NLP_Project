# 🚀 How to Run (With or Without Ollama)

This guide walks you through setting up and running the **RAG Retrieval-vs-Generation Error Decomposition & Failure Analysis** framework on any machine (Windows, macOS, Linux).

---

## ⚡ TL;DR: Can I run this without Ollama?

**YES, 100%!** 

> **No GPU, no Ollama, and no paid API keys are required.**
> 
> The project includes an **Intelligent Local Grounded Synthesizer**. If Ollama is not installed or not running, the system **automatically detects this and switches to Fallback Mode**. 
> 
> In Fallback Mode:
> * Document ingestion and text chunking work identically.
> * Dense Bi-encoder semantic retrieval (`sentence-transformers`), BM25 Okapi, and Hybrid Search work identically.
> * Grounded answer generation and streaming chat work via rule-based contextual synthesis.
> * The **4-Quadrant Error Decomposition** ($R$ vs $G$), Sankey diagrams, Waterfall charts, Top-K sweeps, and failure taxonomy diagnostics work completely offline.
> * All automated unit and integration tests pass without Ollama.

---

## 📋 Step 1: Clone and Set Up Environment

### 1. Clone the repository
```bash
git clone https://github.com/Talha905/NLP_Project.git
cd NLP_Project
```

### 2. Create and activate a virtual environment (Recommended)

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If PowerShell shows a script execution error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

* **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install required Python packages
```bash
pip install -r requirements.txt
```
*(Dependencies: `streamlit`, `sentence-transformers`, `torch`, `rank-bm25`, `fastapi`, `uvicorn`, `plotly`, `pandas`, `requests`, `scikit-learn`)*

---

## 🎮 Step 2: Run the Application

The project includes a single CLI runner: `run_app.py`.

### Option A: Launch the Interactive Web Dashboard (Recommended)
```bash
python run_app.py
```
*(Or alternatively: `streamlit run app/dashboard.py`)*

1. Open your browser to **`http://localhost:8501`**.
2. Look at the left sidebar:
   * If Ollama is not installed, it will display: **`Ollama: Offline (Fallback Mode Active)`**.
   * Everything is ready to use!
3. Switch between the two modes in the sidebar:
   * **💬 Core RAG Assistant**: Ask questions (e.g. about Transformers, Quantum Computing, or Climate Science) and receive streaming responses with real-time zero-ground-truth grounding diagnostics.
   * **🔬 Failure Analysis Studio**: Run 10-query or 50-query benchmarks, explore the 4-Quadrant error breakdown, inspect Sankey & Waterfall charts, and run Top-K parameter sweeps.

---

### Option B: Run Automated Unit Tests
To verify all retrieval, decomposition, and fallback generation components work properly on your system:
```bash
python run_app.py --test
```
You should see:
```text
Ran 12 tests in ~15-20s
OK
```

---

### Option C: Launch the FastAPI REST Server
If you want to integrate the RAG pipeline into external services:
```bash
python run_app.py --api
```
Interactive Swagger documentation will be available at: **`http://127.0.0.1:8000/docs`**.

---

## 🦙 (Optional) Adding Ollama for Real Local Neural LLMs

If you want actual open-weights large language models generating answers (instead of the built-in fallback synthesizer), follow these quick steps:

### 1. Download and Install Ollama
* Visit **[https://ollama.com/download](https://ollama.com/download)** and install for Windows, macOS, or Linux.

### 2. Pull Lightweight Models
Open a new terminal and pull any local models of your choice:
```bash
# Recommended lightweight models (fast on standard laptop CPU):
ollama pull qwen2.5:3b
ollama pull llama3.2:latest
```

### 3. Ensure Ollama is Running
By default, the Ollama desktop app runs in your system tray on port `11434`. You can also start it manually:
```bash
ollama serve
```

### 4. Restart or Refresh the Dashboard
Once Ollama is running:
* The web dashboard sidebar will automatically update to: **`Ollama: Online (localhost:11434)`**.
* The **Model Selector** dropdown will populate with your downloaded models (`qwen2.5:3b`, `llama3.2:latest`, etc.).
* In **🔬 Failure Analysis Studio -> Model Capability Analysis**, you can now compare two different models on the exact same retrieved context to test model capability failures.

---

## 🔍 Comparison: Running With vs Without Ollama

| Feature | Without Ollama (Fallback Mode) | With Ollama (Local Neural LLM) |
| :--- | :---: | :---: |
| **Setup Complexity** | Zero external installs | Requires Ollama app + model pull |
| **Hardware Required** | Low CPU, ~1.5 GB RAM | Laptop CPU / 4–8 GB RAM |
| **Document Ingestion** | Full support | Full support |
| **Hybrid Retrieval (Dense + BM25)** | Full support | Full support |
| **Streaming Chat Assistant** | Built-in Extractive Synthesizer | Generative neural model (`qwen2.5`, `llama3.2`) |
| **Failure Diagnosis ($R$ vs $G$)** | Full support | Full support |
| **Top-K & Strategy Sweeps** | Full support | Full support |
| **Cross-Model Capability Ablation** | Simulates Fallback vs Synthesizer | Compares multiple real Ollama models under identical context |
| **Unit & Integration Tests** | 12/12 Passing | 12/12 Passing |

---

## ❓ Troubleshooting & FAQs

### 1. "Failed to connect to Ollama / Fallback mode active"
* **This is completely normal and expected** if you haven't installed Ollama. The app functions out of the box using its built-in rule synthesizer. You can use the app without changing anything.
* If you *do* have Ollama installed, make sure it is running (`ollama serve` or check if `http://localhost:11434` responds in your browser).

### 2. First Run Delay (Downloading Sentence-Transformers)
* On the first run, PyTorch will download the lightweight bi-encoder embedding model `sentence-transformers/all-MiniLM-L6-v2` (~80 MB). This is cached locally in `~/.cache/huggingface/` and only happens once.

### 3. Port 8501 is already in use
* If port 8501 is occupied, you can launch Streamlit on an alternate port:
  ```bash
  streamlit run app/dashboard.py --server.port 8502
  ```

### 4. Script Execution Policy Error on Windows PowerShell
* If you see `File ... cannot be loaded because running scripts is disabled on this system`:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```

---

## 📚 Further Reading

* For an in-depth breakdown of how every specific failure mode is triggered and analyzed, see **[RUN_AND_TEST_GUIDE.md](RUN_AND_TEST_GUIDE.md)**.
* For the mathematical formulas, system architecture, and research methodology, see **[README.md](README.md)**.
