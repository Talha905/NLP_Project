# Comprehensive Viva Voce Questions & Answers Guide

This guide contains **30 in-depth viva voce questions and answers** covering every theoretical, mathematical, architectural, and practical aspect of the **RAG Retrieval-vs-Generation Error Decomposition & Failure Analysis** project.

---

## 📑 Table of Contents
1. [Core Concepts & Motivation](#section-1-core-concepts--motivation)
2. [Mathematical Formulation & Error Decomposition](#section-2-mathematical-formulation--error-decomposition)
3. [Information Retrieval & Search Algorithms](#section-3-information-retrieval--search-algorithms)
4. [Dense Embeddings & Vector Indexing](#section-4-dense-embeddings--vector-indexing)
5. [Generation Engine & Local LLMs](#section-5-generation-engine--local-llms)
6. [Failure Taxonomy & Diagnostic Classification](#section-6-failure-taxonomy--diagnostic-classification)
7. [Experimental Methodology & Model Capability Ablation](#section-7-experimental-methodology--model-capability-ablation)
8. [Production Deployment, Decoupling & Real-Time Diagnostics](#section-8-production-deployment-decoupling--real-time-diagnostics)

---

## Section 1: Core Concepts & Motivation

### Q1: What is Retrieval-Augmented Generation (RAG) and why is it preferred over fine-tuning for enterprise knowledge bases?
**Answer:**
RAG combines an external information retrieval mechanism with a generative Large Language Model. Given a user query $q$, the system retrieves relevant document chunks $\mathcal{C}_K$ from a knowledge base and prepends them into the LLM's prompt context before generating an answer $\hat{a}$.
It is preferred over fine-tuning because:
1. **Dynamic Knowledge Updates**: Ingesting or deleting documents requires updating the vector index in seconds, whereas fine-tuning requires expensive re-training.
2. **Elimination of Hallucinations**: Answers are directly grounded in retrieved textual evidence with verifiable citations.
3. **Access Control & Privacy**: Role-based access control (RBAC) can be enforced at the retrieval layer before chunks reach the LLM.
4. **Lower Compute Cost**: Avoids multi-GPU parameter updates; runs easily on CPU.

### Q2: What is the central problem this project addresses?
**Answer:**
When an end-to-end RAG system outputs an incorrect answer, traditional evaluations (like BLEU, ROUGE, or accuracy) treat the system as a black box and assign a single scalar penalty. This fails to identify the root cause:
* Did the **search engine fail** to retrieve the necessary facts?
* Or did the **generator fail** to synthesize the answer despite having the evidence in its prompt?

This project develops a scientific framework to **mathematically decouple retrieval errors from generation errors**, categorize granular failure modes (like *Noise Dilution*, *Boundary Truncation*, and *Hallucinations*), and isolate model capability through controlled ablations.

### Q3: Why is this framework designed to be 100% local and GPU-independent?
**Answer:**
1. **Zero Financial Cost & Accessibility**: Uses open-weights models (`all-MiniLM-L6-v2`, `qwen2.5:3b`, `llama3.2:latest`) running via Ollama or a deterministic local synthesizer, requiring no paid OpenAI/Anthropic API keys.
2. **Data Privacy**: Enterprise documents never leave the local environment, satisfying GDPR, HIPAA, and proprietary NDA constraints.
3. **Deterministic Reproducibility**: Local execution avoids non-deterministic rate limits, API deprecations, or hidden server-side prompt updates.

---

## Section 2: Mathematical Formulation & Error Decomposition

### Q4: How are the binary indicators $R$ and $G$ defined mathematically?
**Answer:**
For a query $q$ with retrieved chunks $\mathcal{C}_K = \{c_1, \dots, c_K\}$, expected gold evidence $e^*$, and reference answer $a^*$:
1. **Retrieval Indicator ($R \in \{0, 1\}$)**:
   $$R = \mathbb{I}\left( \max_{c \in \mathcal{C}_K} \text{CosineSim}(\mathbf{e}_c, \mathbf{e}_{e^*}) \ge \tau_R \;\lor\; e^* \subseteq \mathcal{C}_K \right)$$
   Where $\tau_R = 0.72$ (retrieval threshold) and $\mathbf{e}$ are $L_2$-normalized dense embeddings.
2. **Generation Indicator ($G \in \{0, 1\}$)**:
   $$G = \mathbb{I}\left( \text{Token-F1}(\hat{a}, a^*) \ge \tau_A \;\lor\; \text{CosineSim}(\mathbf{e}_{\hat{a}}, \mathbf{e}_{a^*}) \ge \tau_S \right)$$
   Where $\tau_A = 0.50$ (lexical overlap) and $\tau_S = 0.78$ (semantic similarity).

### Q5: Explain the 4-Quadrant Outcome Space. What does each quadrant represent?
**Answer:**
Combining the binary values of $R$ and $G$ produces four distinct quadrants:
* **Quadrant 1 ($R=1, G=1$) - Total Success**: Gold evidence was present in the top-$K$ chunks, and the LLM generated an accurate answer.
* **Quadrant 2 ($R=1, G=0$) - Generation Failure**: Gold evidence was successfully retrieved, but the LLM failed (hallucinated, contradicted context, or refused to answer).
* **Quadrant 3 ($R=0, G=0$) - Retrieval Failure**: Gold evidence was absent from retrieved context, and the LLM produced an incorrect answer or refused.
* **Quadrant 4 ($R=0, G=1$) - Parametric Recall**: Gold evidence was absent from the context, yet the LLM answered correctly relying on pre-trained parametric weights.

### Q6: Why is Quadrant 4 (Parametric Recall) considered dangerous in enterprise RAG?
**Answer:**
While $G=1$ makes Quadrant 4 appear successful on paper, the answer was generated **without verifiable grounding in the retrieved context**. In an enterprise setting (e.g., querying private HR policies or internal codebases), the model's pre-trained weights cannot be trusted. If the model happens to guess right on general trivia, it will also unnoticeably hallucinate on private corporate facts. A trustworthy RAG system should refuse when evidence is missing ($R=0$), rather than relying on ungrounded pretraining memory.

### Q7: Derive the formulas for Total Error Rate, Retrieval Attribution ($\alpha_R$), and Generation Attribution ($\alpha_G$).
**Answer:**
Across $N$ evaluation queries:
* Total Errors: $N_{\text{err}} = N_{Q2} + N_{Q3}$
* **Total Error Rate ($E_{\text{total}}$)**:
  $$E_{\text{total}} = \frac{N_{Q2} + N_{Q3}}{N}$$
* **Retrieval Attribution ($\alpha_R$)**:
  $$\alpha_R = \frac{N_{Q3}}{N_{Q2} + N_{Q3}}$$
* **Generation Attribution ($\alpha_G$)**:
  $$\alpha_G = \frac{N_{Q2}}{N_{Q2} + N_{Q3}}$$
By definition: $\alpha_R + \alpha_G = \frac{N_{Q3} + N_{Q2}}{N_{Q2} + N_{Q3}} = 1.0$.
* If $\alpha_R > \alpha_G$: The search index, chunking, or retrieval pipeline is the primary bottleneck.
* If $\alpha_G > \alpha_R$: The LLM parameter capacity, prompt template, or reasoning capability is the primary bottleneck.

---

## Section 3: Information Retrieval & Search Algorithms

### Q8: What is Hybrid Search, and why is it superior to using Dense Vector Search alone?
**Answer:**
Dense semantic search (using bi-encoders) excels at conceptual paraphrasing and understanding intent, but often fails on exact keyword matching, technical abbreviations, part numbers, or specific person/model names (e.g., searching for `"Section 14.2(b)"` or `"RTX-4090"`).
Sparse lexical search (BM25 Okapi) excels at exact term frequencies and keyword matching, but cannot capture synonyms or semantic intent.
**Hybrid Search** combines both dense and sparse retrieval, capturing both semantic conceptual meaning and exact lexical matches, significantly reducing `MISSING_EVIDENCE` failures.

### Q9: How does Reciprocal Rank Fusion (RRF) work in your project?
**Answer:**
RRF is an unsupervised rank aggregation method that combines rankings from different retrieval systems without requiring raw score normalization (which is often problematic when combining unbounded BM25 scores with cosine similarities in $[0, 1]$).
For each candidate chunk $d$:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$
Where $k$ is a rank smoothing constant (set to $k=60$). A chunk ranked #1 in both systems receives the highest fused score, while chunks found by only one system still receive proportional credit.

### Q10: How does BM25 Okapi calculate relevance scores?
**Answer:**
Given query terms $q_i \in Q$ and document $D$:
$$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* $f(q_i, D)$ is term frequency in chunk $D$.
* $|D| / \text{avgdl}$ penalizes unusually long chunks.
* $k_1 = 1.5$ controls term frequency saturation limits.
* $b = 0.75$ controls document length normalization strength.

---

## Section 4: Dense Embeddings & Vector Indexing

### Q11: Which embedding model is used, and what are its properties?
**Answer:**
We use `sentence-transformers/all-MiniLM-L6-v2`:
* **Architecture**: 6-layer BERT-style Transformer distilled from a larger cross-encoder.
* **Vector Dimension ($d$)**: 384 dimensions.
* **Inference Speed**: ~14,000 sentences/second on GPU, and tens of queries/second on standard laptop CPU.
* **Context Limit**: 256 WordPiece tokens.

### Q12: How does FAISS `IndexFlatIP` compute cosine similarity?
**Answer:**
FAISS `IndexFlatIP` performs exhaustive inner product search:
$$\text{IP}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^d u_i \cdot v_i$$
To compute exact **cosine similarity**, we normalize all chunk embeddings and query embeddings to unit Euclidean length ($L_2$ norm = 1.0) before insertion and querying:
$$\|\mathbf{u}\|_2 = 1, \quad \|\mathbf{v}\|_2 = 1 \implies \text{IP}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \text{CosineSim}(\mathbf{u}, \mathbf{v})$$
This allows FAISS to compute exact cosine similarities via rapid SIMD-accelerated matrix multiplication with zero quantization loss.

### Q13: What is the difference between a Bi-Encoder and a Cross-Encoder? Why not use a Cross-Encoder for first-stage retrieval?
**Answer:**
* **Bi-Encoder**: Encodes query $q$ and document $d$ independently into vector embeddings $\mathbf{e}_q$ and $\mathbf{e}_d$. Chunk embeddings can be **precomputed and indexed** offline. At query time, retrieval requires only a single vector lookup in $\mathcal{O}(M \cdot d)$ time.
* **Cross-Encoder**: Passes query and document together through cross-attention layers: $\text{Model}([q; d])$. It is significantly more accurate because tokens in $q$ attend directly to tokens in $d$, but it requires a complete forward pass for every single document in the corpus, making it $\sim 100 \times$ slower and impractical for first-stage retrieval across thousands of chunks. Cross-encoders are best used as second-stage rerankers.

---

## Section 5: Generation Engine & Local LLMs

### Q14: How does your system integrate with Ollama?
**Answer:**
The system communicates with the locally running Ollama daemon via its HTTP REST API:
* `GET /api/tags`: Dynamically discovers all models installed locally on the user's disk (e.g., `qwen2.5:3b`, `llama3.2:latest`).
* `POST /api/generate`: Sends JSON payloads specifying the target model, user query, retrieved context, system prompt, temperature ($0.1$), and stream flag.
* If `stream: True`, it reads chunks asynchronously line-by-line for real-time word-by-word UI rendering.

### Q15: How does the system handle environments where Ollama is not installed?
**Answer:**
The framework implements a deterministic **Local Grounded Synthesizer** (`_local_grounded_fallback` in `src/generation/llm_client.py`).
If Ollama is not detected on port 11434, the engine automatically catches the connection exception and switches to Fallback Mode. It extracts the top sentences matching the query from the retrieved chunks and formats a strictly grounded response. This guarantees that **100% of unit tests, UI benchmarks, and failure analyses run out of the box with zero external dependencies**.

### Q16: Compare Strict Grounded Prompting versus Chain-of-Thought (CoT) Prompting.
**Answer:**
* **Strict Grounded Prompting**: Instructs the model: *"Answer strictly using only the provided context chunks. If the answer cannot be deduced from the context, state 'I do not have sufficient information.' Never extrapolate."* This minimizes hallucinations and enforces high faithfulness.
* **Chain-of-Thought (CoT) Prompting**: Instructs the model: *"Step 1: Identify key facts from the context. Step 2: Reason step-by-step. Step 3: Conclude the final answer."* This significantly reduces `REASONING_SYNTHESIS_ERROR` and `CONTEXT_CONTRADICTION` on multi-hop questions by forcing intermediate inference steps before generating the final answer.

---

## Section 6: Failure Taxonomy & Diagnostic Classification

### Q17: What is the "Noise Dilution" failure mode and how do you diagnose it?
**Answer:**
* **Description**: The gold evidence chunk was retrieved, but at a lower rank (e.g. rank 4 or 5), while the top ranks were filled with irrelevant or distractor chunks. The LLM's attention mechanism was distracted by the noise, causing it to overlook the evidence.
* **Diagnostic Condition**: $R=1$, but the gold evidence chunk rank is $\ge 3$, context similarity variance is high, and the generated answer failed ($G=0$).
* **Mitigation**: Lower Top-$K$ (e.g., from 8 to 3), or add a cross-encoder reranker to elevate relevant chunks to rank 1.

### Q18: What is "Boundary Truncation" and how is it detected?
**Answer:**
* **Description**: A fixed-window text chunker split a continuous sentence or paragraph across two separate chunks midway through a critical fact, resulting in neither chunk containing the complete coherent premise.
* **Diagnostic Condition**: The highest similarity chunk contains only a partial lexical match ($<60\%$ substring overlap) with gold evidence, and the adjacent chunk index contains the remaining fragment.
* **Mitigation**: Increase chunk size (e.g. 400 $\to$ 800 characters) and expand chunk overlap from 10% to 20%.

### Q19: What is "Conservative Refusal"? Why does it happen?
**Answer:**
* **Description**: The retrieved context chunks contained the exact answer, but the LLM conservatively replied: *"I don't have enough information to answer."*
* **Root Cause**: Overly defensive alignment/RLHF in the LLM or overly aggressive system prompts threatening penalties for hallucination, making the model risk-averse on complex questions.
* **Mitigation**: Soften the negative penalty phrasing in the system prompt; provide 1–2 few-shot examples demonstrating how to answer from partial evidence.

### Q20: How does your system detect "Hallucination (Unsupported)"?
**Answer:**
* **Method**: We compute the Natural Language Inference (NLI) / Lexical Grounding ratio between the generated answer $\hat{a}$ and the retrieved context $\mathcal{C}_K$.
* If the answer makes confident factual claims whose key content tokens (nouns, verbs, entities) have low lexical containment in $\mathcal{C}_K$, and the semantic alignment between $\hat{a}$ and $\mathcal{C}_K$ falls below threshold $\tau_{\text{faith}} = 0.55$, the sample is classified as `HALLUCINATION_UNSUPPORTED`.

---

## Section 7: Experimental Methodology & Model Capability Ablation

### Q21: What is the "Controlled Variable Principle" in your Model Capability Lab?
**Answer:**
In naive benchmarks, comparing two models involves running the entire RAG pipeline twice. However, if random seeds or different retrieval parameters return different chunks, you cannot know whether Model A was smarter or simply received better context.
**Our Controlled Variable Principle holds retrieval context $\mathcal{C}_K$ strictly constant**:
1. Run retrieval once for query $q$ to produce $\mathcal{C}_K$.
2. Feed the **identical** context $\mathcal{C}_K$ to Model A (`qwen2.5:3b`) and Model B (`llama3.2:latest`).
3. Any variance in accuracy, generation attribution ($\alpha_G$), or failure subtypes is **strictly attributable to model reasoning capacity**.

### Q22: What is a "Differential Failure Case" in cross-model analysis?
**Answer:**
A differential failure case is a specific query where:
$$\text{Outcome}(\text{Model}_A, \mathcal{C}_K) = \text{Generation Failure } (R=1, G=0)$$
while:
$$\text{Outcome}(\text{Model}_B, \mathcal{C}_K) = \text{Success } (R=1, G=1)$$
Because both models read the identical context, this isolates the exact linguistic or reasoning pattern that exceeded Model A's capacity but was solved by Model B.

### Q23: What trends occur when sweeping Top-$K$ from $1$ to $8$?
**Answer:**
* **At $K=1$**: Retrieval error ($E_R$) is highest because complex questions spanning multiple facts cannot fit into a single chunk.
* **As $K$ increases ($K=3 \to 5$)**: Retrieval Recall@K increases, and Total Error drops to an optimal minimum.
* **At high $K$ ($K \ge 8$)**: Retrieval error plateaus, but Generation error ($E_G$) begins to rise due to **Noise Dilution** and the **Lost in the Middle** effect, where LLM attention degrades over long context sequences.

---

## Section 8: Production Deployment, Decoupling & Real-Time Diagnostics

### Q24: Why is the Streamlit UI decoupled into "Core RAG Assistant" and "Failure Analysis Studio"?
**Answer:**
User experience research reveals that end-users (e.g. corporate employees asking questions) want a clean, fast, distraction-free chat interface without academic graphs or technical confusion matrices.
Conversely, AI engineers and researchers require deep benchmarking tools, parameter sweeps, Sankey diagrams, and failure taxonomies.
Decoupling into two modes in the sidebar provides **separation of concerns**:
* Mode 1: Clean daily productivity assistant with streaming text and a subtle health badge.
* Mode 2: Full-featured scientific research laboratory.

### Q25: How does the "Zero-Ground-Truth Live Diagnostic" work in the chat interface?
**Answer:**
In live production, gold answers $a^*$ are unavailable. The engine evaluates:
1. **Query-Context Semantic Relevance**: Computes cosine similarity between query $q$ and top chunk $c_1$. If $< 0.38$, flags 🟡 *Low Relevance Context / Distractor Information*.
2. **Sentence-Level Cross-Encoder NLI Faithfulness**: Decomposes generated answer $\hat{a}$ into factual claims and runs deep Natural Language Inference (`cross-encoder/nli-deberta-v3-small`) against retrieved parent-chunk contexts:
   * **Contradiction Check**: If any claim directly contradicts context facts ($p_{\text{contra}} \ge 0.50$), flags 🔴 *Direct Context Contradiction*.
   * **Entailment Check**: If claims lack context support ($p_{\text{entail}} < 0.60$), flags 🔴 *Potential Hallucination Detected*.
3. **Refusal Verification**: Uses regex detection to identify appropriate refusals (e.g., *"insufficient evidence"*), verifying whether evidence was truly missing or if the model exhibited over-conservative refusal.
4. If all claims are entailed, flags 🟢 *Grounded & Supported (Healthy)*.

### Q26: Explain the role of the Plotly Sankey Diagram in error visualization.
**Answer:**
The Sankey diagram visualizes the multi-stage flow of information through the RAG pipeline:
* **Stage 1 (Total Queries)** splits into *Evidence Retrieved ($R=1$)* vs *Evidence Missed ($R=0$)*.
* **Stage 2** splits both retrieval branches into *Generation Success ($G=1$)* vs *Generation Failure ($G=0$)*.
* **Stage 3** routes failures into specific fine-grained diagnostic nodes (`HALLUCINATION`, `NOISE_DILUTION`, `REFUSAL`, etc.).
This gives stakeholders an immediate visual understanding of where queries drop out of the pipeline.

### Q27: How does token-level streaming work without blocking the UI?
**Answer:**
The backend uses Python generators (`yield token`).
In `src/generation/llm_client.py`, the HTTP request to Ollama specifies `stream: True`. As Ollama emits JSON token chunks over the HTTP socket, the generator yields them.
In the Streamlit frontend, `st.write_stream(token_stream)` consumes the generator, appending tokens to the active DOM element in real-time, delivering immediate visual responsiveness.

### Q28: How does the FastAPI backend complement the Streamlit interface?
**Answer:**
Streamlit is optimized for interactive human exploration, but cannot be easily integrated into enterprise microservices.
The FastAPI backend (`src/api/app.py`) provides asynchronous REST endpoints:
* `POST /ask`: Synchronous programmatic Q&A with diagnostic metadata.
* `POST /stream`: Server-Sent Events (SSE) streaming token endpoint.
* `POST /index`: Ingests documents programmatically.
* `POST /evaluate`: Executes automated regression benchmark evaluations in external CI/CD pipelines.

### Q29: What are the main architectural enhancements recently integrated into this framework?
**Answer:**
1. **Two-Stage Retrieval with Cross-Encoder Reranking**: We deployed `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage filter after Hybrid RRF to re-score candidates via full cross-attention.
2. **Hierarchical Parent-Child Chunking**: We index small child chunks (~250c) for search precision while expanding to full parent sections (~1200c) in LLM generation.
3. **Sentence-Level Cross-Encoder NLI**: Upgraded live diagnostics from naive token matching to `cross-encoder/nli-deberta-v3-small` with explicit contradiction detection.
4. **Document-Scoped Retrieval Filtering**: Added `doc_id` scoping to isolate search to specific documents, preventing BM25 cross-document dilution.
5. **Executive Summary Meta-Routing**: Automatically generates structured document overviews during ingestion to answer synthesis queries.

### Q30: What is Parent-Child Hierarchical Chunking and why does it outperform naive fixed chunking?
**Answer:**
Naive chunking forces an engineering compromise: small chunks (e.g. 200c) improve vector search resolution but truncate sentences and strip away context needed by the LLM. Large chunks (e.g. 1500c) preserve context but dilute vector embeddings with unrelated sentences, degrading retrieval precision.
**Parent-Child Hierarchical Chunking** solves this:
* **Child Chunks (~250 characters)**: Extracted and indexed into FAISS and BM25. Their tight semantic focus maximizes recall and precision during search.
* **Parent Sections (~1200 characters)**: Enclose several contiguous child chunks. When a child chunk is retrieved, the engine resolves its `parent_id` and provides the full, coherent parent section to the LLM prompt, deduplicating overlapping parents.

### Q31: Why is a Two-Stage Retrieval pipeline (Bi-Encoder/BM25 + Cross-Encoder Reranker) necessary?
**Answer:**
* **First Stage (Bi-Encoder & BM25)**: Employs independent encoding where query and document vectors are computed separately. This allows pre-computing vector indices for fast nearest-neighbor search ($\mathcal{O}(d \cdot M)$), but misses fine-grained token-level cross-interactions.
* **Second Stage (Cross-Encoder)**: Feeds query and candidate text concatenated together into BERT/MiniLM cross-attention layers. This computes full cross-attention between every query word and every document word, catching semantic subtleties and false positives that fooled the bi-encoder.
Because cross-encoders are computationally expensive, we run them only on the top candidate pool ($K_{\text{cand}} = \max(2K, 6)$) from Stage 1.

### Q32: How does Sentence-Level NLI Faithfulness scoring detect subtle hallucinations and direct contradictions?
**Answer:**
Rather than computing bag-of-words token overlap or cosine similarity across the entire answer, the system decomposes the generated answer into individual sentence claims. Each sentence is treated as a hypothesis, and the retrieved context chunks serve as premises for an NLI model (`cross-encoder/nli-deberta-v3-small`):
* If $p_{\text{entailment}} \ge 0.60$, the claim is factually grounded.
* If $p_{\text{contradiction}} \ge 0.50$, the claim explicitly contradicts stated facts (e.g., claiming "Transformers use sequential recurrence" when the context states "Transformers eliminate recurrence").
* If neither condition holds, the claim is flagged as an unsupported hallucination.

### Q33: What is the cause of multi-document BM25 dilution, and how does Document-Scoped filtering resolve it?
**Answer:**
BM25 scores documents based on term frequency and inverse document frequency. In a multi-document corpus, if a user uploads a document and asks a query whose keywords only appear in one chunk of that document, BM25 scores all other chunks of that document as $0.0$.
During Reciprocal Rank Fusion (RRF), chunks from other unrelated documents in the corpus that happen to contain generic keywords fill the remaining top-$K$ slots, diluting the context with irrelevant noise.
**Document-Scoped Filtering (`doc_id`)** resolves this by enforcing a hard SQL/index predicate on the target document, ensuring that only chunks belonging to the selected document compete for top-$K$ ranks.

### Q34: How were the decision thresholds $\tau_R=0.72, \tau_A=0.50, \tau_S=0.78$ derived empirically?
**Answer:**
Rather than setting arbitrary heuristics, we executed an empirical parameter sweep against the 50-query validation benchmark (`data/benchmarks/rag_failure_benchmark_50.json`):
1. **$\tau_R = 0.72$**: Swept cosine similarity thresholds between retrieved candidate chunks and true gold evidence vs distractor chunks. $\tau_R = 0.72$ yielded the optimal $F_1$ peak ($0.989$) with $100\%$ precision, rejecting distractor noise.
2. **$\tau_A = 0.50$**: Swept token-level F1 overlap between reference answers and model paraphrases vs incorrect answers. $\tau_A = 0.50$ provided $1.000$ precision and $1.000$ recall, avoiding false positives from superficial stopword matches.
3. **$\tau_S = 0.78$**: Swept dense semantic cosine similarity between reference answers and generated paraphrases. $\tau_S = 0.78$ achieved optimal semantic separation ($F_1 = 0.840$), correctly discriminating between conceptually accurate paraphrases and subtly flawed statements.

### Q35: If you had only 60 seconds to summarize your project to an examiner, what would you say?
**Answer:**
*"Our project addresses the core diagnostic ambiguity in Retrieval-Augmented Generation: when a RAG system answers incorrectly, is the search engine failing, or is the language model failing? We built a 100% open-source, local framework that mathematically decomposes total system error into Retrieval Attribution ($\alpha_R$) and Generation Attribution ($\alpha_G$). We deploy a production-grade Two-Stage Retrieval pipeline with Hybrid RRF and Cross-Encoder reranking, Hierarchical Parent-Child chunking, and Sentence-Level NLI Faithfulness scoring. We empirically calibrated our decision thresholds against benchmark sweeps and decoupled the application into a daily streaming assistant with live self-diagnostics and a deep research laboratory for controlled cross-model ablations."*
