# 🧪 Comprehensive RAG Failure Modes Testing Guide

This guide provides step-by-step instructions to **trigger, reproduce, and observe every single failure mode** that this framework is designed to detect and decompose.

Each section lists the **formal definition**, the **exact test query**, the **required UI settings**, what **diagnostic badge and quadrant** will appear, and the **underlying mathematical trigger**.

---

## 📑 Summary: The 4 Quadrants & 10 Failure Modes

```
                                    RETRIEVAL STATUS
                         R = 0 (Retrieval Fail)      R = 1 (Retrieval Pass)
                    +-----------------------------+-----------------------------+
                    |                             |                             |
       G = 1        |        QUADRANT 3:          |        QUADRANT 1:          |
 (Generation Pass)  |    PARAMETRIC RECALL /      |          SUCCESS            |
                    |       SPURIOUS RECALL       |   (Evidence present and     |
                    |  (Answered from pre-trained |     answer is grounded)     |
                    |    weights without context) |                             |
                    +-----------------------------+-----------------------------+
                    |                             |                             |
       G = 0        |        QUADRANT 2:          |        QUADRANT 4:          |
 (Generation Fail)  |      RETRIEVAL FAILURE      |     GENERATION FAILURE      |
                    |   (Missing context, noise   |   (Hallucination, synthesis |
                    |   dilution, or boundary)    |   error, or contradiction)  |
                    |                             |                             |
                    +-----------------------------+-----------------------------+
```

---

## 🔍 Part 1: Testing Retrieval Failures ($R=0, G=0$)

### 1. `MISSING_EVIDENCE` (Information Absent from Index)
* **What it means**: The knowledge base contains zero factual evidence capable of answering the question.
* **How to trigger in 💬 Core RAG Assistant**:
  1. Scope: `🌐 Global (All Docs)`
  2. Query: `"What is the fusion plasma temperature and magnetic confinement time of the ITER tokamak divertor?"`
* **Expected Diagnostic**:
  * Badge: `🟡 Low Relevance Context`
  * Metric: Maximum retrieval cosine similarity $S_{ret} < 0.60$.
  * Explanation: *"The retrieved text chunks have low semantic similarity to your query. The answer may be based on distractor information."*
* **How to observe in 🔬 Failure Analysis Studio**:
  * In the **Benchmark Evaluator**, this appears under Quadrant 2 (`RETRIEVAL_FAILURE`) with subtype `MISSING_EVIDENCE`.

---

### 2. `NOISE_DILUTION` (Context Dilution / Distractor Paragraphs)
* **What it means**: Relevant information was present in the corpus, but setting $K$ too high allowed irrelevant distractor chunks to dilute the prompt.
* **How to trigger in 🔬 Failure Analysis Studio**:
  1. Navigate to **`📈 Parameter Sweeps -> 1. Top-K Sensitivity Sweep`**.
  2. Click **`Run Top-K Parameter Sweep (K = 1, 2, 3, 5, 8)`**.
  3. Inspect the graph at $K=8$. Notice that while Retrieval Recall peaks, Generation Failure rate increases because the model is overwhelmed by distractor noise.

---

### 3. `TRUNCATION_BOUNDARY` (Entity Split Across Chunks)
* **What it means**: A complex formula or multi-sentence premise was cut in half by fixed-length character chunking.
* **How to observe**:
  * In the **Deep Sample Explorer**, filter by `TRUNCATION_BOUNDARY`. The system shows that while half of the sentence was in chunk A, the dependent clause was in chunk B, causing the generator to receive an incomplete premise.

---

## 🤖 Part 2: Testing Generation Failures ($R=1, G=0$)

### 4. `HALLUCINATION_UNSUPPORTED` (Model Fabricates Claims)
* **What it means**: Factual context was successfully retrieved in Top-K ($R=1$), but the language model introduced extraneous claims not supported by the context.
* **How to trigger in 🤖 Model Capability Lab**:
  1. In **Model Capability Lab**, select models: `qwen2.5:3b` and `synthetic-hallucinator`.
  2. Dataset: `Quick Test (10 Questions)`, Top-K: `3`.
  3. Click **`🚀 Run Model Capability Comparison`**.
* **What you will see**:
  * `qwen2.5:3b` achieves **83.9% Faithfulness** ($G=1$).
  * `synthetic-hallucinator` drops to **43.3% Faithfulness** ($G=0$).
  * In the **Differential Failure Cases** table, expand **Case 1**:
    * **Failing Model Output**: The synthetic model generated: *"...peer-reviewed trials conducted by Dr. Thorne at Stanford (2024) confirmed a 43.8% performance anomaly..."*
    * **Failure Subtype**: `HALLUCINATION_UNSUPPORTED`.
    * **Diagnostic Root Cause**: *"Evidence was present, but the LLM hallucinated ungrounded claims not supported by the context."*
    * **NLI Verification**: Sentence-level DeBERTa flagged the claim with Entailment = 0.000.

---

### 5. `CONTEXT_CONTRADICTION` (Model Directly Contradicts Evidence)
* **What it means**: Evidence was present in the context, but the LLM asserted the exact opposite of what the context stated.
* **How to trigger in 🤖 Model Capability Lab**:
  1. Compare `qwen2.5:3b` vs `synthetic-weak-reasoner`.
  2. `synthetic-weak-reasoner` deliberately inverts logical relationships (e.g. asserting that the scaling factor increases dot products rather than reducing them).
* **Expected Diagnostic**:
  * Subtype: `CONTEXT_CONTRADICTION`.
  * Trigger: DeBERTa NLI contradiction probability $P_{contra} \ge 0.60$.
  * Mitigation: Lower model temperature to 0.0 or reinforce negative constraint instructions.

---

### 6. `REASONING_SYNTHESIS_ERROR` (Multi-Hop Deductive Failure)
* **What it means**: All required factual premises were retrieved across multiple chunks, but the model failed to connect the logical dots.
* **How to trigger in 🔬 Benchmark Evaluator**:
  1. Run `Quick Test (10 Questions)` with `qwen2.5:3b`.
  2. Look at **Query 9**: *"Explain the positive ice-albedo feedback loop mechanism in the Arctic."*
  3. While individual facts (albedo values, solar absorption) were retrieved, small 3B models often fail to synthesize the complete multi-hop causal chain.
  4. The evaluator accurately classifies this as `REASONING_SYNTHESIS_ERROR` ($R=1, G=0$).

---

### 7. `CONSERVATIVE_REFUSAL` (False Refusal When Evidence Was Present)
* **What it means**: The evidence was retrieved in Top-K, but the LLM exhibited over-conservative behavior, answering *"The provided context does not contain sufficient evidence to answer this question."*
* **How to observe**:
  * In the **Deep Sample Explorer**, filter by `CONSERVATIVE_REFUSAL`.
  * This happens when prompt constraints are excessively aggressive or when 3B models are queried about tabular marksheets without column reference headers.

---

### 8. `INCOMPLETE_ANSWER` (Partial Coverage of Multi-Part Query)
* **What it means**: The query asked for multiple fields (e.g. operating temperature, coherence time, and gate speed), but the LLM only provided one or two.
* **How to trigger**:
  * In **Sample Inspector**, query: *"Compare the operating temperatures, coherence times, and gate speeds between superconducting transmons and trapped ion qubits."*
  * If the model omits gate speeds, Token F1 drops below 0.50, and the system flags `INCOMPLETE_ANSWER`.

---

## 🧠 Part 3: Testing Spurious & Adversarial Modes

### 9. `PARAMETRIC_RECALL` ($R=0, G=1$: Answered from Memory Without Evidence)
* **What it means**: The retriever failed to find the evidence in the corpus, but the LLM answered correctly anyway by recalling facts memorized during pre-training.
* **How to trigger in 🔬 Failure Analysis Studio**:
  1. In the **Benchmark Evaluator**, run `Quick Test (10 Questions)`.
  2. Inspect **Query 3**: *"What structural refinements do modern LLMs like LLaMA incorporate beyond standard Transformer architectures?"*
  3. Retrieval Precision is 0.00 (the specific LLaMA architecture chunk was absent from Top-K).
  4. However, the LLM answered with high accuracy (Token F1 = 0.62, Semantic Sim = 0.75) because LLaMA's architecture was heavily present in its training weights!
  5. The system flags this as **`PARAMETRIC_SUCCESS (R=0, G=1)`** with subtype `PARAMETRIC_RECALL`.

---

### 10. `UNANSWERABLE_CORRECT_REFUSAL` (Adversarial Query Handled Correctly)
* **What it means**: An adversarial out-of-domain question was asked that has no answer in reality or in the knowledge base. The model correctly refused.
* **How to trigger in 💬 Core RAG Assistant**:
  1. Query: `"What was the exact battery capacity and warp drive core temperature of the Apollo 11 lunar module?"`
  2. The LLM responds: *"The provided context does not contain sufficient evidence to answer this question."*
* **Expected Diagnostic**:
  * Badge: `● Appropriate Refusal` (Green).
  * In the Benchmark Evaluator (Query 10), this is classified as **`SUCCESS`** with subtype `UNANSWERABLE_CORRECT_REFUSAL`.

---

## 📊 Summary of Failure Subtypes & Engineering Mitigations

| Failure Subtype | Primary Cause | Severity | Recommended Engineering Mitigation |
| :--- | :--- | :---: | :--- |
| `MISSING_EVIDENCE` | Retrieval | HIGH | Expand document corpus; add domain-specific data. |
| `NOISE_DILUTION` | Retrieval | MEDIUM | Reduce Top-K (optimal $K=3$); add Cross-Encoder reranker. |
| `TRUNCATION_BOUNDARY` | Ingestion | MEDIUM | Use Parent-Child hierarchical chunking (Parent ~1200c, Child ~250c). |
| `HALLUCINATION_UNSUPPORTED`| Generation | HIGH | Lower temperature to 0.0; enforce strict grounded prompt. |
| `CONTEXT_CONTRADICTION` | Generation | CRITICAL | Add negative constraints ("Do not extrapolate"); run post-generation NLI. |
| `REASONING_SYNTHESIS_ERROR`| Generation | HIGH | Switch prompt style to Chain-of-Thought (CoT); upgrade to larger model. |
| `CONSERVATIVE_REFUSAL` | Generation | MEDIUM | Remove parrot-inducing refusal strings; inject table schema headers. |
| `INCOMPLETE_ANSWER` | Generation | LOW | Request structured JSON output with explicit schema keys. |
| `PARAMETRIC_RECALL` | Spurious Recall | MEDIUM | Warning: answers may diverge from private enterprise documents. |
| `UNANSWERABLE_CORRECT_REFUSAL`| Success | NONE | Ideal behavior for out-of-domain adversarial prompts. |
