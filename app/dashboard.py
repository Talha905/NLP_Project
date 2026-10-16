"""
Streamlit Web Dashboard for RAG Retrieval-vs-Generation Error Decomposition & Failure Analysis.
Divided cleanly into two decoupled environments:
  1. 💬 Core RAG Assistant (Clean, minimal daily chatbot with real-time streaming & live diagnostics)
  2. 🔬 Failure Analysis Studio (Research laboratory for benchmarking, error decomposition, and ablations)
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import contextlib
import pandas as pd
import streamlit as st

@contextlib.contextmanager
def st_popover_compat(label: str):
    """Provides st.popover if available (Streamlit >=1.33), else falls back to st.expander."""
    if hasattr(st, "popover"):
        with st.popover(label):
            yield
    else:
        with st.expander(label, expanded=False):
            yield


def st_segmented_compat(label: str, options: list, default: str):
    """Provides st.segmented_control if available (Streamlit >=1.40), else falls back to horizontal st.radio."""
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options=options, default=default)
        return val or default
    return st.radio(
        label,
        options=options,
        index=options.index(default) if default in options else 0,
        horizontal=True
    )

from src.config import SAMPLE_DOCS_DIR, BENCHMARKS_DIR
from src.core.rag_engine import RAGApp, LiveDiagnostic
from src.evaluation.error_decomposer import ErrorDecomposer, DecompositionResult
from src.evaluation.ablation_engine import AblationEngine
from src.evaluation.failure_classifier import FailureTaxonomyAdvisor
from app.components.charts import (
    create_error_sankey,
    create_waterfall_attribution,
    create_quadrant_heatmap,
    create_subtype_bar_chart,
    create_top_k_sensitivity_chart,
    create_strategy_comparison_chart,
    create_model_comparison_chart,
    create_model_subtype_comparison_chart
)
from app.components.inspector import render_sample_inspector

st.set_page_config(
    page_title="RAG Error Decomposition & Core Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .app-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.1rem;
    }
    .diag-badge {
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
        margin-top: 6px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing RAG Engine and Local Models...")
def load_rag_engine():
    """Initializes the decoupled RAG application and evaluation components."""
    app = RAGApp()
    decomposer = ErrorDecomposer(app.embedder)
    ablation = AblationEngine(app.retriever, app.llm, decomposer)
    return {
        "rag_app": app,
        "decomposer": decomposer,
        "ablation_engine": ablation
    }


pipeline = load_rag_engine()
rag_app: RAGApp = pipeline["rag_app"]
ablation_engine: AblationEngine = pipeline["ablation_engine"]

# Initialize Session State
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "eval_result" not in st.session_state:
    st.session_state["eval_result"] = None
if "ablation_top_k" not in st.session_state:
    st.session_state["ablation_top_k"] = None
if "ablation_strategies" not in st.session_state:
    st.session_state["ablation_strategies"] = None
if "model_comparison_result" not in st.session_state:
    st.session_state["model_comparison_result"] = None
if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = "Core RAG Assistant"
if "prefill_query" not in st.session_state:
    st.session_state["prefill_query"] = ""


# ==============================================================================
# SIDEBAR: TOP-LEVEL NAVIGATION & SETTINGS
# ==============================================================================
st.sidebar.markdown("## 🧭 Workspace Navigation")
selected_mode = st.sidebar.radio(
    "Choose Environment:",
    ["💬 Core RAG Assistant", "🔬 Failure Analysis Studio"],
    index=0 if st.session_state.get("active_mode", "Core RAG Assistant") == "Core RAG Assistant" else 1,
    help="Switch between the clean daily RAG chatbot and the research testing lab."
)
st.session_state["active_mode"] = selected_mode

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Generation Model")

ollama_connected = rag_app.llm.is_ollama_available
if ollama_connected:
    st.sidebar.success("🟢 Ollama Local Service: Connected")
    models_list = rag_app.llm.list_available_models()
    active_model = st.sidebar.selectbox(
        "Active Model",
        models_list if models_list else ["qwen2.5:3b", "llama3.2:latest"]
    )
    rag_app.llm.model_name = active_model
else:
    st.sidebar.warning("🟡 Ollama Service: Offline\n(Using fast local grounded extractor)")
    active_model = "local-fallback-extractor"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Knowledge Base Status")
st.sidebar.caption(f"• Documents: **{len(rag_app.documents)}**")
st.sidebar.caption(f"• Chunks: **{len(rag_app.chunks)}**")
st.sidebar.caption(f"• Embedding: `{rag_app.embedder.model_name}`")
st.sidebar.caption(f"• Vector Dim: **{rag_app.embedder.dimension}**")


# ==============================================================================
# VIEW 1: CORE RAG ASSISTANT (Clean, Decoupled Daily Chatbot)
# ==============================================================================
if selected_mode == "💬 Core RAG Assistant":
    st.markdown('<div class="app-title">💬 Core RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">A clean document assistant powered by local LLMs with real-time streaming and automatic failure diagnostics.</div>', unsafe_allow_html=True)

    # Top Control Bar: Upload & Presets
    c_upload, c_preset, c_adv = st.columns([2, 2, 1])

    with c_upload:
        with st_popover_compat("📁 Upload Custom Document (PDF / TXT)"):
            uploaded = st.file_uploader("Select a PDF, TXT, or Markdown document:", type=["pdf", "txt", "md"])
            if uploaded and st.button("Index Uploaded Document"):
                with st.spinner("Processing & indexing..."):
                    save_path = SAMPLE_DOCS_DIR / uploaded.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    success = rag_app.index_uploaded_file(save_path)
                    if success:
                        st.session_state["selected_doc"] = uploaded.name
                        st.success(f"Indexed '{uploaded.name}'! Ready to ask questions.")
                        time.sleep(1)
                        st.rerun()

    with c_preset:
        preset_choice = st_segmented_compat(
            "RAG Profile:",
            options=["⚡ Fast", "⚖️ Balanced", "🎯 Deep Research"],
            default="⚖️ Balanced"
        )
        if preset_choice == "⚡ Fast":
            rag_top_k = 2
            rag_strategy = "dense"
            rag_style = "grounded"
        elif preset_choice == "🎯 Deep Research":
            rag_top_k = 5
            rag_strategy = "hybrid"
            rag_style = "cot"
        else:
            rag_top_k = 3
            rag_strategy = "hybrid"
            rag_style = "grounded"

    with c_adv:
        with st_popover_compat("⚙️ Advanced Settings"):
            rag_top_k = st.slider("Context Top-K", 1, 8, value=rag_top_k)
            rag_strategy = st.selectbox("Search Strategy", ["hybrid", "dense", "sparse"], index=["hybrid", "dense", "sparse"].index(rag_strategy))
            rag_style = st.selectbox("Prompt Strategy", ["grounded", "cot"], index=0 if rag_style == "grounded" else 1)
            force_offline = st.checkbox("Force Fast Synthesizer", value=not ollama_connected)

    # Document Scoping Control Bar: Focused vs Global
    c_scope1, c_scope2 = st.columns([1, 2])
    with c_scope1:
        search_scope = st.radio("Search Scope:", ["🎯 Focused (Selected Doc)", "🌐 Global (All Docs)"], index=0, horizontal=True)
    with c_scope2:
        available_docs = [
            Path(getattr(d, "filename", None) or getattr(d, "source_filename", None) or d.doc_id).name
            for d in rag_app.documents
        ]
        # Default to recently uploaded doc or assignment PDF if present
        default_idx = 0
        if "selected_doc" in st.session_state and st.session_state["selected_doc"] in available_docs:
            default_idx = available_docs.index(st.session_state["selected_doc"])
        else:
            for i, name in enumerate(available_docs):
                if "Assignment" in name or "23102B0025" in name:
                    default_idx = i
                    break
        selected_doc = st.selectbox("Active Target Document:", available_docs, index=default_idx if available_docs else 0) if search_scope.startswith("🎯") else None
        if selected_doc:
            st.session_state["selected_doc"] = selected_doc

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # If message is assistant, display sources and diagnostic
            if msg["role"] == "assistant" and "diagnostic" in msg:
                diag: LiveDiagnostic = msg["diagnostic"]

                # Render Diagnostic Badge
                st.markdown(
                    f"""
                    <div style="background-color: {diag.color}15; border: 1px solid {diag.color}40; border-radius: 6px; padding: 6px 12px; margin-top: 8px; margin-bottom: 8px;">
                        <span style="color: {diag.color}; font-weight: 700;">● {diag.badge_text}</span>
                        <span style="color: #475569; font-size: 0.88rem; margin-left: 8px;">— {diag.explanation}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Expandable sources
                if msg.get("sources"):
                    with st.expander("📚 Retrieved Source Chunks", expanded=False):
                        for s_idx, chunk in enumerate(msg["sources"], 1):
                            st.markdown(f"**Chunk {s_idx}** `({chunk['source']} | Score: {chunk['score']:.4f})`")
                            st.caption(chunk["text"])

                # If warning/failure occurred, offer quick link to Failure Studio
                if diag.status != "HEALTHY":
                    if st.button("🧪 Inspect in Failure Analysis Studio ➔", key=f"btn_send_{msg.get('msg_id', time.time())}"):
                        st.session_state["active_mode"] = "🔬 Failure Analysis Studio"
                        st.session_state["prefill_query"] = msg.get("query_text", "")
                        st.rerun()

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the indexed knowledge base...")
    if user_query:
        # Display user question
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Assistant generation with streaming
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context chunks..."):
                token_stream, source_dicts, chunk_objs = rag_app.stream_ask(
                    query=user_query,
                    top_k=rag_top_k,
                    strategy=rag_strategy,
                    prompt_style=rag_style,
                    model_name=active_model,
                    doc_id=selected_doc
                )

            # Stream the answer word by word
            full_answer = st.write_stream(token_stream)

            # Compute automatic live diagnosis
            diagnostic = rag_app.diagnose(user_query, chunk_objs, full_answer)

            # Display Diagnostic Badge immediately
            st.markdown(
                f"""
                <div style="background-color: {diagnostic.color}15; border: 1px solid {diagnostic.color}40; border-radius: 6px; padding: 6px 12px; margin-top: 8px; margin-bottom: 8px;">
                    <span style="color: {diagnostic.color}; font-weight: 700;">● {diagnostic.badge_text}</span>
                    <span style="color: #475569; font-size: 0.88rem; margin-left: 8px;">— {diagnostic.explanation}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Expandable sources
            with st.expander("📚 Retrieved Source Chunks", expanded=False):
                for s_idx, chunk in enumerate(source_dicts, 1):
                    st.markdown(f"**Chunk {s_idx}** `({chunk['source']} | Score: {chunk['score']:.4f})`")
                    st.caption(chunk["text"])

            # Save in history
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": full_answer,
                "sources": source_dicts,
                "diagnostic": diagnostic,
                "query_text": user_query,
                "msg_id": time.time()
            })
            st.rerun()


# ==============================================================================
# VIEW 2: FAILURE ANALYSIS STUDIO (The Research & Testing Workbench)
# ==============================================================================
else:
    st.markdown('<div class="app-title">🔬 Failure Analysis & Research Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">A controlled experimental laboratory for decomposing RAG errors into Retrieval vs Generation surfaces, running multi-model ablations, and diagnosing root causes.</div>', unsafe_allow_html=True)

    # Studio Sub-Tabs
    s_tab0, s_tab1, s_tab2, s_tab3, s_tab4, s_tab5 = st.tabs([
        "🔎 Single-Query Diagnostic Inspector",
        "🧪 Benchmark Evaluator",
        "📊 Error Decomposition",
        "🤖 Model Capability Lab",
        "📈 Parameter Sweeps (Top-K & Strategies)",
        "🔍 Deep Sample Explorer & CSV Export"
    ])

    # --------------------------------------------------------------------------
    # SUB-TAB 0: SINGLE-QUERY DEEP DIAGNOSTIC INSPECTOR
    # --------------------------------------------------------------------------
    with s_tab0:
        st.markdown("### 🔎 Single-Query Deep Diagnostic Inspector")
        st.caption("Inspects live retrieval relevance, cosine similarity scores, parent-child chunk expansion, and root-cause failure diagnostics for any individual query.")

        prefilled = st.session_state.get("prefill_query", "")
        inspect_q = st.text_input("Enter or inspect query:", value=prefilled, placeholder="e.g. what is the research gap analysed here")

        i_c1, i_c2, i_c3 = st.columns([1, 1, 1])
        with i_c1:
            i_top_k = st.slider("Inspect Top-K", 1, 8, value=3, key="i_top_k")
        with i_c2:
            i_strat = st.selectbox("Search Strategy", ["hybrid", "dense", "sparse"], index=0, key="i_strat")
        with i_c3:
            avail_docs = ["All Documents"] + [
                Path(getattr(d, "filename", None) or getattr(d, "source_filename", None) or d.doc_id).name
                for d in rag_app.documents
            ]
            i_doc_sel = st.selectbox("Scope to Document:", avail_docs, index=0, key="i_doc_sel")
            i_doc_filter = None if i_doc_sel == "All Documents" else i_doc_sel

        # Trigger on inspect button or if prefilled on initial redirect
        should_run = st.button("🔬 Run Deep Inspection", type="primary") or bool(prefilled)

        if inspect_q and should_run:
            print(f"[FAILURE_STUDIO] [INSPECT] Populated prefill_query='{inspect_q}' (doc_filter={i_doc_filter})")
            with st.spinner("Analyzing retrieval and context grounding..."):
                scored_chunks = rag_app.retrieve(inspect_q, top_k=i_top_k, strategy=i_strat, doc_id=i_doc_filter)
                chunks = [c for c, _ in scored_chunks]

                # Generate sample response and run diagnosis
                sample_ans = "The provided context does not contain sufficient evidence to answer this question." if not ollama_connected else rag_app.llm.generate(inspect_q, chunks)["answer"]
                diag = rag_app.diagnose(inspect_q, chunks, sample_ans)

            st.markdown("---")
            st.markdown(f"#### 🏷️ Diagnostic Status: <span style='color:{diag.color}; font-weight:700;'>● {diag.badge_text}</span>", unsafe_allow_html=True)
            st.info(f"**Root Cause Analysis:** {diag.explanation}")
            st.warning(f"**Actionable Mitigation:** {diag.mitigation}")

            st.markdown(f"#### 📚 Retrieved Chunks ({len(scored_chunks)} items)")
            for r_idx, (chunk, score) in enumerate(scored_chunks, 1):
                with st.expander(f"Chunk #{r_idx}: {chunk.chunk_id} | Score: {score:.4f} | Doc: {chunk.source_filename}", expanded=True):
                    sc_col1, sc_col2 = st.columns([1, 1])
                    with sc_col1:
                        st.markdown("**Search Text (Child Chunk):**")
                        st.code(chunk.text)
                    with sc_col2:
                        st.markdown("**Expanded Parent Context:**" if chunk.parent_text else "**Full Text:**")
                        st.code(chunk.parent_text or chunk.text)

    # --------------------------------------------------------------------------
    # SUB-TAB 1: BENCHMARK EVALUATOR
    # --------------------------------------------------------------------------
    with s_tab1:
        st.markdown("### 🧪 Batch Evaluation on Ground-Truth Benchmark")
        st.caption("Runs automated evaluation against gold evidence and answers to partition errors into Retrieval Failures ($R=0$) vs Generation Failures ($R=1, G=0$).")

        b_c1, b_c2, b_c3 = st.columns([1, 1, 1])
        with b_c1:
            benchmark_selection = st.radio(
                "Select Benchmark Dataset",
                ["Quick Test (10 Questions)", "Full Research Benchmark (50 Questions)", "Upload Custom JSON"],
                index=0
            )
        with b_c2:
            s_top_k = st.slider("Evaluation Top-K", 1, 8, value=3, key="s_top_k")
            s_strategy = st.selectbox("Retrieval Strategy", ["hybrid", "dense", "sparse"], index=0, key="s_strat")
        with b_c3:
            s_gen_mode = st.radio(
                "Generator Engine",
                ["Active Ollama Model" if ollama_connected else "Ollama (Offline)", "Fast Local Extractive Synthesizer"],
                index=0 if ollama_connected else 1
            )
            force_fallback_eval = ("Fast Local" in s_gen_mode)

        custom_eval_data = None
        if benchmark_selection == "Upload Custom JSON":
            up_bench = st.file_uploader("Upload JSON Benchmark:", type=["json"])
            if up_bench:
                custom_eval_data = json.load(up_bench)
                st.success(f"Loaded {len(custom_eval_data)} custom benchmark items.")

        if st.button("⚡ Start Benchmark Evaluation", type="primary"):
            if benchmark_selection == "Quick Test (10 Questions)":
                with open(BENCHMARKS_DIR / "quick_test_benchmark_10.json", "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
            elif benchmark_selection == "Full Research Benchmark (50 Questions)":
                with open(BENCHMARKS_DIR / "rag_failure_benchmark_50.json", "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
            else:
                eval_data = custom_eval_data

            if eval_data:
                prog = st.progress(0)
                status = st.empty()

                def cb(cur, tot):
                    prog.progress(cur / tot)
                    status.text(f"Evaluating query {cur}/{tot}...")

                res = ablation_engine.run_benchmark_evaluation(
                    benchmark_data=eval_data,
                    top_k=s_top_k,
                    strategy=s_strategy,
                    force_fallback=force_fallback_eval,
                    progress_callback=cb
                )
                st.session_state["eval_result"] = res
                status.success("Benchmark Evaluation Completed Successfully!")

        # KPI Summary Cards
        if st.session_state["eval_result"]:
            res: DecompositionResult = st.session_state["eval_result"]
            st.markdown("---")
            st.markdown("### 🎯 Benchmark KPI Summary")

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Queries", res.total_samples)
            k2.metric("Success Rate", f"{(res.success_count / res.total_samples)*100:.1f}%")
            k3.metric("Total Error Rate", f"{res.total_error_rate*100:.1f}%")
            k4.metric("Retrieval Attribution", f"{res.retrieval_attribution*100:.1f}%")
            k5.metric("Generation Attribution", f"{res.generation_attribution*100:.1f}%")
            k6.metric("Mean Faithfulness", f"{res.mean_faithfulness*100:.1f}%")

            summary = FailureTaxonomyAdvisor.generate_executive_summary(res.samples)
            if summary.get("top_failure_count", 0) > 0:
                st.info(f"""
                **Empirical Finding:**
                - **Primary Vulnerable Component:** `{summary.get('primary_vulnerable_component', 'NONE')}`
                - **Top Failure Mode:** `{summary.get('top_failure_subtype', 'NONE')}` ({summary.get('top_failure_count', 0)} occurrences)
                - **Actionable Priority Mitigation:** {summary.get('priority_recommendation', 'Inspect sample queries.')}
                """)
            else:
                st.success(f"""
                **Empirical Finding:**
                - **Primary Vulnerable Component:** `None (Optimal Performance)`
                - **Top Failure Mode:** `None` (0 failures detected)
                - **Actionable Priority Mitigation:** {summary.get('priority_recommendation', 'System demonstrates high end-to-end fidelity.')}
                """)

            with st.expander("📖 What do these KPI metrics mean and how are they derived?", expanded=False):
                st.markdown("""
                - **Total Queries ($N$)**: The total count of ground-truth test questions evaluated in this benchmark run.
                - **Success Rate ($S = N_{\\text{success}} / N$)**: Percentage of queries where both retrieval and generation succeeded ($R=1, G=1$), or unanswerable queries where the model correctly refused ($R=0, G=1$ unanswerable refusal).
                - **Total Error Rate ($E = (N_{\\text{ret\\_fail}} + N_{\\text{gen\\_fail}}) / N$)**: Proportion of queries that failed due to missing context or incorrect/hallucinated generation. *Parametric Memory Recall* ($R=0, G=1$) is tracked separately as spurious recall because it succeeded without grounded evidence.
                - **Retrieval Attribution ($\\alpha_R = N_{\\text{ret\\_fail}} / N_{\\text{errors}}$)**: The exact percentage of system failures caused by the retrieval stage (missing context, chunk boundary truncation, or distractor noise).
                - **Generation Attribution ($\\alpha_G = N_{\\text{gen\\_fail}} / N_{\\text{errors}}$)**: The exact percentage of system failures caused by the language model (hallucination, context contradiction, or reasoning failure). Note that $\\alpha_R + \\alpha_G = 100\\%$, pinpointing where engineering effort should be directed.
                - **Mean Faithfulness**: Average sentence-level NLI entailment rate across all answers ($F \\in [0.0, 1.0]$). Each sentence is independently evaluated against retrieved chunks using `cross-encoder/nli-deberta-v3-small`.
                """)

    # --------------------------------------------------------------------------
    # SUB-TAB 2: ERROR DECOMPOSITION & VISUALIZATIONS
    # --------------------------------------------------------------------------
    with s_tab2:
        st.markdown("### 📊 Mathematical Error Decomposition")
        st.caption("Visualizing the flow and attribution of errors across the Retrieval vs Generation boundary.")

        if not st.session_state["eval_result"]:
            st.info("Please run an evaluation in the 'Benchmark Evaluator' tab to populate this dashboard.")
            if st.button("Run Quick Test Benchmark Now (10 Questions)"):
                with open(BENCHMARKS_DIR / "quick_test_benchmark_10.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.session_state["eval_result"] = ablation_engine.run_benchmark_evaluation(data, top_k=3, force_fallback=True)
                st.rerun()
        else:
            res: DecompositionResult = st.session_state["eval_result"]
            r_dict = res.model_dump()

            c_sankey, c_waterfall = st.columns(2)
            with c_sankey:
                st.plotly_chart(create_error_sankey(r_dict), use_container_width=True)
            with c_waterfall:
                st.plotly_chart(create_waterfall_attribution(r_dict), use_container_width=True)

            c_quad, c_sub = st.columns(2)
            with c_quad:
                st.plotly_chart(create_quadrant_heatmap(r_dict), use_container_width=True)
            with c_sub:
                st.plotly_chart(create_subtype_bar_chart(res.subtype_breakdown), use_container_width=True)

    # --------------------------------------------------------------------------
    # SUB-TAB 3: MODEL CAPABILITY LAB (Frozen Context)
    # --------------------------------------------------------------------------
    with s_tab3:
        st.markdown("### 🤖 Model Capability Analysis (Constant Retrieval Context)")
        st.caption("Investigates: *'When retrieval is held constant, does the choice of generation model affect the frequency and type of generation failures?'*")
        st.info("💡 **Methodology:** The retriever executes once per query and the retrieved context chunks C_K are frozen. All models receive identical factual evidence. Any difference in generation correctness, F1 score, or failure subtypes is directly isolated to model capability.")

        avail_models = rag_app.llm.list_available_models()
        default_candidates = avail_models if avail_models else ["qwen2.5:3b", "llama3.2:latest"]
        all_options = list(dict.fromkeys(default_candidates + ["local-fallback-extractor"]))

        mc_col1, mc_col2, mc_col3 = st.columns([2, 1, 1])
        with mc_col1:
            selected_models_to_compare = st.multiselect(
                "Select Models to Compare (Retrieval Constant)",
                options=all_options,
                default=default_candidates[:2] if len(default_candidates) >= 2 else all_options[:2]
            )
        with mc_col2:
            mc_benchmark = st.selectbox("Benchmark Dataset", ["Quick Test (10 Questions)", "Full Benchmark (50 Questions)"], key="mc_bench_studio")
        with mc_col3:
            mc_top_k = st.slider("Context Depth (Top-K)", min_value=1, max_value=8, value=3, key="mc_top_k_studio")

        if st.button("🚀 Run Model Capability Comparison", type="primary"):
            if len(selected_models_to_compare) < 2:
                st.warning("Please select at least 2 models to compare.")
            else:
                with st.spinner(f"Evaluating {len(selected_models_to_compare)} models across frozen context chunks..."):
                    bench_file = "quick_test_benchmark_10.json" if "10" in mc_benchmark else "rag_failure_benchmark_50.json"
                    with open(BENCHMARKS_DIR / bench_file, "r", encoding="utf-8") as f:
                        bench_data = json.load(f)

                    comp_res = ablation_engine.compare_models(
                        benchmark_data=bench_data,
                        models=selected_models_to_compare,
                        top_k=mc_top_k,
                        strategy="hybrid",
                        force_fallback=False
                    )
                    st.session_state["model_comparison_result"] = comp_res

        if st.session_state.get("model_comparison_result") is not None:
            mc_res = st.session_state["model_comparison_result"]
            sum_df = mc_res["summary_df"]
            st_df = mc_res["subtype_df"]
            discordant = mc_res["discordant_cases"]

            st.markdown("##### 📊 Comparative Model Metrics (Held-Constant Context)")
            st.dataframe(sum_df, use_container_width=True)

            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.plotly_chart(create_model_comparison_chart(sum_df), use_container_width=True)
            with m_c2:
                st.plotly_chart(create_model_subtype_comparison_chart(st_df), use_container_width=True)

            # Discordant / Differential Failure Inspector
            st.markdown("##### 🔍 Differential Failure Cases (Where Model A != Model B)")
            st.caption("Cases where one model succeeded while another failed on the exact same retrieved evidence, or where failure subtypes shifted.")
            if not discordant:
                st.success("No divergent cases detected between models on this dataset.")
            else:
                for d_idx, d_case in enumerate(discordant, 1):
                    with st.expander(f"Case {d_idx}: [{d_case['divergence_type']}] - {d_case['question']}", expanded=(d_idx == 1)):
                        st.markdown(f"**Question:** {d_case['question']}")
                        st.markdown(f"**Gold Answer:** `{d_case['gold_answer']}`")
                        st.markdown("---")
                        c_win, c_fail = st.columns(2)
                        with c_win:
                            st.markdown(f"**Winner / Model 1 ({d_case['winning_model']}):**")
                            st.success(d_case["winner_answer"])
                            st.caption(f"F1 Score: {d_case['winner_f1']:.2f}")
                        with c_fail:
                            st.markdown(f"**Failing Model / Model 2 ({d_case['failing_model']}):**")
                            st.error(d_case["failure_answer"])
                            st.caption(f"Failure Subtype: `{d_case['failure_subtype']}`")
                        st.info(f"**Diagnostic Root Cause:** {d_case['failure_explanation']}")

    # --------------------------------------------------------------------------
    # SUB-TAB 4: PARAMETER SWEEPS (Top-K & Strategies)
    # --------------------------------------------------------------------------
    with s_tab4:
        st.markdown("### 📈 Multi-Factor Parameter Sweeps")
        st.caption("Empirical experiments demonstrating the trade-offs of retrieval depth and retrieval algorithms.")

        # Top-K Sweep
        st.markdown("#### 1. Top-K Sensitivity Sweep (Context Depth Trade-off)")
        st.caption("As Top-K increases, retrieval failure drops (higher recall), but generation failure can rise due to distractor noise.")
        if st.button("Run Top-K Parameter Sweep (K = 1, 2, 3, 5, 8)"):
            with st.spinner("Sweeping Top-K parameters..."):
                with open(BENCHMARKS_DIR / "quick_test_benchmark_10.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                df_top_k = ablation_engine.sweep_top_k(data, k_values=[1, 2, 3, 5, 8], force_fallback=True)
                st.session_state["ablation_top_k"] = df_top_k

        if st.session_state.get("ablation_top_k") is not None:
            df_k = st.session_state["ablation_top_k"]
            ck1, ck2 = st.columns([2, 1])
            with ck1:
                st.plotly_chart(create_top_k_sensitivity_chart(df_k), use_container_width=True)
            with ck2:
                st.dataframe(df_k[["Top_K", "Total_Error_Rate", "Retrieval_Failure_Rate", "Generation_Failure_Rate", "Mean_Token_F1"]], use_container_width=True)

            st.info("""
            **💡 How to interpret the Top-K Parameter Sweep:**
            - **Retrieval Failure Rate drops** as $K$ increases from 1 to 8 because retrieving more chunks increases the probability of capturing the required gold evidence (**Recall@K gain**).
            - **Generation Failure Rate rises or plateaus** at higher $K$ because feeding excessive text dilutes the prompt with irrelevant distractor paragraphs, increasing the likelihood of hallucination (**Context Dilution / Lost in the Middle**).
            - **The Trade-Off Sweet Spot ($K^*=3$)**: Notice where Total Error Rate is minimized. Top-K=3 achieves the optimal balance between high retrieval recall and low prompt noise.
            """)

        st.markdown("---")
        # Strategy Comparison
        st.markdown("#### 2. Retrieval Strategy Sweep (Dense vs Sparse BM25 vs Hybrid RRF)")
        if st.button("Run Retrieval Strategy Comparison"):
            with st.spinner("Comparing retrieval algorithms..."):
                with open(BENCHMARKS_DIR / "quick_test_benchmark_10.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                df_strat = ablation_engine.sweep_retrieval_strategies(data, top_k=3, force_fallback=True)
                st.session_state["ablation_strategies"] = df_strat

        if st.session_state.get("ablation_strategies") is not None:
            df_s = st.session_state["ablation_strategies"]
            cs1, cs2 = st.columns([2, 1])
            with cs1:
                st.plotly_chart(create_strategy_comparison_chart(df_s), use_container_width=True)
            with cs2:
                st.dataframe(df_s[["Strategy", "Total_Error_Rate", "Retrieval_Failures", "Generation_Failures", "Success_Count"]], use_container_width=True)

            st.info("""
            **💡 How to interpret Retrieval Strategies:**
            - **Dense (Vector Search)**: Excels at understanding abstract semantic concepts, paraphrases, and synonyms using `all-MiniLM-L6-v2`, but struggles with exact acronyms, numbers, and out-of-vocabulary technical identifiers.
            - **Sparse (BM25 Okapi)**: Excels at exact keyword and term matching (e.g. specific IDs, chemical equations, mathematical formulas), but misses synonymous phrasing.
            - **Hybrid (Reciprocal Rank Fusion)**: Combines the reciprocal rank orders of both Dense and Sparse retrieval ($RRF(d) = \\sum \\frac{1}{60 + r_m(d)}$), capturing both semantic intent and exact entity matches for lowest total error rate.
            """)

    # --------------------------------------------------------------------------
    # SUB-TAB 5: DEEP SAMPLE EXPLORER & CSV EXPORT
    # --------------------------------------------------------------------------
    with s_tab5:
        st.markdown("### 🔍 Individual Sample Failure Explorer & CSV Export")
        st.caption("Drill down into individual queries to inspect the ground truth, retrieved context chunks, model generation, and exact diagnostic cause.")

        if not st.session_state["eval_result"]:
            st.warning("Please run an evaluation in the 'Benchmark Evaluator' tab first.")
        else:
            res: DecompositionResult = st.session_state["eval_result"]
            samples = res.samples

            f1, f2, f3 = st.columns(3)
            with f1:
                quad_filter = st.selectbox(
                    "Filter by Quadrant",
                    ["ALL", "RETRIEVAL_FAILURE", "GENERATION_FAILURE", "PARAMETRIC_SUCCESS", "SUCCESS"]
                )
            with f2:
                all_subtypes = sorted(list(set(s.failure_subtype for s in samples)))
                subtype_filter = st.selectbox("Filter by Failure Subtype", ["ALL"] + all_subtypes)
            with f3:
                diff_filter = st.selectbox("Filter by Difficulty", ["ALL", "easy", "medium", "hard"])

            filtered = samples
            if quad_filter != "ALL":
                filtered = [s for s in filtered if s.quadrant == quad_filter]
            if subtype_filter != "ALL":
                filtered = [s for s in filtered if s.failure_subtype == subtype_filter]
            if diff_filter != "ALL":
                filtered = [s for s in filtered if s.difficulty == diff_filter]

            st.caption(f"Showing {len(filtered)} of {len(samples)} evaluated queries.")

            # CSV Download
            df_export = pd.DataFrame([s.model_dump() for s in samples])
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Evaluation Results (CSV)",
                data=csv_data,
                file_name="rag_error_decomposition_results.csv",
                mime="text/csv"
            )

            st.markdown("---")
            for s in filtered:
                render_sample_inspector(s.model_dump())
                st.markdown("---")
