"""
Inspector component for drill-down into individual RAG query failures.
"""

from typing import Dict, Any
import streamlit as st
from src.evaluation.failure_classifier import FailureTaxonomyAdvisor


def render_sample_inspector(sample: Dict[str, Any], all_chunks_dict: Dict[str, Any] = None):
    """Renders detailed failure diagnostic card for an evaluation sample."""
    quadrant = sample.get("quadrant", "UNKNOWN")

    # Color badge styling
    badge_colors = {
        "SUCCESS": ("#059669", "#D1FAE5", "End-to-End Success (R=1, G=1)"),
        "GENERATION_FAILURE": ("#D97706", "#FEF3C7", "Generation Failure (R=1, G=0)"),
        "RETRIEVAL_FAILURE": ("#DC2626", "#FEE2E2", "Retrieval Failure (R=0, G=0)"),
        "PARAMETRIC_SUCCESS": ("#7C3AED", "#EDE9FE", "Parametric Memory Recall (R=0, G=1)")
    }
    color, bg, title = badge_colors.get(quadrant, ("#475569", "#F1F5F9", quadrant))

    st.markdown(
        f"""
        <div style="background-color: {bg}; border-left: 6px solid {color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 15px;">
            <div style="color: {color}; font-weight: 700; font-size: 15px; text-transform: uppercase;">
                {title}
            </div>
            <div style="color: #1E293B; margin-top: 4px; font-size: 14px;">
                <b>Taxonomy Subtype:</b> <code>{sample.get('failure_subtype', 'NONE')}</code> | 
                <b>Difficulty:</b> {sample.get('difficulty', 'N/A').title()} | 
                <b>Type:</b> {sample.get('question_type', 'N/A').title()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Token F1", f"{sample.get('token_f1', 0.0):.2f}")
    col2.metric("Semantic Sim", f"{sample.get('semantic_similarity', 0.0):.2f}")
    col3.metric("Faithfulness", f"{sample.get('faithfulness_score', 0.0):.2f}")
    col4.metric("Retrieval Precision", f"{sample.get('retrieval_precision', 0.0):.2f}")

    st.markdown(f"**Question:** {sample.get('question', '')}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Ground Truth")
        st.info(f"**Gold Answer:**\n\n{sample.get('gold_answer', '')}")
        st.caption(f"**Gold Evidence:** {sample.get('gold_evidence', '')}")

    with c_right:
        st.markdown("#### Generated Answer")
        if sample.get("generation_success"):
            st.success(sample.get("generated_answer", ""))
        else:
            st.error(sample.get("generated_answer", ""))

    st.markdown(f"**Diagnostic Root Cause:** *{sample.get('diagnostic_explanation', '')}*")

    # Mitigation recommendations if failure
    subtype = sample.get("failure_subtype", "NONE")
    if subtype != "NONE" and subtype != "UNANSWERABLE_CORRECT_REFUSAL":
        guide = FailureTaxonomyAdvisor.get_mitigation(subtype)
        with st.expander(f"Recommended Mitigations for {subtype}", expanded=True):
            st.markdown(f"**Component:** `{guide['component']}` | **Severity:** `{guide['severity']}`")
            st.markdown(f"**Root Cause:** {guide['root_cause']}")
            st.markdown("**Actionable Engineering Steps:**")
            for m in guide["mitigations"]:
                st.markdown(f"- {m}")
