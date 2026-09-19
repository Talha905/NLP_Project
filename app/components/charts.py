"""
Plotly interactive chart components for RAG Error Decomposition dashboard.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_error_sankey(result_dict: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive Sankey diagram tracing:
    Total Queries -> Retrieval Stage (Success/Fail) -> Generation Stage (Success/Fail) -> Subtypes
    """
    succ = result_dict.get("success_count", 0)
    gen_fail = result_dict.get("generation_failure_count", 0)
    ret_fail = result_dict.get("retrieval_failure_count", 0)
    param = result_dict.get("parametric_success_count", 0)
    subtypes = result_dict.get("subtype_breakdown", {})

    labels = [
        "All Benchmark Queries",          # 0
        "Retrieval Success (R=1)",        # 1
        "Retrieval Failure (R=0)",        # 2
        "End-to-End Success",             # 3
        "Generation Failure",             # 4
        "Retrieval Error",                # 5
        "Parametric Recall"               # 6
    ]

    sources = [0, 0, 1, 1, 2, 2]
    targets = [1, 2, 3, 4, 5, 6]
    values = [
        succ + gen_fail,                  # All -> R=1
        ret_fail + param,                 # All -> R=0
        succ,                             # R=1 -> Success
        gen_fail,                         # R=1 -> Gen Fail
        ret_fail,                         # R=0 -> Ret Fail
        param                             # R=0 -> Parametric
    ]

    # Append subtypes from failures
    current_node = 7
    for st_name, count in subtypes.items():
        if count > 0:
            labels.append(f"{st_name} ({count})")
            # Connect to generation failure or retrieval failure
            if any(k in st_name for k in ["HALLUCINATION", "CONTRADICTION", "REASONING", "INCOMPLETE", "REFUSAL"]):
                sources.append(4)  # Gen failure
            else:
                sources.append(5)  # Ret failure
            targets.append(current_node)
            values.append(count)
            current_node += 1

    node_colors = [
        "#1E293B", "#10B981", "#EF4444", "#059669", "#F59E0B", "#DC2626", "#8B5CF6"
    ] + ["#64748B"] * (len(labels) - 7)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(160, 174, 192, 0.35)"
        )
    )])

    fig.update_layout(
        title_text="<b>Flow of RAG Failures (Sankey Decomposition)</b>",
        font_size=12,
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def create_waterfall_attribution(result_dict: Dict[str, Any]) -> go.Figure:
    """
    Creates a waterfall chart decomposing Total Error into Retrieval vs Generation surfaces.
    """
    total = max(1, result_dict.get("total_samples", 1))
    ret_fails = result_dict.get("retrieval_failure_count", 0)
    gen_fails = result_dict.get("generation_failure_count", 0)
    succ = result_dict.get("success_count", 0)
    param = result_dict.get("parametric_success_count", 0)

    ret_pct = (ret_fails / total) * 100
    gen_pct = (gen_fails / total) * 100
    total_err_pct = ret_pct + gen_pct

    fig = go.Figure(go.Waterfall(
        name="Error Decomposition",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Retrieval Failure (R=0)", "Generation Failure (R=1, G=0)", "Total Error Rate"],
        y=[ret_pct, gen_pct, total_err_pct],
        text=[f"{ret_pct:.1f}%", f"{gen_pct:.1f}%", f"{total_err_pct:.1f}%"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#DC2626"}}
    ))

    fig.update_layout(
        title="<b>Error Attribution Decomposition (% of All Queries)</b>",
        yaxis_title="Percentage of Total Queries (%)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_quadrant_heatmap(result_dict: Dict[str, Any]) -> go.Figure:
    """
    Creates a 2x2 confusion matrix of Retrieval (Success/Failure) vs Generation (Success/Failure).
    """
    succ = result_dict.get("success_count", 0)
    gen_fail = result_dict.get("generation_failure_count", 0)
    ret_fail = result_dict.get("retrieval_failure_count", 0)
    param = result_dict.get("parametric_success_count", 0)

    z = [
        [succ, gen_fail],       # Retrieval Success: Gen Success, Gen Fail
        [param, ret_fail]       # Retrieval Failure: Gen Success, Gen Fail
    ]
    annotations = [
        [f"<b>SUCCESS (Q1)</b><br>Count: {succ}<br>Evidence & Answer OK",
         f"<b>GENERATION FAIL (Q2)</b><br>Count: {gen_fail}<br>Evidence in context, LLM failed"],
        [f"<b>PARAMETRIC (Q4)</b><br>Count: {param}<br>Missing evidence, LLM guessed",
         f"<b>RETRIEVAL FAIL (Q3)</b><br>Count: {ret_fail}<br>Missing evidence & wrong answer"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Generation Success (G=1)", "Generation Failure (G=0)"],
        y=["Retrieval Success (R=1)", "Retrieval Failure (R=0)"],
        colorscale="Blues",
        showscale=False,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 13}
    ))

    fig.update_layout(
        title="<b>2×2 Retrieval vs Generation Outcome Matrix</b>",
        xaxis_title="Generation Outcome",
        yaxis_title="Retrieval Outcome",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_subtype_bar_chart(subtype_dict: Dict[str, int]) -> go.Figure:
    """Horizontal bar chart of fine-grained failure modes."""
    if not subtype_dict:
        fig = go.Figure()
        fig.update_layout(title="No Failure Subtypes Recorded")
        return fig

    items = sorted(subtype_dict.items(), key=lambda x: x[1], reverse=True)
    names = [i[0].replace("_", " ").title() for i in items]
    counts = [i[1] for i in items]

    colors = []
    for name in names:
        n_up = name.upper()
        if any(w in n_up for w in ["HALLUCINATION", "CONTRADICTION", "REASONING", "REFUSAL"]):
            colors.append("#F59E0B")  # Generation color
        else:
            colors.append("#EF4444")  # Retrieval color

    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker=dict(color=colors),
        text=counts,
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Fine-Grained Failure Taxonomy Distribution</b>",
        xaxis_title="Occurrences",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_top_k_sensitivity_chart(df: pd.DataFrame) -> go.Figure:
    """Interactive line chart of Error Rates vs Top-K."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Top_K"], y=df["Total_Error_Rate"],
        mode="lines+markers", name="Total Error Rate",
        line=dict(color="#DC2626", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df["Top_K"], y=df["Retrieval_Failure_Rate"],
        mode="lines+markers", name="Retrieval Failure Rate",
        line=dict(color="#EF4444", dash="dash", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["Top_K"], y=df["Generation_Failure_Rate"],
        mode="lines+markers", name="Generation Failure Rate",
        line=dict(color="#F59E0B", dash="dot", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["Top_K"], y=df["Mean_Token_F1"],
        mode="lines+markers", name="Mean Token F1",
        line=dict(color="#10B981", width=2)
    ))

    fig.update_layout(
        title="<b>Sensitivity of Failure Rates Across Top-K Context Depth</b>",
        xaxis_title="Top-K Context Chunks",
        yaxis_title="Rate / Score",
        height=400,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_strategy_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing Dense vs Sparse vs Hybrid retrieval."""
    fig = go.Figure(data=[
        go.Bar(name="Retrieval Failures", x=df["Strategy"], y=df["Retrieval_Failures"], marker_color="#EF4444"),
        go.Bar(name="Generation Failures", x=df["Strategy"], y=df["Generation_Failures"], marker_color="#F59E0B"),
        go.Bar(name="Successes", x=df["Strategy"], y=df["Success_Count"], marker_color="#10B981")
    ])
    fig.update_layout(
        barmode="group",
        title="<b>Error Breakdown by Retrieval Strategy (Dense vs Sparse vs Hybrid)</b>",
        xaxis_title="Retrieval Algorithm",
        yaxis_title="Number of Queries",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_model_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing generation failure rates and metrics across models with frozen context."""
    fig = go.Figure(data=[
        go.Bar(name="Success Rate", x=df["Model"], y=df["Success_Rate"], marker_color="#10B981"),
        go.Bar(name="Gen Failure Rate", x=df["Model"], y=df["Generation_Failure_Rate"], marker_color="#F59E0B"),
        go.Bar(name="Mean F1", x=df["Model"], y=df["Mean_Token_F1"], marker_color="#3B82F6"),
        go.Bar(name="Mean Faithfulness", x=df["Model"], y=df["Mean_Faithfulness"], marker_color="#8B5CF6")
    ])
    fig.update_layout(
        barmode="group",
        title="<b>Model Capability Impact Under Constant Retrieval (R Held Constant)</b>",
        xaxis_title="Generation Model",
        yaxis_title="Rate / Score (0.0 to 1.0)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_model_subtype_comparison_chart(subtype_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing failure subtype frequencies across models under identical context."""
    fig = go.Figure()
    models = subtype_df["Model"].tolist()
    subtypes = [col for col in subtype_df.columns if col != "Model"]

    colors = px.colors.qualitative.Plotly
    for idx, st_name in enumerate(subtypes):
        fig.add_trace(go.Bar(
            name=st_name.replace("_", " ").title(),
            x=models,
            y=subtype_df[st_name],
            marker_color=colors[idx % len(colors)]
        ))

    fig.update_layout(
        barmode="group",
        title="<b>Generation Failure Subtypes by Model (Same Retrieved Context)</b>",
        xaxis_title="Model",
        yaxis_title="Number of Queries",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

