"""
Ablation and Multi-Factor Sensitivity Analysis Engine.
Quantifies how Error Decomposition changes with Top-K, Retrieval Strategy,
Chunk Size, and Question Difficulty.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from .error_decomposer import ErrorDecomposer, DecompositionResult
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generation.llm_client import LocalLLMClient


class AblationEngine:
    """
    Runs multi-factor ablation sweeps to study failure trends
    across hyperparameters and query attributes.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LocalLLMClient,
        decomposer: ErrorDecomposer
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.decomposer = decomposer

    def run_benchmark_evaluation(
        self,
        benchmark_data: List[Dict[str, Any]],
        top_k: int = 3,
        strategy: str = "hybrid",
        prompt_style: str = "grounded",
        force_fallback: bool = False,
        progress_callback: Optional[Any] = None
    ) -> DecompositionResult:
        """Runs full evaluation on a given benchmark dataset."""
        samples = []
        total = len(benchmark_data)

        for idx, item in enumerate(benchmark_data):
            query = item["question"]
            # 1. Retrieve
            retrieved_chunks = [c for c, _ in self.retriever.retrieve(
                query, top_k=top_k, strategy=strategy
            )]

            # 2. Generate
            gen_output = self.llm_client.generate(
                query=query,
                retrieved_chunks=retrieved_chunks,
                prompt_style=prompt_style,
                force_fallback=force_fallback
            )

            # 3. Evaluate & Decompose
            sample = self.decomposer.evaluate_sample(
                sample_meta=item,
                retrieved_chunks=retrieved_chunks,
                generation_output=gen_output
            )
            samples.append(sample)

            if progress_callback:
                progress_callback(idx + 1, total)

        return self.decomposer.aggregate_results(samples)

    def sweep_top_k(
        self,
        benchmark_data: List[Dict[str, Any]],
        k_values: List[int] = [1, 2, 3, 5, 8],
        strategy: str = "hybrid",
        force_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Sweeps Top-K values to measure the trade-off between
        retrieval recall gain and context noise generation failure.
        """
        records = []
        for k in k_values:
            res = self.run_benchmark_evaluation(
                benchmark_data,
                top_k=k,
                strategy=strategy,
                force_fallback=force_fallback
            )
            records.append({
                "Top_K": k,
                "Total_Error_Rate": res.total_error_rate,
                "Retrieval_Failure_Rate": res.retrieval_failure_count / res.total_samples,
                "Generation_Failure_Rate": res.generation_failure_count / res.total_samples,
                "Retrieval_Attribution": res.retrieval_attribution,
                "Generation_Attribution": res.generation_attribution,
                "Mean_Token_F1": res.mean_token_f1,
                "Mean_Retrieval_Precision": res.mean_retrieval_precision,
                "Mean_Faithfulness": res.mean_faithfulness
            })
        return pd.DataFrame(records)

    def sweep_retrieval_strategies(
        self,
        benchmark_data: List[Dict[str, Any]],
        top_k: int = 3,
        force_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Compares Dense Semantic vs Sparse BM25 vs Hybrid RRF.
        """
        records = []
        strategies = ["dense", "sparse", "hybrid"]
        for strat in strategies:
            res = self.run_benchmark_evaluation(
                benchmark_data,
                top_k=top_k,
                strategy=strat,
                force_fallback=force_fallback
            )
            records.append({
                "Strategy": strat.capitalize(),
                "Total_Error_Rate": res.total_error_rate,
                "Retrieval_Failures": res.retrieval_failure_count,
                "Generation_Failures": res.generation_failure_count,
                "Success_Count": res.success_count,
                "Parametric_Recall_Count": res.parametric_success_count,
                "Retrieval_Attribution": res.retrieval_attribution,
                "Generation_Attribution": res.generation_attribution,
                "Mean_Token_F1": res.mean_token_f1
            })
        return pd.DataFrame(records)

    def stratify_by_difficulty(self, result: DecompositionResult) -> pd.DataFrame:
        """Breaks down error decomposition by query difficulty."""
        data = []
        for s in result.samples:
            data.append({
                "difficulty": s.difficulty,
                "quadrant": s.quadrant,
                "is_error": s.quadrant in ["RETRIEVAL_FAILURE", "GENERATION_FAILURE"],
                "ret_fail": s.quadrant == "RETRIEVAL_FAILURE",
                "gen_fail": s.quadrant == "GENERATION_FAILURE",
                "f1": s.token_f1
            })
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby("difficulty").agg(
            total_samples=("quadrant", "count"),
            error_count=("is_error", "sum"),
            retrieval_failures=("ret_fail", "sum"),
            generation_failures=("gen_fail", "sum"),
            mean_f1=("f1", "mean")
        ).reset_index()

        grouped["error_rate"] = grouped["error_count"] / grouped["total_samples"]
        return grouped

    def stratify_by_question_type(self, result: DecompositionResult) -> pd.DataFrame:
        """Breaks down error decomposition by question type."""
        data = []
        for s in result.samples:
            data.append({
                "question_type": s.question_type,
                "quadrant": s.quadrant,
                "is_error": s.quadrant in ["RETRIEVAL_FAILURE", "GENERATION_FAILURE"],
                "ret_fail": s.quadrant == "RETRIEVAL_FAILURE",
                "gen_fail": s.quadrant == "GENERATION_FAILURE",
                "f1": s.token_f1
            })
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby("question_type").agg(
            total_samples=("quadrant", "count"),
            error_count=("is_error", "sum"),
            retrieval_failures=("ret_fail", "sum"),
            generation_failures=("gen_fail", "sum"),
            mean_f1=("f1", "mean")
        ).reset_index()

        grouped["error_rate"] = grouped["error_count"] / grouped["total_samples"]
        return grouped

    def compare_models(
        self,
        benchmark_data: List[Dict[str, Any]],
        models: List[str],
        top_k: int = 3,
        strategy: str = "hybrid",
        prompt_style: str = "grounded",
        force_fallback: bool = False,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluates multiple generation models under IDENTICAL retrieved contexts.
        Answers: 'When retrieval is held constant, does model capability alter
        generation failure frequency and failure types?'
        """
        model_samples: Dict[str, List] = {m: [] for m in models}
        total_steps = len(benchmark_data) * len(models)
        current_step = 0

        # Step 1: Evaluate each query. Retrieval is executed ONCE per query and frozen.
        for item in benchmark_data:
            query = item["question"]

            # Freeze retrieval context for this question across all models
            retrieved_chunks = [c for c, _ in self.retriever.retrieve(
                query, top_k=top_k, strategy=strategy
            )]

            for m in models:
                use_fallback = force_fallback or (m == "local-fallback-extractor")
                gen_output = self.llm_client.generate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    prompt_style=prompt_style,
                    model_name=m,
                    force_fallback=use_fallback
                )

                sample = self.decomposer.evaluate_sample(
                    sample_meta=item,
                    retrieved_chunks=retrieved_chunks,
                    generation_output=gen_output
                )
                model_samples[m].append(sample)

                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps)

        # Step 2: Aggregate results per model
        model_results: Dict[str, DecompositionResult] = {}
        summary_records = []
        all_subtypes = set()

        for m in models:
            res = self.decomposer.aggregate_results(model_samples[m])
            model_results[m] = res
            all_subtypes.update(res.subtype_breakdown.keys())

            rec = {
                "Model": m,
                "Total_Queries": res.total_samples,
                "Success_Rate": res.success_count / max(1, res.total_samples),
                "Generation_Failure_Rate": res.generation_failure_count / max(1, res.total_samples),
                "Retrieval_Failure_Rate": res.retrieval_failure_count / max(1, res.total_samples),
                "Generation_Attribution": res.generation_attribution,
                "Mean_Token_F1": res.mean_token_f1,
                "Mean_Semantic_Sim": res.mean_semantic_similarity,
                "Mean_Faithfulness": res.mean_faithfulness,
                "Avg_Latency_ms": res.mean_generation_time_ms
            }
            summary_records.append(rec)

        summary_df = pd.DataFrame(summary_records)

        # Step 3: Subtype distribution matrix (models x failure subtypes)
        subtype_records = []
        for m in models:
            row = {"Model": m}
            for st_name in sorted(all_subtypes):
                row[st_name] = model_results[m].subtype_breakdown.get(st_name, 0)
            subtype_records.append(row)
        subtype_df = pd.DataFrame(subtype_records)

        # Step 4: Differential Discordance Analysis (Where Model A != Model B)
        discordant_cases = []
        if len(models) >= 2:
            m1, m2 = models[0], models[1]
            s1_list = model_samples[m1]
            s2_list = model_samples[m2]

            for s1, s2 in zip(s1_list, s2_list):
                # When retrieval was successful (evidence was in context), compare model generation
                if s1.retrieval_success:
                    if s1.generation_success and not s2.generation_success:
                        discordant_cases.append({
                            "sample_id": s1.sample_id,
                            "question": s1.question,
                            "gold_answer": s1.gold_answer,
                            "winning_model": m1,
                            "failing_model": m2,
                            "winner_answer": s1.generated_answer,
                            "winner_f1": s1.token_f1,
                            "failure_answer": s2.generated_answer,
                            "failure_subtype": s2.failure_subtype,
                            "failure_explanation": s2.diagnostic_explanation,
                            "divergence_type": f"{m1} Won / {m2} Failed"
                        })
                    elif not s1.generation_success and s2.generation_success:
                        discordant_cases.append({
                            "sample_id": s1.sample_id,
                            "question": s1.question,
                            "gold_answer": s1.gold_answer,
                            "winning_model": m2,
                            "failing_model": m1,
                            "winner_answer": s2.generated_answer,
                            "winner_f1": s2.token_f1,
                            "failure_answer": s1.generated_answer,
                            "failure_subtype": s1.failure_subtype,
                            "failure_explanation": s1.diagnostic_explanation,
                            "divergence_type": f"{m2} Won / {m1} Failed"
                        })
                    elif not s1.generation_success and not s2.generation_success:
                        if s1.failure_subtype != s2.failure_subtype:
                            discordant_cases.append({
                                "sample_id": s1.sample_id,
                                "question": s1.question,
                                "gold_answer": s1.gold_answer,
                                "winning_model": "None (Both Failed)",
                                "failing_model": f"{m1} & {m2}",
                                "winner_answer": f"[{m1}]: {s1.generated_answer}",
                                "winner_f1": s1.token_f1,
                                "failure_answer": f"[{m2}]: {s2.generated_answer}",
                                "failure_subtype": f"{s1.failure_subtype} vs {s2.failure_subtype}",
                                "failure_explanation": f"Subtype shift: {m1} suffered {s1.failure_subtype}, while {m2} suffered {s2.failure_subtype}.",
                                "divergence_type": "Subtype Divergence"
                            })

        return {
            "summary_df": summary_df,
            "subtype_df": subtype_df,
            "model_results": model_results,
            "model_samples": model_samples,
            "discordant_cases": discordant_cases
        }
