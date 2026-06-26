import json
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import ValidationError

from rag_scorecard.models import RAGSample
from rag_scorecard.evaluators.metrics import calculate_mrr, calculate_hit_rate
from rag_scorecard.evaluators.tracker import LatencyCostTracker
from rag_scorecard.evaluators.judge import MultiProviderEvaluatorJudge

__version__ = '0.4.6'

class AuditEngine:
    @staticmethod
    def run(dataset_path: str, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {dataset_path}")

        # Extract from rag_scorecard/__init__.py parsing section
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Extract the list if wrapped in a dictionary envelope
            if isinstance(raw_data, dict):
                if "data" in raw_data and isinstance(raw_data["data"], list):
                    raw_data = raw_data["data"]
                else:
                    raw_data = [raw_data]

            dataset = [RAGSample(**sample) for sample in raw_data]
        except ValidationError as ve:
            raise ValueError("Dataset JSON layout does not match Pydantic schema requirements.") from ve

        tracker = LatencyCostTracker()

        with tracker.track_stage("deterministic_heuristics"):
            mrr_result = calculate_mrr(dataset)
            hit_rate_result = calculate_hit_rate(dataset)

        llm_report_logs = []
        summary_scores = {"Context Relevance": [], "Faithfulness": [], "Answer Relevance": [], "Completeness": []}

        if provider:
            judge = MultiProviderEvaluatorJudge(provider=provider, model=model, tracker=tracker)
            with tracker.track_stage("llm_judge_evaluations"):
                for sample in dataset:
                    row_metrics = {}

                    c_rel = judge.evaluate_context_relevance(sample)
                    row_metrics["context_relevance"] = c_rel.model_dump()
                    summary_scores["Context Relevance"].append(c_rel.score)

                    if sample.generated_response:
                        faith = judge.evaluate_faithfulness(sample)
                        row_metrics["faithfulness"] = faith.model_dump()
                        summary_scores["Faithfulness"].append(faith.score)

                        a_rel = judge.evaluate_answer_relevance(sample)
                        row_metrics["answer_relevance"] = a_rel.model_dump()
                        summary_scores["Answer Relevance"].append(a_rel.score)

                    if sample.generated_response and sample.ground_truth:
                        comp = judge.evaluate_completeness(sample)
                        row_metrics["completeness"] = comp.model_dump()
                        summary_scores["Completeness"].append(comp.score)

                    llm_report_logs.append({"query": sample.query, "metrics": row_metrics})

        return {
            "dataset_summary": {
                "total_samples": len(dataset),
                "mrr": mrr_result.score,
                "hit_rate": hit_rate_result.score
            },
            "llm_judge_averages": {k: (sum(v)/len(v) if v else 0.0) for k, v in summary_scores.items()},
            "telemetry": {
                "duration_ms": tracker.timings.get("deterministic_heuristics", 0.0) + tracker.timings.get("llm_judge_evaluations", 0.0),
                "total_input_tokens": tracker.total_input_tokens,
                "total_output_tokens": tracker.total_output_tokens,
                "estimated_cost_usd": tracker.get_estimated_cost()
            },
            "detailed_logs": llm_report_logs
        }
