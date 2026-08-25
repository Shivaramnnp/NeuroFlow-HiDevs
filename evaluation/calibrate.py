from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

from evaluation.metrics.faithfulness import evaluate_faithfulness
from backend.providers.client import NeuroFlowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow-calibration")


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient between two lists of float scores."""
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)

    mean_x = np.mean(arr_x)
    mean_y = np.mean(arr_y)

    num = np.sum((arr_x - mean_x) * (arr_y - mean_y))
    den = np.sqrt(np.sum((arr_x - mean_x) ** 2) * np.sum((arr_y - mean_y) ** 2))

    if den == 0:
        return 1.0
    return float(num / den)


async def run_calibration():
    dataset_path = os.path.join(os.path.dirname(__file__), "calibration", "annotated_set.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Calibration dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        annotated_set = json.load(f)

    logger.info(f"Loaded {len(annotated_set)} calibration examples from {dataset_path}")

    client = NeuroFlowClient.get_instance()
    human_scores: List[float] = []
    automated_scores: List[float] = []
    per_example_results: List[Dict[str, Any]] = []

    # Run evaluations with controlled pacing
    for item in annotated_set:
        auto_score = await evaluate_faithfulness(
            query=item["query"],
            answer=item["answer"],
            context=item["context"],
            client=client,
        )
        h_score = float(item["human_score"])
        human_scores.append(h_score)
        automated_scores.append(auto_score)

        per_example_results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "human_score": h_score,
                "automated_score": auto_score,
                "difference": round(abs(h_score - auto_score), 4),
            }
        )
        await asyncio.sleep(0.15)  # Pace requests to respect OpenRouter in-flight limit

    correlation = calculate_pearson_correlation(automated_scores, human_scores)
    mae = float(np.mean([abs(a - h) for a, h in zip(automated_scores, human_scores)]))

    summary = {
        "total_examples": len(annotated_set),
        "pearson_correlation": round(correlation, 4),
        "mean_absolute_error": round(mae, 4),
        "target_correlation_threshold": 0.85,
        "calibration_passed": correlation > 0.85,
        "results": per_example_results,
    }

    output_path = os.path.join(os.path.dirname(__file__), "calibration_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("         NEUROFLOW JUDGE CALIBRATION REPORT            ")
    print("=======================================================")
    print(f"Total Calibration Samples: {len(annotated_set)}")
    print(f"Pearson Correlation:       {correlation:.4f}  (Threshold > 0.85) -> {'PASS' if correlation > 0.85 else 'FAIL'}")
    print(f"Mean Absolute Error:       {mae:.4f}")
    print(f"Results Written to:        {output_path}")
    print("=======================================================\n")

    assert correlation > 0.85, f"Pearson correlation {correlation} is below threshold 0.85"


if __name__ == "__main__":
    asyncio.run(run_calibration())
