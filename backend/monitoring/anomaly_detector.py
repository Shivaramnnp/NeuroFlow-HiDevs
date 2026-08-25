from __future__ import annotations

import logging
import math
import statistics
import time
import uuid
from typing import Any, Dict, List, Optional

import asyncpg

try:
    from backend.db.pool import get_pool
    from backend.services.pipeline_optimizer import PipelineOptimizer
except ImportError:
    from db.pool import get_pool
    from services.pipeline_optimizer import PipelineOptimizer

logger = logging.getLogger("neuroflow-anomaly-detector")


class QualityAnomalyDetector:
    """
    Monitors rolling evaluation quality scores for all active pipelines:
    - Calculates 7-day rolling mean (mu) and standard deviation (sigma) of overall_score.
    - Flags an anomaly if latest_score < (mu - 2 * sigma).
    - Automatically generates optimizer suggestions when an anomaly is detected.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool
        self.optimizer = PipelineOptimizer()

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def check_pipeline_quality(self, pipeline_id: uuid.UUID) -> Dict[str, Any]:
        """
        Check if the latest evaluation score for pipeline_id is anomalous (> 2 std devs below 7-day mean).
        """
        pool = await self._get_db_pool()
        if pool is None:
            return {
                "pipeline_id": str(pipeline_id),
                "is_anomalous": False,
                "reason": "Database pool unavailable",
            }

        async with pool.acquire() as conn:
            # 1. Fetch 7-day historical evaluation scores for pipeline
            rows = await conn.fetch(
                """
                SELECT e.overall_score
                FROM evaluations e
                JOIN pipeline_runs pr ON pr.id = e.run_id
                WHERE pr.pipeline_id = $1
                  AND e.evaluated_at >= NOW() - INTERVAL '7 days'
                ORDER BY e.evaluated_at ASC;
                """,
                pipeline_id,
            )

            scores = [float(r["overall_score"]) for r in rows if r["overall_score"] is not None]

            if len(scores) < 5:
                return {
                    "pipeline_id": str(pipeline_id),
                    "is_anomalous": False,
                    "reason": "Insufficient historical evaluation samples (< 5)",
                    "sample_count": len(scores),
                }

            latest_score = scores[-1]
            history_scores = scores[:-1]

            mean = statistics.mean(history_scores)
            stdev = statistics.stdev(history_scores) if len(history_scores) > 1 else 0.05

            threshold_2sigma = mean - (2.0 * stdev)
            is_anomaly = latest_score < threshold_2sigma

            suggestions = []
            if is_anomaly:
                logger.warning(
                    f"QUALITY ANOMALY DETECTED for pipeline {pipeline_id}: "
                    f"latest={latest_score:.4f} < threshold={threshold_2sigma:.4f} (mean={mean:.4f}, std={stdev:.4f})"
                )
                try:
                    sug_res = await self.optimizer.generate_suggestions(pipeline_id)
                    suggestions = sug_res.get("suggestions", [])
                except Exception as err:
                    logger.warning(f"Could not generate suggestions for anomalous pipeline {pipeline_id}: {err}")

            return {
                "pipeline_id": str(pipeline_id),
                "latest_score": latest_score,
                "mean_7d": round(mean, 4),
                "stdev_7d": round(stdev, 4),
                "threshold_2sigma": round(threshold_2sigma, 4),
                "is_anomalous": is_anomaly,
                "suggestions": suggestions,
            }
