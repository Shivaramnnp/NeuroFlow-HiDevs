from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import asyncpg
import numpy as np
# pyrefly: ignore [missing-import]
from opentelemetry import trace

from .metrics.answer_relevance import evaluate_answer_relevance
from .metrics.context_precision import evaluate_context_precision
from .metrics.context_recall import evaluate_context_recall
from .metrics.faithfulness import evaluate_faithfulness

try:
    from backend.db.pool import get_pool
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from db.pool import get_pool
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-judge")
tracer = trace.get_tracer("neuroflow-evaluation")


class EvaluationJudge:
    """
    Automated LLM-as-a-Judge for RAG generations:
    - Runs 4 RAGAS metrics in parallel via asyncio.gather:
      1. Faithfulness (weight: 0.35)
      2. Answer Relevance (weight: 0.30)
      3. Context Precision (weight: 0.20)
      4. Context Recall (weight: 0.15)
    - Persists evaluation results into PostgreSQL evaluations table
    - Automatically extracts high-quality training pairs (overall_score > 0.8) into training_pairs table
    - Routes exclusively to capable judge models (never fine-tuned)
    - Emits OpenTelemetry span 'evaluation.judge'
    - Supports self-consistency multi-sampling with variance tracking
    """

    def __init__(
        self,
        client: Optional[NeuroFlowClient] = None,
        pool: Optional[asyncpg.Pool] = None,
    ):
        self.client = client or NeuroFlowClient.get_instance()
        self.pool = pool

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        chunks: Optional[List[str]] = None,
        run_id: Optional[Union[uuid.UUID, str]] = None,
        system_prompt: Optional[str] = None,
        judge_model: str = "gpt-4o",
        self_consistency: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive evaluation of a RAG query/answer/context generation.
        """
        active_run_id = uuid.UUID(str(run_id)) if run_id else uuid.uuid4()
        chunk_list = chunks or ([c.strip() for c in context.split("\n\n") if c.strip()] if context else [])

        start_time = time.perf_counter()

        with tracer.start_as_current_span("evaluation.judge") as span:
            span.set_attribute("run_id", str(active_run_id))

            if not self_consistency:
                # 1. Parallel execution of all 4 metrics
                f_task = evaluate_faithfulness(query, answer, context, client=self.client)
                ar_task = evaluate_answer_relevance(query, answer, client=self.client)
                cp_task = evaluate_context_precision(query, chunk_list, answer, client=self.client)
                cr_task = evaluate_context_recall(query, chunk_list, answer, client=self.client)

                faithfulness, answer_relevance, context_precision, context_recall = await asyncio.gather(
                    f_task, ar_task, cp_task, cr_task
                )

                # 2. Weighted overall score
                overall_score = round(
                    0.35 * faithfulness + 0.30 * answer_relevance + 0.20 * context_precision + 0.15 * context_recall,
                    4,
                )
                variance_info = {}
            else:
                # Self-consistency: 3 independent evaluations
                samples = []
                for _ in range(3):
                    f, ar, cp, cr = await asyncio.gather(
                        evaluate_faithfulness(query, answer, context, client=self.client),
                        evaluate_answer_relevance(query, answer, client=self.client),
                        evaluate_context_precision(query, chunk_list, answer, client=self.client),
                        evaluate_context_recall(query, chunk_list, answer, client=self.client),
                    )
                    s_score = 0.35 * f + 0.30 * ar + 0.20 * cp + 0.15 * cr
                    samples.append((f, ar, cp, cr, s_score))

                faithfulness = round(float(np.mean([s[0] for s in samples])), 4)
                answer_relevance = round(float(np.mean([s[1] for s in samples])), 4)
                context_precision = round(float(np.mean([s[2] for s in samples])), 4)
                context_recall = round(float(np.mean([s[3] for s in samples])), 4)
                overall_score = round(float(np.mean([s[4] for s in samples])), 4)

                stdev = float(np.std([s[4] for s in samples]))
                variance_info = {
                    "self_consistency_runs": 3,
                    "score_std": round(stdev, 4),
                    "high_variance": stdev > 0.20,
                }

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Record OpenTelemetry Span Attributes
            span.set_attribute("faithfulness", faithfulness)
            span.set_attribute("answer_relevance", answer_relevance)
            span.set_attribute("context_precision", context_precision)
            span.set_attribute("context_recall", context_recall)
            span.set_attribute("overall_score", overall_score)
            span.set_attribute("judge_model", judge_model)
            span.set_attribute("latency_ms", latency_ms)

            # 3. Write evaluation record to PostgreSQL evaluations table
            eval_id = uuid.uuid4()
            pool = await self._get_db_pool()
            if pool is not None:
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                """
                                INSERT INTO evaluations (id, run_id, faithfulness, answer_relevance, context_precision, context_recall, overall_score, judge_model)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
                                """,
                                eval_id,
                                active_run_id,
                                faithfulness,
                                answer_relevance,
                                context_precision,
                                context_recall,
                                overall_score,
                                judge_model,
                            )

                            # 4. If overall_score > 0.8, extract high-quality pair for fine-tuning
                            if overall_score > 0.8:
                                tp_id = uuid.uuid4()
                                await conn.execute(
                                    """
                                    INSERT INTO training_pairs (id, run_id, system_prompt, user_message, assistant_message, quality_score)
                                    VALUES ($1, $2, $3, $4, $5, $6);
                                    """,
                                    tp_id,
                                    active_run_id,
                                    system_prompt or "You are a precise research assistant.",
                                    query,
                                    answer,
                                    overall_score,
                                )
                                logger.info(f"Extracted high-quality training pair for run {active_run_id} (score: {overall_score})")
                except Exception as err:
                    logger.warning(f"Failed to persist evaluation to database: {err}")

            return {
                "eval_id": str(eval_id),
                "run_id": str(active_run_id),
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "overall_score": overall_score,
                "judge_model": judge_model,
                "latency_ms": latency_ms,
                "training_pair_extracted": overall_score > 0.8,
                **variance_info,
            }
