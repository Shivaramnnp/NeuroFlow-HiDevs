from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

try:
    from backend.db.pool import get_pool
    from backend.pipelines.retrieval.context_assembler import count_tokens
except ImportError:
    from db.pool import get_pool
    from pipelines.retrieval.context_assembler import count_tokens

logger = logging.getLogger("neuroflow-extractor")

# Regex for PII detection: Emails and standard phone numbers
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CITATION_REGEX = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)


@dataclass
class ExtractedTrainingPair:
    id: str
    run_id: str
    system_prompt: str
    user_message: str
    assistant_message: str
    quality_score: float
    token_count: int
    created_at: Optional[str] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Format as OpenAI fine-tuning JSONL message schema."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_message},
                {"role": "assistant", "content": self.assistant_message},
            ]
        }

    def to_dpo_format(self, rejected_message: Optional[str] = None) -> Dict[str, Any]:
        """Format as Direct Preference Optimization (DPO) preference pair."""
        return {
            "prompt": self.user_message,
            "chosen": self.assistant_message,
            "rejected": rejected_message or "I am sorry, but I do not have enough context to answer.",
        }


class TrainingDataExtractor:
    """
    Queries, validates, filters, and formats candidate training pairs
    from PostgreSQL for fine-tuning dataset export.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    @staticmethod
    def validate_pair(
        user_message: str,
        assistant_message: str,
        quality_score: float,
        min_tokens: int = 50,
        max_tokens: int = 2000,
        min_faithfulness: float = 0.80,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate training pair against PII, token budget, citation presence, and quality rules:
        - Rejects if user query contains email or phone number
        - Rejects if assistant token count < 50 or > 2000
        - Rejects if no [Source N] citation present
        - Rejects if quality_score < min_faithfulness
        """
        # Rule 1: PII check in query
        if EMAIL_REGEX.search(user_message):
            return False, "Query contains email PII pattern"
        if PHONE_REGEX.search(user_message):
            return False, "Query contains phone number PII pattern"

        # Rule 2: Token length check
        tok_count = count_tokens(assistant_message)
        if tok_count < min_tokens:
            return False, f"Assistant message token count ({tok_count}) is below minimum {min_tokens}"
        if tok_count > max_tokens:
            return False, f"Assistant message token count ({tok_count}) exceeds maximum {max_tokens}"

        # Rule 3: Citation presence
        if not CITATION_REGEX.search(assistant_message):
            return False, "Assistant message lacks required [Source N] citation"

        # Rule 4: Quality / Faithfulness threshold
        if quality_score < min_faithfulness:
            return False, f"Quality score ({quality_score}) is below threshold {min_faithfulness}"

        return True, None

    async def extract_and_export(
        self,
        job_id: uuid.UUID,
        min_quality_score: float = 0.82,
        output_dir: str = "training_data",
        format_type: str = "sft",  # "sft" or "dpo"
    ) -> Dict[str, Any]:
        """
        Extract eligible training pairs from DB, apply validation filters,
        write to training_data/{job_id}.jsonl, and mark included_in_job.
        """
        pool = await self._get_db_pool()
        raw_pairs = []

        if pool is not None:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT tp.id, tp.run_id, tp.system_prompt, tp.user_message,
                           tp.assistant_message, tp.quality_score, tp.created_at,
                           e.user_rating
                    FROM training_pairs tp
                    JOIN pipeline_runs pr ON pr.id = tp.run_id
                    LEFT JOIN evaluations e ON e.run_id = pr.id
                    WHERE tp.quality_score >= $1
                      AND tp.included_in_job IS NULL
                      AND (e.user_rating >= 4 OR e.user_rating IS NULL)
                    ORDER BY tp.quality_score DESC;
                    """,
                    min_quality_score,
                )
                raw_pairs = rows
        else:
            # Fallback mock pairs for local sandbox testing
            raw_pairs = [
                {
                    "id": uuid.uuid4(),
                    "run_id": uuid.uuid4(),
                    "system_prompt": "You are a precise research assistant.",
                    "user_message": "What is HNSW indexing in NeuroFlow?",
                    "assistant_message": "HNSW (Hierarchical Navigable Small World) is a graph-based vector index [Source 1] that provides logarithmic search complexity [Source 2] for sub-millisecond approximate nearest neighbor lookups in pgvector.",
                    "quality_score": 0.95,
                    "created_at": None,
                }
            ]

        validated_pairs: List[ExtractedTrainingPair] = []
        rejected_count = 0
        pair_ids_to_update: List[uuid.UUID] = []

        for row in raw_pairs:
            u_msg = row["user_message"]
            a_msg = row["assistant_message"]
            q_score = float(row["quality_score"])

            is_valid, reason = self.validate_pair(u_msg, a_msg, q_score)
            if is_valid:
                p_id = row["id"] if isinstance(row["id"], uuid.UUID) else uuid.UUID(str(row["id"]))
                pair_ids_to_update.append(p_id)
                validated_pairs.append(
                    ExtractedTrainingPair(
                        id=str(p_id),
                        run_id=str(row["run_id"]),
                        system_prompt=row["system_prompt"] or "You are a precise research assistant.",
                        user_message=u_msg,
                        assistant_message=a_msg,
                        quality_score=q_score,
                        token_count=count_tokens(a_msg),
                        created_at=str(row["created_at"]) if row.get("created_at") else None,
                    )
                )
            else:
                logger.info(f"Rejected training pair {row['id']}: {reason}")
                rejected_count += 1

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{job_id}.jsonl")

        with open(file_path, "w", encoding="utf-8") as f:
            for pair in validated_pairs:
                if format_type.lower() == "dpo":
                    line = json.dumps(pair.to_dpo_format())
                else:
                    line = json.dumps(pair.to_openai_format())
                f.write(line + "\n")

        # Mark extracted pairs in DB
        if pool is not None and pair_ids_to_update:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE training_pairs
                    SET included_in_job = $1
                    WHERE id = ANY($2::uuid[]);
                    """,
                    job_id,
                    pair_ids_to_update,
                )

        return {
            "job_id": str(job_id),
            "file_path": file_path,
            "total_extracted": len(raw_pairs),
            "total_validated": len(validated_pairs),
            "total_rejected": rejected_count,
            "pairs": validated_pairs,
        }

    async def preview_samples(self, limit: int = 5, min_quality_score: float = 0.82) -> List[Dict[str, Any]]:
        """
        Preview candidate training pairs without submitting a job or modifying DB state.
        """
        pool = await self._get_db_pool()
        if pool is None:
            return [
                {
                    "system_prompt": "You are a precise research assistant.",
                    "user_message": "Explain pgvector HNSW indexing",
                    "assistant_message": "pgvector supports HNSW indexing for rapid nearest neighbor search [Source 1].",
                    "quality_score": 0.92,
                    "validation_status": "valid",
                }
            ]

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tp.id, tp.system_prompt, tp.user_message, tp.assistant_message, tp.quality_score
                FROM training_pairs tp
                JOIN pipeline_runs pr ON pr.id = tp.run_id
                LEFT JOIN evaluations e ON e.run_id = pr.id
                WHERE tp.quality_score >= $1
                  AND tp.included_in_job IS NULL
                  AND (e.user_rating >= 4 OR e.user_rating IS NULL)
                ORDER BY tp.quality_score DESC
                LIMIT $2;
                """,
                min_quality_score,
                limit,
            )

            previews = []
            for r in rows:
                is_valid, reason = self.validate_pair(r["user_message"], r["assistant_message"], float(r["quality_score"]))
                previews.append(
                    {
                        "id": str(r["id"]),
                        "system_prompt": r["system_prompt"],
                        "user_message": r["user_message"],
                        "assistant_message": r["assistant_message"],
                        "quality_score": float(r["quality_score"]),
                        "is_valid": is_valid,
                        "validation_reason": reason or "valid",
                    }
                )
            return previews
