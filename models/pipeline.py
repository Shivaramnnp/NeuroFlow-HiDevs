from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking_strategy: Literal["fixed_size", "semantic", "hierarchical", "auto"] = Field(
        default="fixed_size",
        description="Chunking algorithm to apply",
    )
    chunk_size_tokens: int = Field(default=512, ge=64, le=4096, description="Chunk token size")
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512, description="Overlap tokens between adjacent chunks")
    extractors_enabled: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "image", "csv", "url", "pptx"],
        description="List of enabled extractor types",
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_k: int = Field(default=20, ge=1, le=100, description="Top-k dense vector candidates")
    sparse_k: int = Field(default=20, ge=1, le=100, description="Top-k BM25/sparse candidates")
    reranker: Literal["cross-encoder", "none", "local", "api"] = Field(
        default="cross-encoder",
        description="Reranking strategy",
    )
    top_k_after_rerank: int = Field(default=10, ge=1, le=50, description="Top candidates retained after reranking")
    query_expansion: bool = Field(default=True, description="Enable query expansion (2-3 phrasings)")
    metadata_filters_enabled: bool = Field(default=True, description="Enable metadata filter extraction")


class ModelRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(default="rag_generation", description="Task type for model routing")
    max_cost_per_call: Optional[float] = Field(default=0.05, ge=0.0, description="Max USD cost ceiling per call")


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_routing: ModelRoutingConfig = Field(
        default_factory=ModelRoutingConfig,
        description="Model router criteria",
    )
    max_context_tokens: int = Field(default=4000, ge=500, le=32000, description="Context budget limit in tokens")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    system_prompt_variant: Literal["precise", "factual", "analytical", "comparative", "procedural"] = Field(
        default="precise",
        description="System prompt style directive",
    )


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_evaluate: bool = Field(default=True, description="Automatically trigger RAGAS evaluation on complete")
    training_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Score threshold above which run is saved to training_pairs",
    )


class PipelineConfig(BaseModel):
    """
    Validated, strict configuration schema for a NeuroFlow RAG Pipeline.
    Rejects any unknown keys to prevent configuration corruption.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, description="Unique pipeline identifier name")
    description: Optional[str] = Field(default="", description="Detailed human description of pipeline purpose")
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
