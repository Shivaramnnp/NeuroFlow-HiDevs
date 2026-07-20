class RoutingCriteria:
    task_type: str          # "rag_generation" | "evaluation" | "embedding" | "classification"
    max_cost_per_call: float | None
    require_vision: bool
    require_long_context: bool  # > 32k tokens
    latency_budget_ms: int | None
    prefer_fine_tuned: bool