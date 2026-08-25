export interface PipelineMetrics {
  total_runs: number;
  avg_latency_ms: number;
  avg_overall_score: number;
  faithfulness?: number;
  answer_relevance?: number;
  context_precision?: number;
  context_recall?: number;
}

export interface Pipeline {
  id: string;
  name: string;
  description?: string;
  version: number;
  status: string;
  config: Record<string, any>;
  metrics?: PipelineMetrics;
  created_at?: string;
  updated_at?: string;
}

export interface Citation {
  source_number: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number?: number;
  section_title?: string;
  content: string;
}

export interface EvaluationData {
  eval_id?: string;
  run_id?: string;
  faithfulness: number;
  answer_relevance: number;
  context_precision: number;
  context_recall: number;
  overall_score: number;
  judge_model?: string;
  user_rating?: number;
  evaluated_at?: string;
}

export interface QueryRun {
  run_id: string;
  query: string;
  generation: string;
  citations: Citation[];
  sources: any[];
  latency_ms?: number;
  status: string;
  evaluation?: EvaluationData;
}

export interface CompareResult {
  query: string;
  pipeline_a: {
    run_id: string;
    pipeline_id: string;
    pipeline_name?: string;
    pipeline_version: number;
    generation: string;
    retrieval_latency_ms: number;
    total_latency_ms: number;
    chunks_used: number;
    eval_score?: number;
    citations_count: number;
  };
  pipeline_b: {
    run_id: string;
    pipeline_id: string;
    pipeline_name?: string;
    pipeline_version: number;
    generation: string;
    retrieval_latency_ms: number;
    total_latency_ms: number;
    chunks_used: number;
    eval_score?: number;
    citations_count: number;
  };
  winner?: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  source_type: string;
  status: "queued" | "processing" | "indexed" | "failed";
  chunk_count: number;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface ChunkItem {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  token_count: number;
  metadata?: Record<string, any>;
  similarity_score?: number;
}
