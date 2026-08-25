import axios from "axios";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const api = {
  // Health
  getHealth: () => apiClient.get("/health"),

  // Pipelines
  getPipelines: () => apiClient.get("/pipelines"),
  getPipeline: (id: string) => apiClient.get(`/pipelines/${id}`),
  createPipeline: (config: any) => apiClient.post("/pipelines", config),
  updatePipeline: (id: string, config: any) => apiClient.patch(`/pipelines/${id}`, config),
  deletePipeline: (id: string) => apiClient.delete(`/pipelines/${id}`),
  getPipelineAnalytics: (id: string) => apiClient.get(`/pipelines/${id}/analytics`),
  getPipelineSuggestions: (id: string) => apiClient.get(`/pipelines/${id}/suggestions`),
  getPipelineRuns: (id: string, page = 1, pageSize = 20) =>
    apiClient.get(`/pipelines/${id}/runs?page=${page}&page_size=${pageSize}`),

  // Query & Compare
  submitQuery: (data: { query: string; pipeline_id?: string; stream?: boolean; max_tokens?: number }) =>
    apiClient.post("/query", data),
  comparePipelines: (data: { query: string; pipeline_a_id: string; pipeline_b_id: string }) =>
    apiClient.post("/pipelines/compare", data),
  submitRating: (runId: string, rating: number) =>
    apiClient.patch(`/runs/${runId}/rating`, { rating }),

  // Evaluations
  getEvaluations: (params?: { pipeline_id?: string; min_overall_score?: number; min_faithfulness?: number }) =>
    apiClient.get("/evaluations", { params }),

  // Ingestion & Documents
  uploadDocument: (formData: FormData) =>
    apiClient.post("/ingest", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  getDocuments: () => apiClient.get("/documents"),
  getDocumentChunks: (docId: string) => apiClient.get(`/documents/${docId}/chunks`),

  // Fine-tuning
  getFinetuneJobs: () => apiClient.get("/finetune/jobs"),
  getTrainingPreview: () => apiClient.get("/finetune/training-data/preview"),
  triggerFinetune: (data: any) => apiClient.post("/finetune/jobs", data),
};
