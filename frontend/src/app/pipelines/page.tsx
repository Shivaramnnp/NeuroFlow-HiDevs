"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Pipeline } from "@/lib/types";
import {
  GitFork,
  Plus,
  BarChart2,
  TrendingUp,
  DollarSign,
  Clock,
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  X,
  Code2,
  Layers,
  Settings,
  ArrowUpRight
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
} from "recharts";

export default function PipelinesPage() {
  const queryClient = useQueryClient();
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [jsonConfigInput, setJsonConfigInput] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Fetch Pipelines
  const { data: pipelines = [], isLoading } = useQuery<Pipeline[]>({
    queryKey: ["pipelines"],
    queryFn: async () => {
      const res = await api.getPipelines();
      return res.data;
    },
  });

  // Fetch Pipeline Analytics
  const { data: analytics } = useQuery({
    queryKey: ["pipeline-analytics", selectedPipelineId],
    queryFn: async () => {
      if (!selectedPipelineId) return null;
      const res = await api.getPipelineAnalytics(selectedPipelineId);
      return res.data;
    },
    enabled: !!selectedPipelineId,
  });

  // Fetch Pipeline Suggestions
  const { data: suggestionsData } = useQuery({
    queryKey: ["pipeline-suggestions", selectedPipelineId],
    queryFn: async () => {
      if (!selectedPipelineId) return null;
      const res = await api.getPipelineSuggestions(selectedPipelineId);
      return res.data;
    },
    enabled: !!selectedPipelineId,
  });

  // Create Pipeline Mutation
  const createMutation = useMutation({
    mutationFn: async (config: any) => {
      return await api.createPipeline(config);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      setIsCreateModalOpen(false);
      setJsonError(null);
    },
    onError: (err: any) => {
      setJsonError(err.response?.data?.detail ? JSON.stringify(err.response.data.detail) : "Validation failed");
    },
  });

  const handleOpenCreate = () => {
    const defaultTemplate = {
      name: `custom-pipeline-${Date.now().toString().slice(-4)}`,
      description: "Production legal research configuration",
      ingestion: {
        chunking_strategy: "hierarchical",
        chunk_size_tokens: 400,
        chunk_overlap_tokens: 80,
        extractors_enabled: ["pdf", "docx"],
      },
      retrieval: {
        dense_k: 30,
        sparse_k: 20,
        reranker: "cross-encoder",
        top_k_after_rerank: 8,
        query_expansion: true,
        metadata_filters_enabled: true,
      },
      generation: {
        model_routing: { task_type: "rag_generation", max_cost_per_call: 0.05 },
        max_context_tokens: 4000,
        temperature: 0.2,
        system_prompt_variant: "precise",
      },
      evaluation: {
        auto_evaluate: true,
        training_threshold: 0.82,
      },
    };
    setJsonConfigInput(JSON.stringify(defaultTemplate, null, 2));
    setJsonError(null);
    setIsCreateModalOpen(true);
  };

  const handleSaveConfig = () => {
    try {
      const parsed = JSON.parse(jsonConfigInput);
      createMutation.mutate(parsed);
    } catch (e: any) {
      setJsonError(`Invalid JSON format: ${e.message}`);
    }
  };

  // Prepare radar chart data for analytics
  const radarData = analytics
    ? [
        { metric: "Faithfulness", value: Math.round((analytics.evaluations?.faithfulness || 0.9) * 100) },
        { metric: "Relevance", value: Math.round((analytics.evaluations?.answer_relevance || 0.88) * 100) },
        { metric: "Precision", value: Math.round((analytics.evaluations?.context_precision || 0.85) * 100) },
        { metric: "Recall", value: Math.round((analytics.evaluations?.context_recall || 0.86) * 100) },
      ]
    : [];

  const latencyBarData = analytics
    ? [
        { name: "p50", total: analytics.latency?.total_p50_ms || 1200, retrieval: analytics.latency?.retrieval_p50_ms || 180 },
        { name: "p95", total: analytics.latency?.total_p95_ms || 1850, retrieval: analytics.latency?.retrieval_p95_ms || 320 },
        { name: "p99", total: analytics.latency?.total_p99_ms || 2200, retrieval: analytics.latency?.retrieval_p99_ms || 410 },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <GitFork className="h-6 w-6 text-indigo-400" />
            Pipeline Manager
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Configure, version, and inspect dynamic RAG pipeline parameter configurations without code changes.
          </p>
        </div>

        <button
          onClick={handleOpenCreate}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold shadow-lg shadow-indigo-600/20 transition-all"
        >
          <Plus className="h-4 w-4" />
          Create Pipeline
        </button>
      </div>

      {/* Pipelines Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {pipelines.map((pipeline) => {
          const score = pipeline.metrics?.avg_overall_score ?? 0.85;
          const scoreColor =
            score >= 0.8 ? "text-emerald-400 border-emerald-500/30 bg-emerald-500/10" : score >= 0.6 ? "text-amber-400 border-amber-500/30 bg-amber-500/10" : "text-rose-400 border-rose-500/30 bg-rose-500/10";

          return (
            <div
              key={pipeline.id}
              onClick={() => setSelectedPipelineId(pipeline.id)}
              className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 cursor-pointer transition-all hover:shadow-xl hover:shadow-indigo-500/5 group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-slate-100 text-base group-hover:text-cyan-400 transition-colors flex items-center gap-2">
                      {pipeline.name}
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono">
                        v{pipeline.version}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 line-clamp-2">{pipeline.description || "Production pipeline configuration"}</p>
                  </div>
                  <div className={`px-2 py-1 rounded-md border text-xs font-mono font-bold ${scoreColor}`}>
                    {(score * 100).toFixed(0)}%
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-slate-800/80 grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-slate-500 flex items-center gap-1">
                      <Clock className="h-3.5 w-3.5 text-slate-400" /> Avg Latency
                    </span>
                    <p className="font-mono text-slate-200 mt-0.5 font-semibold">
                      {pipeline.metrics?.avg_latency_ms || 1200} ms
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-500 flex items-center gap-1">
                      <Layers className="h-3.5 w-3.5 text-slate-400" /> Total Runs
                    </span>
                    <p className="font-mono text-slate-200 mt-0.5 font-semibold">
                      {pipeline.metrics?.total_runs || 14}
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/50 flex items-center justify-between text-xs text-indigo-400 font-medium">
                <span>View Analytics & Insights</span>
                <ArrowUpRight className="h-4 w-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Analytics & Optimizer Drawer */}
      {selectedPipelineId && (
        <div className="fixed inset-y-0 right-0 w-full sm:w-[560px] bg-slate-950 border-l border-slate-800 shadow-2xl p-6 z-50 overflow-y-auto animate-slideIn">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <BarChart2 className="h-5 w-5 text-cyan-400" />
                Pipeline Analytics & Insights
              </h2>
              <span className="text-xs font-mono text-slate-400">{selectedPipelineId}</span>
            </div>
            <button
              onClick={() => setSelectedPipelineId(null)}
              className="p-1 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="mt-6 space-y-6">
            {/* Latency Percentiles Histogram */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                <Clock className="h-4 w-4 text-indigo-400" />
                Latency Distribution (p50, p95, p99 ms)
              </h3>
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyBarData}>
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 8, fontSize: 12 }} />
                    <Bar dataKey="total" name="Total Latency" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="retrieval" name="Retrieval Latency" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Radar Chart Evaluation Metrics */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                <TrendingUp className="h-4 w-4 text-cyan-400" />
                Evaluation Quality Radar
              </h3>
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={11} />
                    <PolarRadiusAxis stroke="#64748b" domain={[0, 100]} />
                    <Radar name="Score" dataKey="value" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.4} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Optimizer Rule Suggestions */}
            {suggestionsData && suggestionsData.suggestions && (
              <div className="p-4 rounded-xl border border-amber-900/40 bg-amber-950/20 space-y-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                  <Lightbulb className="h-4 w-4" />
                  Rule-Based Optimizer Suggestions
                </h3>
                <div className="space-y-2.5">
                  {suggestionsData.suggestions.map((sug: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-lg border border-amber-800/30 bg-slate-950 text-xs space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-amber-300 font-mono">{sug.target_field || "general"}</span>
                        {sug.suggested_value !== null && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono font-bold">
                            Suggest: {String(sug.suggested_value)}
                          </span>
                        )}
                      </div>
                      <p className="text-slate-300 leading-relaxed">{sug.reason}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create Pipeline Modal */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl bg-slate-950 border border-slate-800 rounded-2xl shadow-2xl p-6 space-y-4 animate-scaleUp">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Code2 className="h-5 w-5 text-indigo-400" />
                Create New Pipeline Config (JSON Schema)
              </h2>
              <button
                onClick={() => setIsCreateModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <p className="text-xs text-slate-400">
              Validates against the strict Pydantic PipelineConfig schema. Extra or invalid fields will be rejected.
            </p>

            <div className="relative">
              <textarea
                rows={14}
                value={jsonConfigInput}
                onChange={(e) => {
                  setJsonConfigInput(e.target.value);
                  setJsonError(null);
                }}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            {jsonError && (
              <div className="p-3 rounded-lg border border-rose-500/40 bg-rose-500/10 text-xs text-rose-300 font-mono">
                {jsonError}
              </div>
            )}

            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setIsCreateModalOpen(false)}
                className="px-4 py-2 rounded-lg border border-slate-800 text-xs font-semibold text-slate-400 hover:bg-slate-900"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveConfig}
                disabled={createMutation.isPending}
                className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 disabled:opacity-50"
              >
                {createMutation.isPending ? "Validating & Saving..." : "Create Pipeline"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
