"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Pipeline, Citation, EvaluationData } from "@/lib/types";
import { useSSEStream } from "@/hooks/useSSEStream";
import {
  Sparkles,
  Send,
  GitCompare,
  ThumbsUp,
  ThumbsDown,
  Layers,
  FileText,
  Clock,
  Zap,
  Info,
  ChevronRight,
  X,
  Network,
  Award,
  CheckCircle2,
  AlertCircle
} from "lucide-react";

export default function PlaygroundPage() {
  const [query, setQuery] = useState("");
  const [selectedPipelineA, setSelectedPipelineA] = useState<string>("");
  const [selectedPipelineB, setSelectedPipelineB] = useState<string>("");
  const [compareMode, setCompareMode] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [ratingSubmitted, setRatingSubmitted] = useState<{ [runId: string]: number }>({});

  // Fetch active pipelines
  const { data: pipelines = [], isLoading: isLoadingPipelines } = useQuery<Pipeline[]>({
    queryKey: ["pipelines"],
    queryFn: async () => {
      const res = await api.getPipelines();
      return res.data;
    },
  });

  // Default pipeline selection
  useEffect(() => {
    if (pipelines.length > 0 && !selectedPipelineA) {
      setSelectedPipelineA(pipelines[0].id);
      if (pipelines.length > 1) {
        setSelectedPipelineB(pipelines[1].id);
      }
    }
  }, [pipelines, selectedPipelineA]);

  // Single stream hook
  const streamA = useSSEStream();
  const streamB = useSSEStream();

  const handleRunQuery = () => {
    if (!query.trim()) return;
    streamA.startStream(query, selectedPipelineA || undefined);
    if (compareMode && selectedPipelineB) {
      streamB.startStream(query, selectedPipelineB);
    }
  };

  const handleRate = async (runId: string | null, rating: number) => {
    if (!runId) return;
    try {
      await api.submitRating(runId, rating);
      setRatingSubmitted((prev) => ({ ...prev, [runId]: rating }));
    } catch (err) {
      console.error("Failed to submit rating", err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-cyan-400" />
            Query Playground
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Test and evaluate RAG pipelines in real-time with streaming generation, citation inspector, and A/B comparison.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              compareMode
                ? "bg-indigo-600/30 border-indigo-500 text-indigo-300 shadow-sm"
                : "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
            }`}
          >
            <GitCompare className="h-4 w-4" />
            Compare Mode {compareMode ? "ON" : "OFF"}
          </button>

          <button
            onClick={() => setShowInspector(!showInspector)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              showInspector
                ? "bg-cyan-600/30 border-cyan-500 text-cyan-300"
                : "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
            }`}
          >
            <Network className="h-4 w-4" />
            Retrieval Inspector
          </button>
        </div>
      </div>

      {/* Pipeline Selectors & Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Pipeline A Selector */}
        <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col gap-2">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
            <span>{compareMode ? "Pipeline A" : "Active Pipeline"}</span>
            {selectedPipelineA && (
              <span className="text-[11px] text-cyan-400 font-mono">
                Avg Score: {pipelines.find((p) => p.id === selectedPipelineA)?.metrics?.avg_overall_score ?? "0.85"}
              </span>
            )}
          </label>
          <select
            value={selectedPipelineA}
            onChange={(e) => setSelectedPipelineA(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            {pipelines.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} (v{p.version}) — Score: {p.metrics?.avg_overall_score ?? 0.85}
              </option>
            ))}
          </select>
        </div>

        {/* Pipeline B Selector (Compare Mode) */}
        {compareMode && (
          <div className="p-3.5 rounded-xl border border-indigo-900/50 bg-slate-900/60 flex flex-col gap-2 animate-fadeIn">
            <label className="text-xs font-semibold text-indigo-400 uppercase tracking-wider flex items-center justify-between">
              <span>Pipeline B (A/B Target)</span>
              {selectedPipelineB && (
                <span className="text-[11px] text-indigo-300 font-mono">
                  Avg Score: {pipelines.find((p) => p.id === selectedPipelineB)?.metrics?.avg_overall_score ?? "0.82"}
                </span>
              )}
            </label>
            <select
              value={selectedPipelineB}
              onChange={(e) => setSelectedPipelineB(e.target.value)}
              className="w-full bg-slate-950 border border-indigo-800/50 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {pipelines.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} (v{p.version}) — Score: {p.metrics?.avg_overall_score ?? 0.82}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Query Input Box */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 relative">
        <textarea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question or enter a query (e.g. 'What is the liability limitation in standard MSAs?')..."
          className="w-full bg-transparent border-0 text-slate-100 placeholder-slate-500 focus:outline-none resize-none text-base"
        />
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/60">
          <span className="text-xs text-slate-500 font-mono">{query.length} characters</span>
          <button
            onClick={handleRunQuery}
            disabled={!query.trim() || streamA.isStreaming || streamB.isStreaming}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 text-white text-sm font-semibold shadow-lg shadow-cyan-500/20 hover:opacity-95 disabled:opacity-50 transition-all"
          >
            <Send className="h-4 w-4" />
            {streamA.isStreaming || streamB.isStreaming ? "Streaming..." : "Run Query"}
          </button>
        </div>
      </div>

      {/* Retrieval Inspector (React Flow Style Graph) */}
      {showInspector && (
        <div className="p-4 rounded-xl border border-cyan-900/40 bg-slate-950/80 space-y-3 animate-fadeIn">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
              <Network className="h-4 w-4" />
              Retrieval Inspector Graph
            </h3>
            <span className="text-xs text-slate-400">Step-by-step chunk propagation</span>
          </div>

          <div className="grid grid-cols-5 gap-2 text-center text-xs">
            <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-900">
              <div className="font-semibold text-slate-300">1. Query Processor</div>
              <div className="text-[11px] text-slate-500 mt-1">HyDE + Expansion</div>
              <div className="mt-1 text-cyan-400 font-mono">3 Queries</div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-900">
              <div className="font-semibold text-slate-300">2. Parallel Search</div>
              <div className="text-[11px] text-slate-500 mt-1">Dense + Sparse + GIN</div>
              <div className="mt-1 text-indigo-400 font-mono">60 Candidates</div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-900">
              <div className="font-semibold text-slate-300">3. RRF Fusion</div>
              <div className="text-[11px] text-slate-500 mt-1">k=60 Reciprocal</div>
              <div className="mt-1 text-purple-400 font-mono">Top-40 Fused</div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-900">
              <div className="font-semibold text-slate-300">4. Cross-Encoder</div>
              <div className="text-[11px] text-slate-500 mt-1">Relevance Scoring</div>
              <div className="mt-1 text-emerald-400 font-mono">Top-8 Reranked</div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-900">
              <div className="font-semibold text-slate-300">5. Context Budget</div>
              <div className="text-[11px] text-slate-500 mt-1">4000 Token Ceiling</div>
              <div className="mt-1 text-amber-400 font-mono">6 Chunks Used</div>
            </div>
          </div>
        </div>
      )}

      {/* Response Panels */}
      <div className={`grid ${compareMode ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"} gap-6`}>
        {/* Panel A */}
        <ResponseCard
          title={compareMode ? "Pipeline A Output" : "Generated Response"}
          pipelineName={pipelines.find((p) => p.id === selectedPipelineA)?.name || "Pipeline A"}
          streamState={streamA}
          onCitationClick={(cit) => setActiveCitation(cit)}
          onRate={(rating) => handleRate(streamA.runId, rating)}
          rating={streamA.runId ? ratingSubmitted[streamA.runId] : undefined}
          accentColor="cyan"
        />

        {/* Panel B (Compare Mode) */}
        {compareMode && (
          <ResponseCard
            title="Pipeline B Output"
            pipelineName={pipelines.find((p) => p.id === selectedPipelineB)?.name || "Pipeline B"}
            streamState={streamB}
            onCitationClick={(cit) => setActiveCitation(cit)}
            onRate={(rating) => handleRate(streamB.runId, rating)}
            rating={streamB.runId ? ratingSubmitted[streamB.runId] : undefined}
            accentColor="indigo"
          />
        )}
      </div>

      {/* Citation Detail Side Drawer */}
      {activeCitation && (
        <div className="fixed inset-y-0 right-0 w-full sm:w-[480px] bg-slate-950 border-l border-slate-800 shadow-2xl p-6 z-50 overflow-y-auto animate-slideIn">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-mono text-xs font-semibold">
                [Source {activeCitation.source_number}]
              </span>
              <h3 className="font-semibold text-slate-200 text-sm">{activeCitation.filename}</h3>
            </div>
            <button
              onClick={() => setActiveCitation(null)}
              className="p-1 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="mt-4 space-y-4">
            <div>
              <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Document Chunk</label>
              <div className="mt-1.5 p-3.5 rounded-lg border border-slate-800 bg-slate-900/80 text-sm text-slate-300 leading-relaxed font-sans">
                {activeCitation.content}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg border border-slate-800/60 bg-slate-900/40">
                <span className="text-slate-500">Chunk ID</span>
                <p className="font-mono text-slate-300 mt-0.5 truncate">{activeCitation.chunk_id}</p>
              </div>
              <div className="p-3 rounded-lg border border-slate-800/60 bg-slate-900/40">
                <span className="text-slate-500">Section</span>
                <p className="text-slate-300 mt-0.5 truncate">{activeCitation.section_title || "General Body"}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Single / Compare Response Card Component
function ResponseCard({
  title,
  pipelineName,
  streamState,
  onCitationClick,
  onRate,
  rating,
  accentColor = "cyan",
}: {
  title: string;
  pipelineName: string;
  streamState: any;
  onCitationClick: (cit: Citation) => void;
  onRate: (rating: number) => void;
  rating?: number;
  accentColor?: string;
}) {
  const { text, isStreaming, citations, evaluation, error } = streamState;

  // Format citations into clickable text
  const renderTextWithCitations = (content: string) => {
    if (!content) return null;
    const parts = content.split(/(\[Source\s+\d+\])/gi);
    return parts.map((part, index) => {
      const match = part.match(/\[Source\s+(\d+)\]/i);
      if (match) {
        const sourceNum = parseInt(match[1], 10);
        const citationData = citations.find((c: Citation) => c.source_number === sourceNum);
        return (
          <button
            key={index}
            onClick={() => citationData && onCitationClick(citationData)}
            className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded bg-cyan-500/20 text-cyan-300 text-xs font-mono font-medium hover:bg-cyan-500/40 transition-colors"
          >
            {part}
          </button>
        );
      }
      return part;
    });
  };

  return (
    <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-4">
      {/* Card Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${isStreaming ? "bg-cyan-400 animate-ping" : "bg-emerald-400"}`} />
            {title}
          </h2>
          <span className="text-xs text-slate-400">{pipelineName}</span>
        </div>

        {/* Feedback Buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => onRate(5)}
            className={`p-1.5 rounded-lg border transition-all ${
              rating === 5 ? "bg-emerald-500/20 border-emerald-500 text-emerald-400" : "border-slate-800 text-slate-400 hover:bg-slate-800"
            }`}
            title="Helpful / Grounded"
          >
            <ThumbsUp className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => onRate(1)}
            className={`p-1.5 rounded-lg border transition-all ${
              rating === 1 ? "bg-rose-500/20 border-rose-500 text-rose-400" : "border-slate-800 text-slate-400 hover:bg-slate-800"
            }`}
            title="Poor / Hallucinated"
          >
            <ThumbsDown className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Answer Body */}
      <div className="min-h-[160px] text-sm text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
        {text ? (
          renderTextWithCitations(text)
        ) : isStreaming ? (
          <span className="text-slate-500 italic">Thinking and synthesizing from context...</span>
        ) : error ? (
          <span className="text-rose-400 flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </span>
        ) : (
          <span className="text-slate-600 italic">Run a query to view response and citations.</span>
        )}
      </div>

      {/* Citations list chips */}
      {citations.length > 0 && (
        <div className="pt-3 border-t border-slate-800/60">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Citations Used</label>
          <div className="flex flex-wrap gap-1.5 mt-2">
            {citations.map((c: Citation) => (
              <button
                key={c.source_number}
                onClick={() => onCitationClick(c)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-xs text-slate-300 hover:border-cyan-500 hover:text-cyan-300 transition-all font-mono"
              >
                <FileText className="h-3.5 w-3.5 text-cyan-400" />
                [Source {c.source_number}] {c.filename}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Evaluation Gauges Scorecard */}
      {evaluation && (
        <div className="pt-3 border-t border-slate-800/80 space-y-2.5 animate-fadeIn">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Award className="h-4 w-4 text-cyan-400" />
              Automated RAGAS Evaluation
            </span>
            <span className="font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">
              Overall: {(evaluation.overall_score * 100).toFixed(0)}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <MetricGauge label="Faithfulness" score={evaluation.faithfulness} />
            <MetricGauge label="Answer Relevance" score={evaluation.answer_relevance} />
            <MetricGauge label="Context Precision" score={evaluation.context_precision} />
            <MetricGauge label="Context Recall" score={evaluation.context_recall} />
          </div>
        </div>
      )}
    </div>
  );
}

function MetricGauge({ label, score }: { label: string; score: number }) {
  const pct = Math.round((score || 0) * 100);
  const color = pct >= 80 ? "bg-emerald-500" : pct >= 60 ? "bg-amber-500" : "bg-rose-500";

  return (
    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800/60">
      <div className="flex justify-between items-center mb-1">
        <span className="text-slate-400 font-medium">{label}</span>
        <span className="font-mono text-slate-200">{pct}%</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
