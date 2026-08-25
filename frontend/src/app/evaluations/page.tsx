"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, API_BASE_URL } from "@/lib/api";
import { EvaluationData } from "@/lib/types";
import {
  BarChart3,
  Radio,
  Filter,
  Search,
  Award,
  Calendar,
  ChevronDown,
  ChevronUp,
  Clock,
  Sparkles,
  Layers,
  FileCheck
} from "lucide-react";

export default function EvaluationsPage() {
  const [evaluations, setEvaluations] = useState<any[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterMinScore, setFilterMinScore] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [sseConnected, setSseConnected] = useState<boolean>(false);

  // Fetch initial history
  const { data: initialEvals = [], isLoading } = useQuery({
    queryKey: ["evaluations-history"],
    queryFn: async () => {
      const res = await api.getEvaluations();
      return res.data;
    },
  });

  useEffect(() => {
    if (initialEvals.length > 0) {
      setEvaluations(initialEvals);
    }
  }, [initialEvals]);

  // Connect to Real-time SSE Feed: GET /evaluations/stream
  useEffect(() => {
    const sse = new EventSource(`${API_BASE_URL}/evaluations/stream`);

    sse.addEventListener("connected", () => {
      setSseConnected(true);
    });

    sse.addEventListener("evaluation", (event) => {
      try {
        const newEval = JSON.parse(event.data);
        setEvaluations((prev) => [newEval, ...prev]);
      } catch (e) {
        console.error("Evaluation parse error", e);
      }
    });

    sse.onerror = () => {
      setSseConnected(false);
    };

    return () => {
      sse.close();
    };
  }, []);

  const filteredEvaluations = evaluations.filter((item) => {
    const matchesScore = (item.overall_score || 0) >= filterMinScore;
    const matchesQuery = searchQuery
      ? item.query?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.pipeline_name?.toLowerCase().includes(searchQuery.toLowerCase())
      : true;
    return matchesScore && matchesQuery;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-purple-400" />
            Live Evaluation Feed
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Real-time streaming evaluation stream monitoring faithfulness, relevance, precision, and recall per generation.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-slate-800 bg-slate-900/80 text-xs font-mono">
          <Radio className={`h-3.5 w-3.5 ${sseConnected ? "text-emerald-400 animate-pulse" : "text-slate-500"}`} />
          <span className={sseConnected ? "text-emerald-400" : "text-slate-500"}>
            {sseConnected ? "SSE Feed Active (Pub/Sub)" : "Connecting to SSE..."}
          </span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col md:flex-row items-center gap-3 justify-between">
        <div className="flex items-center gap-2 w-full md:w-80 relative">
          <Search className="h-4 w-4 text-slate-500 absolute left-3" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search query or pipeline..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
          />
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <span className="text-xs text-slate-400 flex items-center gap-1">
            <Filter className="h-3.5 w-3.5" /> Min Score:
          </span>
          <select
            value={filterMinScore}
            onChange={(e) => setFilterMinScore(parseFloat(e.target.value))}
            className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none"
          >
            <option value={0}>All Scores (0%+)</option>
            <option value={0.7}>Quality &gt;= 70%</option>
            <option value={0.8}>Fine-tuning eligible &gt;= 80%</option>
            <option value={0.9}>Exemplary &gt;= 90%</option>
          </select>
        </div>
      </div>

      {/* Feed List */}
      <div className="space-y-3">
        {filteredEvaluations.map((item, idx) => {
          const isExpanded = expandedId === (item.eval_id || item.run_id || idx.toString());
          const score = item.overall_score || 0.85;
          const scoreColor =
            score >= 0.8 ? "text-emerald-400 border-emerald-500/30 bg-emerald-500/10" : score >= 0.6 ? "text-amber-400 border-amber-500/30 bg-amber-500/10" : "text-rose-400 border-rose-500/30 bg-rose-500/10";

          return (
            <div
              key={item.eval_id || item.run_id || idx}
              className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 transition-all hover:border-slate-700"
            >
              {/* Card Summary Line */}
              <div
                onClick={() => setExpandedId(isExpanded ? null : (item.eval_id || item.run_id || idx.toString()))}
                className="flex items-center justify-between cursor-pointer gap-4"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-purple-300">
                      {item.pipeline_name || "default_pipeline"}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">
                      {item.evaluated_at?.slice(11, 19) || "Just now"}
                    </span>
                  </div>
                  <h3 className="font-semibold text-slate-200 text-sm mt-1 truncate">{item.query}</h3>
                </div>

                <div className="flex items-center gap-3">
                  <div className={`px-2.5 py-1 rounded-md border text-xs font-mono font-bold ${scoreColor}`}>
                    {(score * 100).toFixed(0)}%
                  </div>
                  {isExpanded ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
                </div>
              </div>

              {/* Metric Bars */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 pt-3 border-t border-slate-800/60 text-[11px]">
                <div>
                  <span className="text-slate-500">Faithfulness</span>
                  <p className="font-mono text-slate-200 font-bold">{Math.round((item.faithfulness || 0) * 100)}%</p>
                </div>
                <div>
                  <span className="text-slate-500">Relevance</span>
                  <p className="font-mono text-slate-200 font-bold">{Math.round((item.answer_relevance || 0) * 100)}%</p>
                </div>
                <div>
                  <span className="text-slate-500">Precision</span>
                  <p className="font-mono text-slate-200 font-bold">{Math.round((item.context_precision || 0) * 100)}%</p>
                </div>
                <div>
                  <span className="text-slate-500">Recall</span>
                  <p className="font-mono text-slate-200 font-bold">{Math.round((item.context_recall || 0) * 100)}%</p>
                </div>
              </div>

              {/* Expanded Answer / Generation */}
              {isExpanded && item.answer && (
                <div className="mt-4 pt-3 border-t border-slate-800 space-y-2 animate-fadeIn">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Generated Answer</span>
                  <div className="p-3.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-300 leading-relaxed font-sans whitespace-pre-wrap">
                    {item.answer}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
