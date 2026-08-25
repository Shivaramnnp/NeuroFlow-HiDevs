import { useState, useCallback, useRef } from "react";
import { API_BASE_URL } from "@/lib/api";
import { Citation, EvaluationData } from "@/lib/types";

interface SSEStreamState {
  runId: string | null;
  text: string;
  isStreaming: boolean;
  citations: Citation[];
  sources: any[];
  evaluation: EvaluationData | null;
  error: string | null;
  totalTokens: number;
}

export function useSSEStream() {
  const [state, setState] = useState<SSEStreamState>({
    runId: null,
    text: "",
    isStreaming: false,
    citations: [],
    sources: [],
    evaluation: null,
    error: null,
    totalTokens: 0,
  });

  const eventSourceRef = useRef<EventSource | null>(null);

  const startStream = useCallback((query: string, pipelineId?: string) => {
    // Reset state
    setState({
      runId: null,
      text: "",
      isStreaming: true,
      citations: [],
      sources: [],
      evaluation: null,
      error: null,
      totalTokens: 0,
    });

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Step 1: POST /query with stream=true
    fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        pipeline_id: pipelineId,
        stream: true,
      }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to initiate stream: HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        const runId = data.run_id;
        setState((prev) => ({ ...prev, runId }));

        // Step 2: Open SSE stream GET /query/{runId}/stream
        const sse = new EventSource(`${API_BASE_URL}/query/${runId}/stream`);
        eventSourceRef.current = sse;

        sse.addEventListener("token", (event) => {
          try {
            const parsed = JSON.parse(event.data);
            const token = parsed.token || "";
            setState((prev) => ({
              ...prev,
              text: prev.text + token,
              totalTokens: prev.totalTokens + 1,
            }));
          } catch (e) {
            console.error("Token parse error", e);
          }
        });

        sse.addEventListener("citations", (event) => {
          try {
            const parsed = JSON.parse(event.data);
            setState((prev) => ({
              ...prev,
              citations: parsed.citations || [],
            }));
          } catch (e) {
            console.error("Citations parse error", e);
          }
        });

        sse.addEventListener("done", (event) => {
          try {
            const parsed = JSON.parse(event.data);
            setState((prev) => ({
              ...prev,
              isStreaming: false,
              sources: parsed.sources || [],
            }));
            sse.close();

            // Step 3: Fetch evaluation after 1.5s delay
            setTimeout(() => {
              fetch(`${API_BASE_URL}/evaluations?run_id=${runId}&limit=1`)
                .then((r) => r.json())
                .then((evalList) => {
                  if (evalList && evalList.length > 0) {
                    setState((prev) => ({
                      ...prev,
                      evaluation: evalList[0],
                    }));
                  }
                })
                .catch((e) => console.warn("Failed to fetch evaluation", e));
            }, 1500);
          } catch (e) {
            sse.close();
          }
        });

        sse.onerror = (err) => {
          console.error("SSE error", err);
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            error: "Streaming connection closed or interrupted.",
          }));
          sse.close();
        };
      })
      .catch((err) => {
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          error: err.message || "Failed to start query stream",
        }));
      });
  }, []);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      setState((prev) => ({ ...prev, isStreaming: false }));
    }
  }, []);

  return {
    ...state,
    startStream,
    stopStream,
  };
}
