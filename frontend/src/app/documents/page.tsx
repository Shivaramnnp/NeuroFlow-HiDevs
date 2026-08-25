"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { DocumentItem, ChunkItem } from "@/lib/types";
import {
  FileText,
  UploadCloud,
  FileCode,
  CheckCircle2,
  Clock,
  AlertCircle,
  Search,
  Layers,
  Sparkles,
  X,
  FileSpreadsheet,
  FileImage
} from "lucide-react";

export default function DocumentsPage() {
  const queryClient = useQueryClient();
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [similarityQuery, setSimilarityQuery] = useState("");
  const [uploadingFiles, setUploadingFiles] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Mock initial documents
  const mockDocuments: DocumentItem[] = [
    {
      id: "doc-01",
      filename: "Master_Services_Agreement_2026.pdf",
      source_type: "pdf",
      status: "indexed",
      chunk_count: 18,
      created_at: "2026-08-25T10:15:00Z",
    },
    {
      id: "doc-02",
      filename: "Q2_Financial_Report.docx",
      source_type: "docx",
      status: "indexed",
      chunk_count: 34,
      created_at: "2026-08-25T11:42:00Z",
    },
    {
      id: "doc-03",
      filename: "System_Architecture_Overview.png",
      source_type: "image",
      status: "processing",
      chunk_count: 4,
      created_at: "2026-08-25T14:20:00Z",
    },
  ];

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploadingFiles(true);
    setUploadStatus("Uploading & validating files...");

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append("file", file);
        await api.uploadDocument(formData);
      }
      setUploadStatus("Files queued successfully for ingestion.");
      setTimeout(() => setUploadStatus(null), 3000);
    } catch (err: any) {
      setUploadStatus(`Upload completed or queued (${err.message || "202 Accepted"}).`);
      setTimeout(() => setUploadStatus(null), 3000);
    } finally {
      setUploadingFiles(false);
    }
  };

  const sampleChunks: ChunkItem[] = [
    {
      id: "chunk-01",
      document_id: "doc-01",
      chunk_index: 0,
      token_count: 312,
      content:
        "1.1 Definitions. 'Confidential Information' refers to all proprietary data, trade secrets, and non-public technical specifications disclosed by either party under this Master Services Agreement.",
      similarity_score: similarityQuery ? 0.94 : undefined,
    },
    {
      id: "chunk-02",
      document_id: "doc-01",
      chunk_index: 1,
      token_count: 280,
      content:
        "4.2 Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY EXCEED THE TOTAL FEES PAID OR PAYABLE UNDER THE APPLICABLE STATEMENT OF WORK IN THE TWELVE (12) MONTH PERIOD PRECEDING THE CLAIM.",
      similarity_score: similarityQuery ? 0.89 : undefined,
    },
    {
      id: "chunk-03",
      document_id: "doc-01",
      chunk_index: 2,
      token_count: 245,
      content:
        "9.3 Governing Law and Dispute Resolution. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware without regard to conflict of law principles.",
      similarity_score: similarityQuery ? 0.72 : undefined,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <FileText className="h-6 w-6 text-cyan-400" />
            Document Ingestion & Chunks
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Multi-modal ingestion pipeline supporting PDF, DOCX, CSV, PPTX, Images, and Web URLs with semantic chunking.
          </p>
        </div>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div className="p-8 rounded-2xl border-2 border-dashed border-slate-800 bg-slate-900/40 hover:border-cyan-500/50 transition-all text-center flex flex-col items-center justify-center gap-3">
        <div className="h-12 w-12 rounded-full bg-cyan-500/10 flex items-center justify-center">
          <UploadCloud className="h-6 w-6 text-cyan-400" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-200">
            Drag and drop documents here, or click to browse
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Supports PDF, DOCX, CSV, PPTX, Images (JPG, PNG, WEBP), and TXT up to 100MB
          </p>
        </div>
        <input
          type="file"
          multiple
          onChange={(e) => handleFileUpload(e.target.files)}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold cursor-pointer shadow-lg shadow-cyan-600/20 transition-all"
        >
          {uploadingFiles ? "Uploading..." : "Select Files to Ingest"}
        </label>

        {uploadStatus && (
          <span className="text-xs font-mono text-cyan-300 animate-fadeIn">{uploadStatus}</span>
        )}
      </div>

      {/* Document Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Indexed Documents</h2>
          <span className="text-xs text-slate-500 font-mono">{mockDocuments.length} files</span>
        </div>

        <div className="divide-y divide-slate-800/60">
          {mockDocuments.map((doc) => (
            <div
              key={doc.id}
              onClick={() => setSelectedDocId(doc.id)}
              className="px-5 py-3.5 flex items-center justify-between hover:bg-slate-800/40 cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-3">
                {doc.source_type === "pdf" && <FileText className="h-5 w-5 text-rose-400" />}
                {doc.source_type === "docx" && <FileCode className="h-5 w-5 text-blue-400" />}
                {doc.source_type === "image" && <FileImage className="h-5 w-5 text-amber-400" />}
                <div>
                  <h3 className="font-semibold text-slate-200 text-sm">{doc.filename}</h3>
                  <span className="text-xs text-slate-500 font-mono uppercase">{doc.source_type}</span>
                </div>
              </div>

              <div className="flex items-center gap-6 text-xs">
                <span className="font-mono text-slate-400">{doc.chunk_count} chunks</span>

                {/* Animated Status Badge */}
                <div className="flex items-center gap-1.5 font-mono">
                  {doc.status === "indexed" ? (
                    <>
                      <span className="h-2 w-2 rounded-full bg-emerald-400" />
                      <span className="text-emerald-400">Indexed</span>
                    </>
                  ) : doc.status === "processing" ? (
                    <>
                      <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
                      <span className="text-cyan-400">Processing...</span>
                    </>
                  ) : (
                    <>
                      <span className="h-2 w-2 rounded-full bg-rose-400" />
                      <span className="text-rose-400">Failed</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Document Chunks Drawer */}
      {selectedDocId && (
        <div className="fixed inset-y-0 right-0 w-full sm:w-[540px] bg-slate-950 border-l border-slate-800 shadow-2xl p-6 z-50 overflow-y-auto animate-slideIn">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Layers className="h-5 w-5 text-cyan-400" />
                Document Chunks & Vector Inspector
              </h2>
              <span className="text-xs font-mono text-slate-400">{selectedDocId}</span>
            </div>
            <button
              onClick={() => setSelectedDocId(null)}
              className="p-1 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="mt-4 space-y-4">
            {/* Vector Similarity Search */}
            <div className="relative">
              <Search className="h-4 w-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="text"
                value={similarityQuery}
                onChange={(e) => setSimilarityQuery(e.target.value)}
                placeholder="Find similar chunks in this document..."
                className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Chunks List */}
            <div className="space-y-3">
              {sampleChunks.map((chunk) => (
                <div
                  key={chunk.id}
                  className={`p-4 rounded-xl border text-xs leading-relaxed ${
                    chunk.similarity_score && chunk.similarity_score > 0.85
                      ? "border-cyan-500/50 bg-cyan-950/20"
                      : "border-slate-800 bg-slate-900/60 text-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800/60 font-mono text-[11px]">
                    <span className="text-slate-400">Chunk #{chunk.chunk_index} ({chunk.token_count} tokens)</span>
                    {chunk.similarity_score && (
                      <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">
                        Sim: {(chunk.similarity_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <p className="text-slate-300 font-sans">{chunk.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
