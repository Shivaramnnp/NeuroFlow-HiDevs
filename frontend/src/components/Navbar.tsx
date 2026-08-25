"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { 
  Terminal, 
  GitFork, 
  BarChart3, 
  FileText, 
  Activity, 
  Cpu, 
  CheckCircle2, 
  AlertTriangle,
  XCircle
} from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const [healthStatus, setHealthStatus] = useState<string>("checking");

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await api.getHealth();
        setHealthStatus(res.data.status || "ok");
      } catch (err) {
        setHealthStatus("critical");
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { name: "Query Playground", href: "/playground", icon: Terminal },
    { name: "Pipelines", href: "/pipelines", icon: GitFork },
    { name: "Evaluations Feed", href: "/evaluations", icon: BarChart3 },
    { name: "Documents", href: "/documents", icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link href="/playground" className="flex items-center gap-2.5 group">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="h-full w-full bg-slate-950 rounded-[7px] flex items-center justify-center">
              <Cpu className="h-5 w-5 text-cyan-400 group-hover:rotate-12 transition-transform" />
            </div>
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg tracking-tight text-white flex items-center gap-1.5">
              NeuroFlow <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 font-mono">v1.0</span>
            </span>
            <span className="text-[10px] text-slate-400 font-medium tracking-wider uppercase">Enterprise RAG Engine</span>
          </div>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-slate-800 text-cyan-400 shadow-inner"
                    : "text-slate-300 hover:bg-slate-900 hover:text-white"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Resilience Health Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full border border-slate-800 bg-slate-900/60 text-xs font-mono">
            {healthStatus === "ok" && (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400">System Healthy</span>
              </>
            )}
            {healthStatus === "degraded" && (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-amber-400">Degraded / Circuit Open</span>
              </>
            )}
            {healthStatus === "critical" && (
              <>
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                <span className="text-rose-400">Backend Offline</span>
              </>
            )}
            {healthStatus === "checking" && (
              <>
                <span className="h-2 w-2 rounded-full bg-slate-500 animate-pulse" />
                <span className="text-slate-400">Connecting...</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
