export default function KnowledgeGraph() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Neo4j Knowledge Graph</h1>
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 min-h-[500px] flex items-center justify-center relative overflow-hidden">
        {/* Mock representation of a graph network */}
        <div className="absolute w-3 h-3 bg-blue-500 rounded-full top-1/4 left-1/4 shadow-[0_0_15px_rgba(59,130,246,0.5)]"></div>
        <div className="absolute w-3 h-3 bg-red-500 rounded-full top-1/2 left-1/2 shadow-[0_0_15px_rgba(239,68,68,0.5)]"></div>
        <div className="absolute w-3 h-3 bg-emerald-500 rounded-full bottom-1/4 right-1/3 shadow-[0_0_15px_rgba(16,185,129,0.5)]"></div>
        
        <svg className="absolute inset-0 w-full h-full opacity-20 pointer-events-none">
          <line x1="25%" y1="25%" x2="50%" y2="50%" stroke="#fff" strokeWidth="2" />
          <line x1="50%" y1="50%" x2="66%" y2="75%" stroke="#fff" strokeWidth="2" />
        </svg>

        <div className="z-10 bg-slate-950/80 backdrop-blur border border-slate-800 p-6 rounded-xl max-w-sm">
          <h3 className="font-bold text-red-400 mb-2">Node: Failure [evt_0912]</h3>
          <p className="text-sm text-slate-300">High vibration anomaly detected</p>
          <div className="mt-4 pt-4 border-t border-slate-800">
            <span className="text-xs text-slate-500">RESOLVED_BY</span>
            <p className="text-sm text-emerald-400 mt-1">AgentDecision: Adjusted cooling flow rate by 5%.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
