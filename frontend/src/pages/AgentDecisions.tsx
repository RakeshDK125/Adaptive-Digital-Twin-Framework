export default function AgentDecisions() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Agentic AI Swarm Blackboard</h1>
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 bg-slate-900 rounded-xl border border-slate-800 p-6 min-h-[400px]">
          <h2 className="text-lg font-medium mb-4">Live Swarm Communication</h2>
          <div className="space-y-4">
            <div className="bg-slate-800 p-3 rounded-lg border border-slate-700">
              <span className="text-xs text-blue-400 font-bold">MONITORING AGENT</span>
              <p className="text-sm mt-1">Telemetry spike detected on Vibration Sensor (6.2 Hz).</p>
            </div>
            <div className="bg-slate-800 p-3 rounded-lg border border-slate-700 ml-8">
              <span className="text-xs text-purple-400 font-bold">REASONING AGENT</span>
              <p className="text-sm mt-1">Diagnosing... Likely caused by wear factor exceeding 80% threshold.</p>
            </div>
            <div className="bg-slate-800 p-3 rounded-lg border border-slate-700 ml-16">
              <span className="text-xs text-emerald-400 font-bold">KNOWLEDGE AGENT</span>
              <p className="text-sm mt-1">Historical precedent found: Node [AgentDecision-9382].</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 className="text-lg font-medium mb-4">Shared Memory Context</h2>
          <pre className="text-xs text-emerald-400 bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto">
            {JSON.stringify({
              "anomaly_detected": true,
              "current_diagnosis": "High wear detected",
              "proposed_plan": [
                {"step": 1, "task": "Query historical context"},
                {"step": 2, "task": "Trigger RL Online Finetuning"}
              ]
            }, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
