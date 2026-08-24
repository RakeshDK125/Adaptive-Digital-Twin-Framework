export default function Explainability() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Explainable AI (XAI)</h1>
      
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 className="text-lg font-medium mb-4">SHAP Feature Importance (RL Policy)</h2>
          <div className="space-y-4">
            {/* Mock SHAP values */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Wear Factor</span>
                <span className="text-blue-400">0.85</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '85%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Vibration (Hz)</span>
                <span className="text-blue-400">0.62</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '62%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Temperature (C)</span>
                <span className="text-blue-400">0.31</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '31%' }}></div>
              </div>
            </div>
          </div>
          <p className="text-sm text-slate-400 mt-6">
            The RL agent is heavily prioritizing the wear factor when determining the continuous control action for the cooling flow.
          </p>
        </div>

        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 className="text-lg font-medium mb-4">Counterfactual "What-If" Analysis</h2>
          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 mb-4">
            <h4 className="text-sm text-slate-400 mb-2">Hypothetical Scenario</h4>
            <p className="text-sm">If <span className="font-bold text-emerald-400">Temperature</span> was <span className="font-bold text-emerald-400">20% lower</span>...</p>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
              <span className="text-slate-400">Original Action (Cooling)</span>
              <span className="font-mono text-slate-200">0.45</span>
            </div>
            <div className="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
              <span className="text-slate-400">Counterfactual Action</span>
              <span className="font-mono text-emerald-400">0.12</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-400">Delta</span>
              <span className="font-mono text-red-400">-0.33 (Decrease)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
