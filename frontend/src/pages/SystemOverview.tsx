import Plot from 'react-plotly.js';

export default function SystemOverview() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">System Overview</h1>
      
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
          <h3 className="text-slate-400 text-sm">Overall Health Score</h3>
          <p className="text-3xl font-bold text-emerald-400 mt-2">92%</p>
        </div>
        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
          <h3 className="text-slate-400 text-sm">Active RL Policies</h3>
          <p className="text-3xl font-bold text-blue-400 mt-2">PPO (v2.4)</p>
        </div>
        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
          <h3 className="text-slate-400 text-sm">Swarm Agents Active</h3>
          <p className="text-3xl font-bold text-purple-400 mt-2">9</p>
        </div>
        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
          <h3 className="text-slate-400 text-sm">Risk Level</h3>
          <p className="text-3xl font-bold text-emerald-400 mt-2">Low</p>
        </div>
      </div>

      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
        <h2 className="text-lg font-medium mb-4">Historical Risk & Confidence Trend</h2>
        <div className="h-[400px]">
          <Plot
            data={[
              {
                x: ['10:00', '11:00', '12:00', '13:00', '14:00'],
                y: [20, 22, 18, 45, 12],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Risk Score',
                marker: { color: '#ef4444' }
              },
              {
                x: ['10:00', '11:00', '12:00', '13:00', '14:00'],
                y: [95, 96, 95, 88, 97],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'AI Confidence',
                marker: { color: '#3b82f6' }
              }
            ]}
            layout={{
              autosize: true,
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8' },
              margin: { t: 10, l: 40, r: 10, b: 40 },
              xaxis: { gridcolor: '#1e293b' },
              yaxis: { gridcolor: '#1e293b' }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </div>
  );
}
