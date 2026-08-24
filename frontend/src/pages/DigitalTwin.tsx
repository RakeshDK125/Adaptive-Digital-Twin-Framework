export default function DigitalTwin() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Digital Twin Virtual Representation</h1>
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 flex flex-col items-center justify-center min-h-[400px]">
        <div className="text-center space-y-4">
          <div className="w-32 h-32 border-4 border-emerald-500/30 border-t-emerald-400 rounded-full animate-spin mx-auto"></div>
          <h2 className="text-xl font-medium text-emerald-400">TURBINE-01 Synced</h2>
          <p className="text-slate-400">Live physical asset synchronized with virtual engine.</p>
        </div>
      </div>
    </div>
  );
}
