import { Link, useLocation } from 'react-router-dom';
import { 
  Activity, Server, Brain, Users, GitMerge, 
  LineChart, Eye, PenTool, PieChart, Settings 
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'System Overview', icon: Activity },
  { path: '/twin', label: 'Digital Twin', icon: Server },
  { path: '/rl', label: 'RL Training', icon: Brain },
  { path: '/agents', label: 'Agent Decisions', icon: Users },
  { path: '/knowledge', label: 'Knowledge Graph', icon: GitMerge },
  { path: '/sensors', label: 'Sensor Monitoring', icon: LineChart },
  { path: '/xai', label: 'Explainability', icon: Eye },
  { path: '/maintenance', label: 'Maintenance', icon: PenTool },
  { path: '/analytics', label: 'Analytics', icon: PieChart },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-screen">
      <div className="h-16 flex items-center px-6 border-b border-slate-800">
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          Adaptive Twin AI
        </h1>
      </div>
      <nav className="flex-1 overflow-y-auto py-4">
        <ul className="space-y-1 px-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                    isActive 
                      ? 'bg-blue-500/10 text-blue-400' 
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
                  }`}
                >
                  <Icon size={18} />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
