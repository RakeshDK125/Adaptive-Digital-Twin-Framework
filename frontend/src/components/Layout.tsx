import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { Bell, User } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen bg-slate-950 text-slate-50 overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="h-16 border-b border-slate-800 bg-slate-900/50 backdrop-blur flex items-center justify-between px-6">
          <div className="text-sm font-medium text-slate-400">
            <span className="text-emerald-400 mr-2">●</span>
            System Online (Machine ID: <span className="text-slate-200">TURBINE-01</span>)
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <button className="hover:text-slate-100 transition"><Bell size={18} /></button>
            <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-slate-300">
              <User size={16} />
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
