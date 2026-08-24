import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import SystemOverview from './pages/SystemOverview';
import DigitalTwin from './pages/DigitalTwin';
import AgentDecisions from './pages/AgentDecisions';
import KnowledgeGraph from './pages/KnowledgeGraph';
import Explainability from './pages/Explainability';

// Mock empty pages for the rest to satisfy router
const Placeholder = ({ title }: { title: string }) => (
  <div className="p-6"><h1 className="text-2xl font-bold">{title}</h1></div>
);

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<SystemOverview />} />
          <Route path="/twin" element={<DigitalTwin />} />
          <Route path="/rl" element={<Placeholder title="RL Training & Evaluation" />} />
          <Route path="/agents" element={<AgentDecisions />} />
          <Route path="/knowledge" element={<KnowledgeGraph />} />
          <Route path="/sensors" element={<Placeholder title="Sensor Monitoring" />} />
          <Route path="/xai" element={<Explainability />} />
          <Route path="/maintenance" element={<Placeholder title="Maintenance Recommendations" />} />
          <Route path="/analytics" element={<Placeholder title="Analytics & Degradation" />} />
          <Route path="/settings" element={<Placeholder title="System Settings" />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
