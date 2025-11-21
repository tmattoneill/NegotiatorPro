/**
 * Main App component - clean layout with sidebar
 */
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import { useAdminStore } from './store/adminStore';
import Sidebar from './components/Sidebar';
import ChatContainer from './components/ChatContainer';
import SystemPromptEditor from './components/SystemPromptEditor';
import './App.css';

function App() {
  const navigate = useNavigate();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const adminView = useAdminStore((state) => state.currentView);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, navigate]);

  // Don't render the app if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="app-container">
      <Sidebar />
      {adminView === 'system-prompt' ? <SystemPromptEditor /> : <ChatContainer />}
    </div>
  );
}

export default App;
