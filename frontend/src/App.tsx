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
import AdminPanel from './components/AdminPanel';
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

  // Render appropriate view based on admin state
  const renderMainContent = () => {
    switch (adminView) {
      case 'system-prompt':
        return <SystemPromptEditor />;
      case 'admin-panel':
        return <AdminPanel />;
      default:
        return <ChatContainer />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar />
      {renderMainContent()}
    </div>
  );
}

export default App;
