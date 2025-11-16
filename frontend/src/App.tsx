/**
 * Main App component - clean layout with sidebar
 */
import Sidebar from './components/Sidebar';
import ChatContainer from './components/ChatContainer';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <ChatContainer />
    </div>
  );
}

export default App;
