/**
 * Sidebar component with session management
 */
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/chatStore';
import { useAuthStore } from '../store/authStore';
import SettingsPanel from './SettingsPanel';

export default function Sidebar() {
  const navigate = useNavigate();
  const { sessions, currentSessionId, createNewSession, switchSession } = useChatStore();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>Negotiator Pro</h1>
        <p>AI negotiation guidance</p>
      </div>

      {/* Settings Panel */}
      <SettingsPanel />

      {/* Separator */}
      <div className="sidebar-separator"></div>

      <button className="new-session-btn" onClick={createNewSession}>
        + New Negotiation
      </button>

      <div className="sessions-list">
        <h3 style={{ fontSize: '12px', color: '#6c757d', padding: '8px 16px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Negotiation Sessions
        </h3>
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
            onClick={() => switchSession(session.id)}
          >
            <div style={{ fontSize: '14px', fontWeight: 500, marginBottom: '4px' }}>
              {session.title || 'New Conversation'}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.7 }}>
              {session.messageCount} messages • {new Date(session.createdAt).toLocaleDateString()}
            </div>
          </div>
        ))}

        {sessions.length === 0 && (
          <div style={{ padding: '16px', textAlign: 'center', color: '#6c757d', fontSize: '13px' }}>
            No sessions yet. Click "New Negotiation" to start.
          </div>
        )}
      </div>

      {/* User info and actions */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        marginTop: 'auto'
      }}>
        <div style={{
          fontSize: '13px',
          color: '#9fadbd',
          marginBottom: '12px'
        }}>
          Logged in as <strong>{user?.username || 'User'}</strong>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => navigate('/profile')}
            style={{
              width: '100%',
              padding: '8px 16px',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            }}
          >
            My Profile
          </button>
          <button
            onClick={handleLogout}
            style={{
              width: '100%',
              padding: '8px 16px',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            }}
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
}
