/**
 * Sidebar component with session management
 */
import { useChatStore } from '../store/chatStore';
import SettingsPanel from './SettingsPanel';

export default function Sidebar() {
  const { sessions, currentSessionId, createNewSession, switchSession } = useChatStore();

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
    </div>
  );
}
