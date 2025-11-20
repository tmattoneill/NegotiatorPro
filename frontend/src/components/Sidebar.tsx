/**
 * Sidebar component with session management
 */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/chatStore';
import { useAuthStore } from '../store/authStore';
import SettingsModal from './SettingsModal';
import ModelSelector from './ModelSelector';

export default function Sidebar() {
  const navigate = useNavigate();
  const { sessions, currentSessionId, createNewSession, switchSession } = useChatStore();
  const { user, logout } = useAuthStore();
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

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

      {/* Model Selector - shows models from selected provider */}
      <div style={{ padding: '16px 16px 0 16px' }}>
        <ModelSelector />
      </div>

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
          marginBottom: '16px'
        }}>
          Logged in as <strong>{user?.username || 'User'}</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
          {/* Profile Icon */}
          <button
            onClick={() => navigate('/profile')}
            title="My Profile"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <i className="fa-light fa-user"></i>
          </button>

          {/* Settings Icon */}
          <button
            onClick={() => setShowSettingsModal(true)}
            title="Settings"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <i className="fa-light fa-gear"></i>
          </button>

          {/* Logout Icon */}
          <button
            onClick={() => setShowLogoutModal(true)}
            title="Logout"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <i className="fa-light fa-right-from-bracket"></i>
          </button>
        </div>
      </div>

      {/* Logout Confirmation Modal */}
      {showLogoutModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}>
          <div style={{
            background: '#ffffff',
            borderRadius: '8px',
            padding: '32px',
            maxWidth: '400px',
            width: '90%',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
          }}>
            <h2 style={{
              fontSize: '20px',
              fontWeight: '600',
              color: '#191919',
              marginBottom: '12px',
            }}>
              Confirm Logout
            </h2>
            <p style={{
              fontSize: '14px',
              color: '#9fadbd',
              marginBottom: '24px',
              lineHeight: '1.5',
            }}>
              Are you sure you want to log out? Any unsaved changes will be lost.
            </p>
            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'flex-end',
            }}>
              <button
                onClick={() => setShowLogoutModal(false)}
                style={{
                  padding: '10px 20px',
                  background: '#efefef',
                  color: '#191919',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#e0e0e0';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#efefef';
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowLogoutModal(false);
                  handleLogout();
                }}
                style={{
                  padding: '10px 20px',
                  background: '#b68947',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#a07839';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#b68947';
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      <SettingsModal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} />
    </div>
  );
}
