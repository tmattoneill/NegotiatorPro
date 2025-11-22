/**
 * Sidebar component with session management
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/chatStore';
import { useAuthStore } from '../store/authStore';
import { usePersonaStore } from '../store/personaStore';
import { useNegotiationStore } from '../store/negotiationStore';
import SettingsModal from './SettingsModal';
import ModelSelector from './ModelSelector';
import { useAdminStore } from '../store/adminStore';

export default function Sidebar() {
  const navigate = useNavigate();
  const { sessions, currentSessionId, createNewSession, switchSession, loadConversations } = useChatStore();
  const { user, logout } = useAuthStore();
  const { userPersonas, partnerPersonas: _partnerPersonas, fetchUserPersonas, fetchPartnerPersonas } = usePersonaStore();
  void _partnerPersonas; // Available for future use
  const { negotiations, loadNegotiations, setCurrentNegotiation, currentNegotiationId } = useNegotiationStore();
  // Compute currentNegotiation from negotiations and currentNegotiationId
  const currentNegotiation = negotiations.find(n => n.id === currentNegotiationId) || null;

  // Debug: log when currentNegotiationId changes
  useEffect(() => {
    console.log('Current Negotiation ID changed:', currentNegotiationId);
    console.log('Current Negotiation object:', currentNegotiation);
  }, [currentNegotiationId, currentNegotiation]);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const { currentView, setView } = useAdminStore();

  // Check if user is admin
  const isAdmin = user?.role === 'admin';

  // Fetch personas and negotiations on mount
  useEffect(() => {
    if (user?.id) {
      fetchUserPersonas(user.id);
      fetchPartnerPersonas(user.id);
      loadNegotiations(user.id);
    }
  }, [user?.id]);

  // Load conversations when negotiation changes
  useEffect(() => {
    console.log('Negotiation changed:', currentNegotiation?.id);
    if (currentNegotiation?.id && user?.id) {
      console.log('Loading conversations for negotiation:', currentNegotiation.id);
      loadConversations(currentNegotiation.id, user.id);
    }
  }, [currentNegotiation?.id, user?.id]);

  // Get active personas
  const activeUserPersona = userPersonas.find(p => p.is_default) || userPersonas[0];
  const activePartnerPersona = currentNegotiation?.partners?.[0];

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

      {/* Active Personas Display */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(255, 255, 255, 0.05)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      }}>
        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '10px', color: '#9fadbd', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
            You
          </div>
          <div style={{ fontSize: '13px', color: '#fff', fontWeight: 500 }}>
            {activeUserPersona ? (
              <>
                <i className="fa-light fa-user" style={{ marginRight: '6px', opacity: 0.7 }}></i>
                {activeUserPersona.name}
                {activeUserPersona.role_title && (
                  <span style={{ opacity: 0.6, fontWeight: 400 }}> · {activeUserPersona.role_title}</span>
                )}
              </>
            ) : (
              <span style={{ opacity: 0.5, fontStyle: 'italic' }}>No persona set</span>
            )}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '10px', color: '#9fadbd', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
            Partner
          </div>
          <div style={{ fontSize: '13px', color: '#fff', fontWeight: 500 }}>
            {activePartnerPersona ? (
              <>
                <i className="fa-light fa-handshake" style={{ marginRight: '6px', opacity: 0.7 }}></i>
                {activePartnerPersona.name}
                {activePartnerPersona.company && (
                  <span style={{ opacity: 0.6, fontWeight: 400 }}> · {activePartnerPersona.company}</span>
                )}
              </>
            ) : (
              <span style={{ opacity: 0.5, fontStyle: 'italic' }}>Select a negotiation</span>
            )}
          </div>
        </div>
      </div>

      {/* Negotiation Selector */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(255, 255, 255, 0.05)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      }}>
        <div style={{ fontSize: '10px', color: '#9fadbd', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px' }}>
          Active Negotiation
        </div>
        {negotiations.length > 0 ? (
          <select
            value={currentNegotiation?.id || ''}
            onChange={(e) => {
              const selectedId = e.target.value;
              console.log('Negotiation dropdown changed to:', selectedId);
              console.log('Available negotiations:', negotiations);
              setCurrentNegotiation(selectedId);
              console.log('After setCurrentNegotiation, currentNegotiationId should be:', selectedId);
            }}
            style={{
              width: '100%',
              padding: '8px 10px',
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '13px',
              cursor: 'pointer',
            }}
          >
            <option value="">Select a negotiation...</option>
            {negotiations.map((neg) => (
              <option key={neg.id} value={neg.id}>
                {neg.title || neg.name || 'Untitled Negotiation'}
              </option>
            ))}
          </select>
        ) : (
          <div style={{ fontSize: '13px', color: '#fff', opacity: 0.5, fontStyle: 'italic' }}>
            No negotiations yet
          </div>
        )}
        {negotiations.length === 0 && (
          <button
            onClick={() => navigate('/profile')}
            style={{
              marginTop: '8px',
              width: '100%',
              padding: '8px 12px',
              background: 'rgba(182, 137, 71, 0.2)',
              border: '1px solid rgba(182, 137, 71, 0.5)',
              borderRadius: '6px',
              color: '#b68947',
              fontSize: '12px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            + Create Negotiation
          </button>
        )}
      </div>

      {/* Model Selector - shows models from selected provider */}
      <div style={{ padding: '16px 16px 0 16px' }}>
        <ModelSelector />
      </div>

      {/* Separator */}
      <div className="sidebar-separator"></div>

      <button
        className="new-session-btn"
        onClick={() => currentNegotiation?.id && user?.id && createNewSession(currentNegotiation.id, user.id)}
        disabled={!currentNegotiation?.id}
        title={!currentNegotiation?.id ? 'Please select a negotiation first' : 'Create a new conversation'}
      >
        + New Conversation
      </button>

      <div className="sessions-list">
        <h3 style={{ fontSize: '12px', color: '#6c757d', padding: '8px 16px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Negotiation Sessions
        </h3>
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
            onClick={() => user?.id && switchSession(session.id, user.id)}
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
            {currentNegotiation ? 'No conversations yet. Click "New Conversation" to start.' : 'Select a negotiation to view conversations.'}
          </div>
        )}
      </div>

      {/* Admin Section - only visible to admin users */}
      {isAdmin && (
        <div style={{
          padding: '16px',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        }}>
          <div style={{
            fontSize: '10px',
            color: '#9fadbd',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <i className="fa-light fa-shield-halved"></i>
            Admin
          </div>
          <button
            onClick={() => setView(currentView === 'system-prompt' ? 'none' : 'system-prompt')}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: currentView === 'system-prompt' ? 'rgba(182, 137, 71, 0.2)' : 'rgba(255, 255, 255, 0.05)',
              border: currentView === 'system-prompt' ? '1px solid rgba(182, 137, 71, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '13px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              textAlign: 'left',
              transition: 'all 0.2s',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
            }}
          >
            <i className="fa-light fa-terminal" style={{ opacity: 0.7 }}></i>
            System Prompt
          </button>
        </div>
      )}

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
