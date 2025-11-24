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
import NegotiationModal from './NegotiationModal';
import { useAdminStore } from '../store/adminStore';

export default function Sidebar() {
  const navigate = useNavigate();
  const { sessions, currentSessionId, createNewSession, switchSession, loadConversations, deleteSession, renameSession } = useChatStore();
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
  const [showNegotiationModal, setShowNegotiationModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
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

  const handleDeleteClick = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation(); // Prevent switching to this session
    setSessionToDelete(sessionId);
    setShowDeleteModal(true);
  };

  const confirmDelete = async () => {
    if (sessionToDelete && user?.id) {
      try {
        await deleteSession(sessionToDelete, user.id);
        setShowDeleteModal(false);
        setSessionToDelete(null);
      } catch (error) {
        console.error('Failed to delete conversation:', error);
        alert('Failed to delete conversation. Please try again.');
      }
    }
  };

  const cancelDelete = () => {
    setShowDeleteModal(false);
    setSessionToDelete(null);
  };

  const startEditing = (e: React.MouseEvent, sessionId: string, currentTitle: string) => {
    e.stopPropagation(); // Prevent switching to this session
    setEditingSessionId(sessionId);
    setEditingTitle(currentTitle);
  };

  const handleRenameSubmit = async (sessionId: string) => {
    if (editingTitle.trim() && user?.id) {
      try {
        await renameSession(sessionId, user.id, editingTitle);
        setEditingSessionId(null);
        setEditingTitle('');
      } catch (error) {
        console.error('Failed to rename conversation:', error);
        alert('Failed to rename conversation. Please try again.');
      }
    } else {
      // If empty, cancel the edit
      setEditingSessionId(null);
      setEditingTitle('');
    }
  };

  const cancelEditing = () => {
    setEditingSessionId(null);
    setEditingTitle('');
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
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px'
        }}>
          <div style={{ fontSize: '10px', color: '#9fadbd', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Active Negotiation
          </div>
          <button
            onClick={() => setShowNegotiationModal(true)}
            style={{
              padding: '4px 8px',
              background: 'rgba(182, 137, 71, 0.2)',
              border: '1px solid rgba(182, 137, 71, 0.5)',
              borderRadius: '4px',
              color: '#b68947',
              fontSize: '11px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
            title="Create new negotiation"
          >
            + New
          </button>
        </div>
        {negotiations.length > 0 ? (
          <div className="relative">
            <select
              value={currentNegotiation?.id || ''}
              onChange={(e) => {
                const selectedId = e.target.value;
                console.log('Negotiation dropdown changed to:', selectedId);
                console.log('Available negotiations:', negotiations);
                setCurrentNegotiation(selectedId);
                console.log('After setCurrentNegotiation, currentNegotiationId should be:', selectedId);
              }}
              className="np-select w-full appearance-none bg-none px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white text-[13px] pr-8 cursor-pointer"
            >
              <option value="">Select a negotiation...</option>
              {negotiations.map((neg) => (
                <option key={neg.id} value={neg.id}>
                  {neg.title || neg.name || 'Untitled Negotiation'}
                </option>
              ))}
            </select>
            {/* Chevron indicator */}
            <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-white/70">
              <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.08 1.04l-4.25 4.25a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
              </svg>
            </span>
          </div>
        ) : (
          <div style={{ fontSize: '13px', color: '#fff', opacity: 0.7, fontStyle: 'italic', lineHeight: '1.4' }}>
            <div style={{ marginBottom: '4px' }}>
              Welcome! Click "+ New" above to start your first negotiation.
            </div>
          </div>
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
        disabled={!currentNegotiation?.id || !user?.id}
        title={!currentNegotiation?.id ? 'Loading...' : 'Start a new conversation'}
      >
        + New Conversation
      </button>

      <div className="sessions-list">
        <h3 style={{ fontSize: '12px', color: '#6c757d', padding: '8px 16px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Chat Sessions
        </h3>
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
            onClick={() => editingSessionId !== session.id && switchSession(session.id, user?.id)}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              {editingSessionId === session.id ? (
                <input
                  type="text"
                  value={editingTitle}
                  onChange={(e) => setEditingTitle(e.target.value.slice(0, 64))}
                  onBlur={() => handleRenameSubmit(session.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleRenameSubmit(session.id);
                    } else if (e.key === 'Escape') {
                      cancelEditing();
                    }
                  }}
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                  maxLength={64}
                  style={{
                    width: '100%',
                    fontSize: '14px',
                    fontWeight: 500,
                    background: 'rgba(255, 255, 255, 0.1)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    color: '#fff',
                    outline: 'none',
                  }}
                />
              ) : (
                <>
                  <div
                    style={{ fontSize: '14px', fontWeight: 500, marginBottom: '4px', cursor: 'pointer' }}
                    onClick={(e) => startEditing(e, session.id, session.title || 'New Conversation')}
                    title="Click to rename"
                  >
                    {session.title || 'New Conversation'}
                  </div>
                  <div style={{ fontSize: '11px', opacity: 0.7 }}>
                    {session.messageCount} messages • {new Date(session.createdAt).toLocaleDateString()}
                  </div>
                </>
              )}
            </div>
            {editingSessionId !== session.id && (
              <button
                onClick={(e) => handleDeleteClick(e, session.id)}
                className="delete-session-btn"
                title="Delete conversation"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#ff6b6b',
                  cursor: 'pointer',
                  padding: '6px',
                  marginLeft: '8px',
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  opacity: 0.6,
                  transition: 'opacity 0.2s, background 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '1';
                  e.currentTarget.style.background = 'rgba(255, 107, 107, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = '0.6';
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                <i className="fa-light fa-trash" style={{ fontSize: '14px' }}></i>
              </button>
            )}
          </div>
        ))}

        {sessions.length === 0 && (
          <div style={{ padding: '16px', textAlign: 'center', color: '#6c757d', fontSize: '13px' }}>
            No chat sessions yet. Click "New Conversation" to start.
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => setView(currentView === 'admin-panel' ? 'none' : 'admin-panel')}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: currentView === 'admin-panel' ? 'rgba(182, 137, 71, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                border: currentView === 'admin-panel' ? '1px solid rgba(182, 137, 71, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
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
                if (currentView !== 'admin-panel') {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                }
              }}
              onMouseOut={(e) => {
                if (currentView !== 'admin-panel') {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                }
              }}
            >
              <i className="fa-light fa-shield-halved" style={{ opacity: 0.7 }}></i>
              Admin Panel
            </button>
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
                if (currentView !== 'system-prompt') {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                }
              }}
              onMouseOut={(e) => {
                if (currentView !== 'system-prompt') {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                }
              }}
            >
              <i className="fa-light fa-terminal" style={{ opacity: 0.7 }}></i>
              System Prompt
            </button>
          </div>
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

      {/* Delete Conversation Confirmation Modal */}
      {showDeleteModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: '#1e2936',
            padding: '32px',
            borderRadius: '12px',
            maxWidth: '400px',
            width: '90%',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}>
            <h2 style={{
              fontSize: '20px',
              fontWeight: '600',
              color: '#ffffff',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <i className="fa-light fa-trash" style={{ color: '#ff6b6b' }}></i>
              Delete Conversation?
            </h2>
            <p style={{
              fontSize: '14px',
              color: '#9fadbd',
              marginBottom: '24px',
              lineHeight: '1.5',
            }}>
              Are you sure you want to delete this conversation? This action cannot be undone and all messages will be permanently removed.
            </p>
            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'flex-end',
            }}>
              <button
                onClick={cancelDelete}
                style={{
                  padding: '10px 20px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  color: '#ffffff',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: '10px 20px',
                  background: '#ff6b6b',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#ff5252';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#ff6b6b';
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      <SettingsModal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} />

      {/* Negotiation Modal */}
      <NegotiationModal isOpen={showNegotiationModal} onClose={() => setShowNegotiationModal(false)} />
    </div>
  );
}
