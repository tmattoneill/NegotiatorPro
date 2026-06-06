/**
 * Sidebar component with session management
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/chatStore';
import { useAuthStore } from '../store/authStore';
import { usePersonaStore } from '../store/personaStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { useSettingsStore } from '../store/settingsStore';
import SettingsModal from './SettingsModal';
import ModelSelector from './ModelSelector';
import NegotiationModal from './NegotiationModal';
import PartnerPersonaModal from './PartnerPersonaModal';
import Portal from './Portal';
import { useAdminStore } from '../store/adminStore';
import { FaRegUser, FaHandshake, FaFlask, FaShieldAlt, FaTerminal, FaCog, FaSignOutAlt, FaTrash, FaComments, FaPen } from 'react-icons/fa';

export default function Sidebar() {
  const navigate = useNavigate();
  const { sessions, currentSessionId, createNewSession, switchSession, loadConversations, deleteSession, renameSession } = useChatStore();
  const { user, logout } = useAuthStore();
  const { userPersonas, fetchUserPersonas, fetchPartnerPersonas } = usePersonaStore();
  const { negotiations, loadNegotiations, setCurrentNegotiation, currentNegotiationId, updateNegotiation } = useNegotiationStore();
  const { availableModels, setProviderModel } = useSettingsStore();
  // Compute currentNegotiation from negotiations and currentNegotiationId
  const currentNegotiation = negotiations.find(n => n.id === currentNegotiationId) || null;

  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showNegotiationModal, setShowNegotiationModal] = useState(false);
  const [showPartnerModal, setShowPartnerModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const { currentView, setView } = useAdminStore();

  // Check if user is admin or super admin
  const isAdmin = user?.role === 'admin';
  const isSuperAdmin = user?.is_super_admin === true;

  // Fetch personas and negotiations on mount (skip for super admin)
  useEffect(() => {
    if (user?.id && !isSuperAdmin) {
      fetchUserPersonas(user.id);
      fetchPartnerPersonas(user.id);
      loadNegotiations(user.id);
    }
  }, [user?.id, isSuperAdmin]);

  // Load conversations when negotiation changes
  useEffect(() => {
    if (currentNegotiation?.id && user?.id) {
      loadConversations(currentNegotiation.id, user.id);
    }
  }, [currentNegotiation?.id, user?.id]);

  // Resolve a preferred provider/model against the available list. Keeps the
  // preferred model only if that provider actually offers it, else falls back to
  // the provider's first model. Returns null when the provider isn't available.
  const resolvePreferredModel = (
    provider?: string | null,
    model?: string | null,
  ): { provider: string; model: string | null } | null => {
    if (!provider || !availableModels[provider]) return null;
    const models = availableModels[provider].models;
    const keep = model && models.some((m) => m.id === model);
    return { provider, model: keep ? model! : (models[0]?.id ?? null) };
  };

  // The active negotiation drives the model selector. Precedence on load:
  // negotiation.settings (explicit per-negotiation override) → the bound "You"
  // persona's preferred provider/model → the user's profile default → leave as is.
  const modelKeyCount = Object.keys(availableModels).length;
  useEffect(() => {
    if (!currentNegotiation?.id || modelKeyCount === 0) return;
    const settings = (currentNegotiation.settings ?? {}) as { provider?: string; model?: string };
    if (settings.provider && availableModels[settings.provider]) {
      setProviderModel(settings.provider, settings.model ?? null);
      return;
    }
    const persona = userPersonas.find((p) => p.id === currentNegotiation.user_persona_id);
    const fromPersona = resolvePreferredModel(persona?.preferred_provider, persona?.preferred_model);
    if (fromPersona) {
      setProviderModel(fromPersona.provider, fromPersona.model);
      return;
    }
    const fromUser = resolvePreferredModel(
      (user as { preferred_provider?: string | null })?.preferred_provider,
      (user as { preferred_model?: string | null })?.preferred_model,
    );
    if (fromUser) {
      setProviderModel(fromUser.provider, fromUser.model);
    }
    // else: leave the current selection untouched
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentNegotiation?.id, currentNegotiation?.user_persona_id, modelKeyCount, userPersonas.length]);

  // Get active personas. The "You" persona is the one BOUND to the current
  // negotiation (user_persona_id) — that is what the server-side briefing uses
  // to build the prompt. Showing the global default here would contradict the
  // actual prompt when the two differ.
  const activeUserPersona =
    userPersonas.find(p => p.id === currentNegotiation?.user_persona_id)
    || currentNegotiation?.user_persona
    || userPersonas.find(p => p.is_default)
    || userPersonas[0];
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

      {/* Active Personas Display - hidden for super admin */}
      {!isSuperAdmin && (
        <div style={{
          padding: '12px 16px',
          background: 'var(--sidebar-panel-bg)',
          borderBottom: '1px solid var(--sidebar-control-border)',
        }}>
          <div style={{ marginBottom: '10px' }}>
            <div style={{ fontSize: '10px', color: 'var(--sidebar-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
              You
            </div>
            {currentNegotiation && userPersonas.length > 0 ? (
              <div className="relative">
                <FaRegUser style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', opacity: 0.7, pointerEvents: 'none' }} />
                <select
                  value={currentNegotiation.user_persona_id ?? ''}
                  onChange={(e) => {
                    const personaId = e.target.value;
                    if (!user?.id || !currentNegotiation?.id || !personaId) return;
                    // Follow the persona: re-point the model to its preferred
                    // provider/model and persist that onto the negotiation.
                    const persona = userPersonas.find((p) => p.id === personaId);
                    const patch: { user_persona_id: string; settings?: Record<string, unknown> } = {
                      user_persona_id: personaId,
                    };
                    const picked = resolvePreferredModel(persona?.preferred_provider, persona?.preferred_model);
                    if (picked) {
                      setProviderModel(picked.provider, picked.model);
                      patch.settings = {
                        ...(currentNegotiation.settings ?? {}),
                        provider: picked.provider,
                        model: picked.model,
                      };
                    }
                    updateNegotiation(user.id, currentNegotiation.id, patch);
                  }}
                  title="Who you are in this negotiation — used to tailor advice"
                  className="np-select w-full appearance-none bg-none py-2 bg-black/30 border border-white/20 rounded-md text-white text-[13px] cursor-pointer"
                  style={{ paddingLeft: '30px', paddingRight: '28px' }}
                >
                  {!currentNegotiation.user_persona_id && (
                    <option value="">No persona set</option>
                  )}
                  {userPersonas.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}{p.role_title ? ` · ${p.role_title}` : ''}
                    </option>
                  ))}
                </select>
                <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-white/70">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.08 1.04l-4.25 4.25a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                  </svg>
                </span>
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: 'var(--sidebar-fg)', fontWeight: 500 }}>
                {activeUserPersona ? (
                  <>
                    <FaRegUser style={{ marginRight: '6px', opacity: 0.7, display: 'inline' }} />
                    {activeUserPersona.name}
                    {activeUserPersona.role_title && (
                      <span style={{ opacity: 0.6, fontWeight: 400 }}> · {activeUserPersona.role_title}</span>
                    )}
                  </>
                ) : (
                  <span style={{ opacity: 0.5, fontStyle: 'italic' }}>No persona set</span>
                )}
              </div>
            )}
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--sidebar-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
              Partner
            </div>
            {activePartnerPersona ? (
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <div style={{ flex: 1, minWidth: 0, fontSize: '13px', color: 'var(--sidebar-fg)', fontWeight: 500 }}>
                  <FaHandshake style={{ marginRight: '6px', opacity: 0.7, display: 'inline' }} />
                  {activePartnerPersona.name}
                  {activePartnerPersona.company && (
                    <span style={{ opacity: 0.6, fontWeight: 400 }}> · {activePartnerPersona.company}</span>
                  )}
                </div>
                <button
                  onClick={() => setShowPartnerModal(true)}
                  title="Edit this negotiation's partner"
                  className="shrink-0 rounded-md border border-white/20 bg-black/30 px-2 py-2 text-white/80 hover:bg-white/10"
                >
                  <FaPen style={{ fontSize: '11px' }} />
                </button>
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: 'var(--sidebar-fg)', fontWeight: 500 }}>
                <span style={{ opacity: 0.5, fontStyle: 'italic' }}>Select a negotiation</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Negotiation Selector - hidden for super admin */}
      {!isSuperAdmin && (
        <div style={{
          padding: '12px 16px',
          background: 'var(--sidebar-panel-bg)',
          borderBottom: '1px solid var(--sidebar-control-border)',
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '8px'
          }}>
            <div style={{ fontSize: '10px', color: 'var(--sidebar-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Active Negotiation
            </div>
            <button
              onClick={() => setShowNegotiationModal(true)}
              style={{
                padding: '4px 8px',
                background: 'var(--color-pop-20)',
                border: '1px solid var(--color-pop-50)',
                borderRadius: '4px',
                color: 'var(--color-pop)',
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
                onChange={(e) => setCurrentNegotiation(e.target.value)}
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
            <div style={{ fontSize: '13px', color: 'var(--sidebar-fg)', opacity: 0.7, fontStyle: 'italic', lineHeight: '1.4' }}>
              <div style={{ marginBottom: '4px' }}>
                Welcome! Click "+ New" above to start your first negotiation.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Selector - shows models from selected provider */}
      <div style={{ padding: '16px 16px 0 16px' }}>
        <ModelSelector />
      </div>

      {/* Separator */}
      <div className="sidebar-separator"></div>

      {/* Conversations section - hidden for super admin */}
      {!isSuperAdmin && (
        <>
          <button
            className="new-session-btn"
            onClick={() => currentNegotiation?.id && user?.id && createNewSession(currentNegotiation.id, user.id)}
            disabled={!currentNegotiation?.id || !user?.id}
            title={!currentNegotiation?.id ? 'Loading...' : 'Start a new conversation'}
          >
            + New Conversation
          </button>

          <div className="sessions-list">
            <h3 style={{ fontSize: '12px', color: 'var(--sidebar-muted)', padding: '8px 16px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
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
                        background: 'var(--sidebar-control-bg)',
                        border: '1px solid var(--sidebar-control-border)',
                        borderRadius: '4px',
                        padding: '4px 8px',
                        color: 'var(--sidebar-fg)',
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
                      color: 'var(--color-pop)',
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
                      e.currentTarget.style.background = 'var(--color-pop-10)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.opacity = '0.6';
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <FaTrash style={{ fontSize: '14px' }} />
                  </button>
                )}
              </div>
            ))}

            {sessions.length === 0 && (
              <div style={{ padding: '16px', textAlign: 'center', color: 'var(--sidebar-muted)', fontSize: '13px' }}>
                No chat sessions yet. Click "New Conversation" to start.
              </div>
            )}
          </div>
        </>
      )}

      {/* Super Admin Testing Mode Notice */}
      {isSuperAdmin && (
        <div style={{
          padding: '16px',
          margin: '16px',
          background: 'var(--color-pop-10)',
          border: '1px solid var(--color-pop-30)',
          borderRadius: '8px',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '11px', color: 'var(--color-pop)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
            <FaFlask style={{ marginRight: '6px' }} />
            Testing Mode
          </div>
          <div style={{ fontSize: '12px', color: 'var(--sidebar-muted)', lineHeight: '1.4' }}>
            Chat with models to test. Conversations are not saved.
          </div>
        </div>
      )}

      {/* Admin Section - only visible to admin users */}
      {isAdmin && (
        <div style={{
          padding: '16px',
          borderTop: '1px solid var(--sidebar-control-border)',
        }}>
          <div style={{
            fontSize: '10px',
            color: 'var(--sidebar-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <FaShieldAlt />
            Admin
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => setView('none')}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: currentView === 'none' ? 'var(--color-pop-20)' : 'var(--sidebar-panel-bg)',
                border: currentView === 'none' ? '1px solid var(--color-pop-50)' : '1px solid var(--sidebar-control-border)',
                borderRadius: '6px',
                color: 'var(--sidebar-fg)',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                textAlign: 'left',
                transition: 'all 0.2s',
              }}
              onMouseOver={(e) => {
                if (currentView !== 'none') e.currentTarget.style.background = 'var(--sidebar-control-bg)';
              }}
              onMouseOut={(e) => {
                if (currentView !== 'none') e.currentTarget.style.background = 'var(--sidebar-panel-bg)';
              }}
            >
              <FaComments style={{ opacity: 0.7 }} />
              Chat
            </button>
            <button
              onClick={() => setView(currentView === 'admin-panel' ? 'none' : 'admin-panel')}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: currentView === 'admin-panel' ? 'var(--color-pop-20)' : 'var(--sidebar-panel-bg)',
                border: currentView === 'admin-panel' ? '1px solid var(--color-pop-50)' : '1px solid var(--sidebar-control-border)',
                borderRadius: '6px',
                color: 'var(--sidebar-fg)',
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
                  e.currentTarget.style.background = 'var(--sidebar-control-bg)';
                }
              }}
              onMouseOut={(e) => {
                if (currentView !== 'admin-panel') {
                  e.currentTarget.style.background = 'var(--sidebar-panel-bg)';
                }
              }}
            >
              <FaShieldAlt style={{ opacity: 0.7 }} />
              Admin Panel
            </button>
            <button
              onClick={() => setView(currentView === 'system-prompt' ? 'none' : 'system-prompt')}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: currentView === 'system-prompt' ? 'var(--color-pop-20)' : 'var(--sidebar-panel-bg)',
                border: currentView === 'system-prompt' ? '1px solid var(--color-pop-50)' : '1px solid var(--sidebar-control-border)',
                borderRadius: '6px',
                color: 'var(--sidebar-fg)',
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
                  e.currentTarget.style.background = 'var(--sidebar-control-bg)';
                }
              }}
              onMouseOut={(e) => {
                if (currentView !== 'system-prompt') {
                  e.currentTarget.style.background = 'var(--sidebar-panel-bg)';
                }
              }}
            >
              <FaTerminal style={{ opacity: 0.7 }} />
              System Prompt
            </button>
          </div>
        </div>
      )}

      {/* User info and actions */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid var(--sidebar-control-border)',
        marginTop: 'auto'
      }}>
        <div style={{
          fontSize: '13px',
          color: 'var(--sidebar-muted)',
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
              background: 'var(--sidebar-control-bg)',
              color: 'var(--sidebar-fg)',
              border: '1px solid var(--sidebar-control-border)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg-hover)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <FaRegUser />
          </button>

          {/* Settings Icon */}
          <button
            onClick={() => setShowSettingsModal(true)}
            title="Settings"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'var(--sidebar-control-bg)',
              color: 'var(--sidebar-fg)',
              border: '1px solid var(--sidebar-control-border)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg-hover)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <FaCog />
          </button>

          {/* Logout Icon */}
          <button
            onClick={() => setShowLogoutModal(true)}
            title="Logout"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'var(--sidebar-control-bg)',
              color: 'var(--sidebar-fg)',
              border: '1px solid var(--sidebar-control-border)',
              fontSize: '18px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg-hover)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--sidebar-control-bg)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            <FaSignOutAlt />
          </button>
        </div>
      </div>

      {/* Logout Confirmation Modal */}
      {showLogoutModal && (
        <Portal>
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
            background: 'var(--color-surface)',
            borderRadius: '8px',
            padding: '32px',
            maxWidth: '400px',
            width: '90%',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
          }}>
            <h2 style={{
              fontSize: '20px',
              fontWeight: '600',
              color: 'var(--color-heading)',
              marginBottom: '12px',
            }}>
              Confirm Logout
            </h2>
            <p style={{
              fontSize: '14px',
              color: 'var(--color-muted)',
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
                  background: 'var(--color-border)',
                  color: 'var(--color-heading)',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'var(--color-muted)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'var(--color-border)';
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
                  background: 'var(--color-pop)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'var(--color-pop-hover)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'var(--color-pop)';
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
        </Portal>
      )}

      {/* Delete Conversation Confirmation Modal */}
      {showDeleteModal && (
        <Portal>
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
            background: 'var(--color-surface)',
            padding: '32px',
            borderRadius: '12px',
            maxWidth: '400px',
            width: '90%',
            border: '1px solid var(--color-border)',
          }}>
            <h2 style={{
              fontSize: '20px',
              fontWeight: '600',
              color: 'var(--color-heading)',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <FaTrash style={{ color: 'var(--color-pop)' }} />
              Delete Conversation?
            </h2>
            <p style={{
              fontSize: '14px',
              color: 'var(--color-muted)',
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
                  background: 'var(--color-border)',
                  color: 'var(--color-heading)',
                  border: '1px solid var(--color-muted)',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'var(--color-muted)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'var(--color-border)';
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: '10px 20px',
                  background: 'var(--color-pop)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'var(--color-pop-hover)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'var(--color-pop)';
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
        </Portal>
      )}

      {/* Settings Modal */}
      <SettingsModal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} />

      {/* Negotiation Modal */}
      <NegotiationModal isOpen={showNegotiationModal} onClose={() => setShowNegotiationModal(false)} />

      {/* Partner Persona Edit Modal */}
      <PartnerPersonaModal
        isOpen={showPartnerModal}
        partner={activePartnerPersona ?? null}
        negotiationId={currentNegotiation?.id ?? null}
        onClose={() => setShowPartnerModal(false)}
        onSaved={() => {
          setShowPartnerModal(false);
          if (user?.id) loadNegotiations(user.id);
        }}
      />
    </div>
  );
}
