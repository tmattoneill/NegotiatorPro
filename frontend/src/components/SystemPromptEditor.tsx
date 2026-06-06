/**
 * SystemPromptEditor Component
 *
 * Admin-only full-page editor for the system prompt with backup/restore support.
 */

import { useState, useEffect } from 'react';
import api from '../services/api';
import { useAdminStore } from '../store/adminStore';
import { FaArrowLeft, FaHistory, FaSpinner } from 'react-icons/fa';

interface BackupInfo {
  filename: string;
  timestamp: string;
  size: number;
}

export default function SystemPromptEditor() {
  const setView = useAdminStore((state) => state.setView);
  const [content, setContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [lastModified, setLastModified] = useState<string | null>(null);
  const [backups, setBackups] = useState<BackupInfo[]>([]);
  const [showBackups, setShowBackups] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    fetchSystemPrompt();
    fetchBackups();
  }, []);

  const fetchSystemPrompt = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/admin/system-prompt');
      setContent(response.data.content || '');
      setOriginalContent(response.data.content || '');
      setLastModified(response.data.last_modified);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load system prompt');
    } finally {
      setLoading(false);
    }
  };

  const fetchBackups = async () => {
    try {
      const response = await api.get('/admin/system-prompt/backups');
      setBackups(response.data);
    } catch (err) {
      console.error('Failed to load backups:', err);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const response = await api.put('/admin/system-prompt', { content });
      setOriginalContent(content);
      setSuccess(response.data.backup_created
        ? `Saved! Backup created: ${response.data.backup_created}`
        : 'Saved successfully!');
      fetchBackups();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save system prompt');
    } finally {
      setSaving(false);
    }
  };

  const handleRestore = async (filename: string) => {
    if (!confirm(`Restore from ${filename}? Current prompt will be backed up.`)) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await api.post(`/admin/system-prompt/restore/${filename}`);
      setSuccess(`Restored from ${filename}`);
      fetchSystemPrompt();
      fetchBackups();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to restore backup');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (hasChanges && !confirm('You have unsaved changes. Discard them?')) {
      return;
    }
    setView('none');
  };

  const hasChanges = content !== originalContent;

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--color-surface)',
      height: '100vh',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 32px',
        background: 'var(--color-background)',
        borderBottom: '1px solid var(--color-border)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={handleClose}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--color-muted)',
                fontSize: '18px',
                cursor: 'pointer',
                padding: '4px 8px',
                display: 'flex',
                alignItems: 'center',
              }}
              title="Back to Chat"
            >
              <FaArrowLeft />
            </button>
            <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 600, color: 'var(--color-heading)' }}>
              Amfonica Meta Prompt
            </h1>
          </div>
          <div style={{ fontSize: '13px', color: 'var(--color-muted)', marginTop: '4px', marginLeft: '42px' }}>
            Cross-persona identity & output rules. Sales and negotiation personae live in <code>prompts/*.yaml</code> and are edited in the codebase, not here.
            {lastModified && <> &middot; Last modified: {new Date(lastModified).toLocaleString()}</>}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button
            onClick={() => setShowBackups(!showBackups)}
            style={{
              background: showBackups ? 'var(--color-pop-10)' : 'var(--color-background)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              padding: '10px 16px',
              color: showBackups ? 'var(--color-pop)' : 'var(--color-body)',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <FaHistory />
            Backups ({backups.length})
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || saving}
            style={{
              background: hasChanges ? 'var(--color-pop)' : 'var(--color-border)',
              border: 'none',
              borderRadius: '8px',
              padding: '10px 24px',
              color: hasChanges ? 'white' : 'var(--color-muted)',
              fontSize: '14px',
              fontWeight: 500,
              cursor: hasChanges ? 'pointer' : 'not-allowed',
            }}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Messages */}
      {(error || success) && (
        <div style={{ padding: '0 32px' }}>
          {error && (
            <div style={{
              marginTop: '16px',
              padding: '12px 16px',
              background: 'rgba(220, 53, 69, 0.1)',
              border: '1px solid rgba(220, 53, 69, 0.3)',
              borderRadius: '8px',
              color: '#dc3545',
              fontSize: '14px',
            }}>
              {error}
            </div>
          )}
          {success && (
            <div style={{
              marginTop: '16px',
              padding: '12px 16px',
              background: 'rgba(40, 167, 69, 0.1)',
              border: '1px solid rgba(40, 167, 69, 0.3)',
              borderRadius: '8px',
              color: '#28a745',
              fontSize: '14px',
            }}>
              {success}
            </div>
          )}
        </div>
      )}

      {/* Content Area */}
      <div style={{
        flex: 1,
        display: 'flex',
        padding: '24px 32px',
        gap: '24px',
        overflow: 'hidden',
      }}>
        {/* Editor */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--color-background)',
          borderRadius: '12px',
          border: '1px solid var(--color-border)',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '12px 16px',
            borderBottom: '1px solid var(--color-border)',
            fontSize: '12px',
            color: 'var(--color-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            fontWeight: 600,
          }}>
            Prompt Content (Markdown)
          </div>
          {loading ? (
            <div style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--color-muted)',
            }}>
              <FaSpinner className="animate-spin" style={{ marginRight: '8px' }} />
              Loading...
            </div>
          ) : (
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter the system prompt here. This prompt is sent to the LLM at the beginning of every conversation to define its behavior and personality."
              style={{
                flex: 1,
                width: '100%',
                padding: '20px',
                fontSize: '14px',
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
                lineHeight: '1.7',
                background: 'transparent',
                border: 'none',
                color: 'var(--color-body)',
                resize: 'none',
                outline: 'none',
              }}
            />
          )}
        </div>

        {/* Backups Panel */}
        {showBackups && (
          <div style={{
            width: '320px',
            background: 'var(--color-background)',
            borderRadius: '12px',
            border: '1px solid var(--color-border)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--color-border)',
              fontSize: '12px',
              color: 'var(--color-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              fontWeight: 600,
            }}>
              Previous Versions
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
              {backups.length === 0 ? (
                <div style={{
                  padding: '24px',
                  textAlign: 'center',
                  color: 'var(--color-muted)',
                  fontSize: '14px',
                }}>
                  No backups yet. Backups are created automatically when you save.
                </div>
              ) : (
                backups.map((backup) => (
                  <div
                    key={backup.filename}
                    style={{
                      padding: '14px 16px',
                      background: 'var(--color-surface)',
                      borderRadius: '8px',
                      marginBottom: '8px',
                    }}
                  >
                    <div style={{
                      fontSize: '14px',
                      color: 'var(--color-heading)',
                      marginBottom: '6px',
                      fontWeight: 500,
                    }}>
                      {new Date(backup.timestamp).toLocaleString()}
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: 'var(--color-muted)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}>
                      <span>{(backup.size / 1024).toFixed(1)} KB</span>
                      <button
                        onClick={() => handleRestore(backup.filename)}
                        style={{
                          background: 'transparent',
                          border: '1px solid var(--color-pop)',
                          borderRadius: '4px',
                          color: 'var(--color-pop)',
                          fontSize: '12px',
                          cursor: 'pointer',
                          padding: '4px 10px',
                        }}
                      >
                        Restore
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
