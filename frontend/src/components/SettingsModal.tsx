/**
 * SettingsModal Component
 *
 * Standalone modal for provider selection and settings.
 */

import { useEffect } from 'react';
import { useSettingsStore } from '../store/settingsStore';
import Portal from './Portal';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const {
    selectedProvider,
    availableModels,
    isLoadingModels,
    modelsError,
    usePreprocessing,
    setProvider,
    setPreprocessing,
    loadAvailableModels,
  } = useSettingsStore();

  // Load available models on component mount
  useEffect(() => {
    if (isOpen) {
      loadAvailableModels();
    }
  }, [isOpen, loadAvailableModels]);

  // Get all available providers
  const availableProviders = Object.entries(availableModels);

  if (!isOpen) return null;

  return (
    <Portal>
    <div
      style={{
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
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--color-surface)',
          borderRadius: '8px',
          padding: '32px',
          maxWidth: '500px',
          width: '90%',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2
          style={{
            fontSize: '24px',
            fontWeight: '600',
            color: 'var(--color-heading)',
            marginBottom: '24px',
          }}
        >
          Settings
        </h2>

        {/* Provider Selection */}
        <div style={{ marginBottom: '24px' }}>
          <label
            htmlFor="provider-select"
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: 'var(--color-heading)',
              marginBottom: '8px',
            }}
          >
            Provider
          </label>

          {isLoadingModels ? (
            <div
              style={{
                padding: '12px',
                textAlign: 'center',
                color: 'var(--color-muted)',
                fontSize: '14px',
              }}
            >
              Loading providers...
            </div>
          ) : modelsError ? (
            <div
              style={{
                padding: '12px',
                background: 'rgba(220, 53, 69, 0.1)',
                border: '1px solid rgba(220, 53, 69, 0.3)',
                borderRadius: '6px',
                color: '#dc3545',
                fontSize: '14px',
              }}
            >
              Failed to load providers: {modelsError}
            </div>
          ) : (
            <select
              id="provider-select"
              value={selectedProvider || ''}
              onChange={(e) => setProvider(e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '14px',
                border: '1px solid var(--color-border)',
                borderRadius: '6px',
                background: 'var(--color-background)',
                color: 'var(--color-body)',
                cursor: 'pointer',
              }}
            >
              {availableProviders.map(([backendId, backend]) => (
                <option key={backendId} value={backendId}>
                  {backend.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Optimize Text Toggle */}
        <div style={{ marginBottom: '32px' }}>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              gap: '12px',
            }}
          >
            <input
              type="checkbox"
              checked={usePreprocessing}
              onChange={(e) => setPreprocessing(e.target.checked)}
              style={{
                width: '20px',
                height: '20px',
                cursor: 'pointer',
              }}
            />
            <div>
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: '500',
                  color: 'var(--color-heading)',
                  marginBottom: '2px',
                }}
              >
                Optimize Text
              </div>
              <div
                style={{
                  fontSize: '13px',
                  color: 'var(--color-muted)',
                }}
              >
                Remove boilerplate to reduce tokens
              </div>
            </div>
          </label>
        </div>

        {/* Close Button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 24px',
              background: 'var(--color-accent)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--color-accent-strong)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--color-accent)';
            }}
          >
            Done
          </button>
        </div>
      </div>
    </div>
    </Portal>
  );
}
