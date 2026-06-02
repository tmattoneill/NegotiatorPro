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
          background: '#ffffff',
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
            color: '#191919',
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
              color: '#191919',
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
                color: '#9fadbd',
                fontSize: '14px',
              }}
            >
              Loading providers...
            </div>
          ) : modelsError ? (
            <div
              style={{
                padding: '12px',
                background: '#fff5f5',
                border: '1px solid #fc8181',
                borderRadius: '6px',
                color: '#c53030',
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
                border: '1px solid #9fadbd',
                borderRadius: '6px',
                background: '#ffffff',
                color: '#191919',
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
                  color: '#191919',
                  marginBottom: '2px',
                }}
              >
                Optimize Text
              </div>
              <div
                style={{
                  fontSize: '13px',
                  color: '#9fadbd',
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
              background: '#3498db',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#2980b9';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = '#3498db';
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
