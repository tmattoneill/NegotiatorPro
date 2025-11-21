/**
 * ModelSelector Component
 *
 * Allows users to select LLM provider and specific model.
 * Dynamically updates model dropdown based on selected provider.
 */

import { useEffect } from 'react';
import { useSettingsStore } from '../store/settingsStore';

export default function ModelSelector() {
  const {
    selectedProvider,
    selectedModel,
    availableModels,
    isLoadingModels,
    modelsError,
    setProvider,
    setModel,
    loadAvailableModels,
  } = useSettingsStore();

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
  }, [loadAvailableModels]);

  // Get models for currently selected provider
  const currentProviderModels = selectedProvider
    ? availableModels[selectedProvider]?.models || []
    : [];

  // Check if current provider has a connection error (e.g., Ollama not running)
  const providerError = selectedProvider
    ? availableModels[selectedProvider]?.error
    : null;

  // Get all available backend IDs
  const availableBackends = Object.keys(availableModels);

  if (isLoadingModels) {
    return (
      <div className="model-selector loading">
        <div className="loading-spinner"></div>
        <span>Loading models...</span>
      </div>
    );
  }

  if (modelsError) {
    return (
      <div className="model-selector error">
        <span className="error-icon">⚠️</span>
        <span>Failed to load models: {modelsError}</span>
      </div>
    );
  }

  if (availableBackends.length === 0) {
    return (
      <div className="model-selector error">
        <span>No LLM backends available</span>
      </div>
    );
  }

  // Get provider name for selected provider
  const providerName = selectedProvider ? availableModels[selectedProvider]?.name : '';

  return (
    <div style={{ marginBottom: '8px' }}>
      {/* Model Dropdown */}
      <div>
        <label
          htmlFor="model-select"
          style={{
            display: 'block',
            fontSize: '11px',
            fontWeight: '600',
            color: '#9fadbd',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          Model ({providerName})
        </label>
        {providerError ? (
          <div
            style={{
              padding: '10px 12px',
              fontSize: '12px',
              background: 'rgba(220, 53, 69, 0.2)',
              border: '1px solid rgba(220, 53, 69, 0.5)',
              borderRadius: '6px',
              color: '#ff6b6b',
            }}
          >
            {providerError}
          </div>
        ) : (
          <select
            id="model-select"
            value={selectedModel || ''}
            onChange={(e) => setModel(e.target.value)}
            disabled={currentProviderModels.length === 0}
            style={{
              width: '100%',
              padding: '8px 12px',
              fontSize: '13px',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              cursor: currentProviderModels.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            {currentProviderModels.length === 0 ? (
              <option value="">No models available</option>
            ) : (
              currentProviderModels.map((model) => (
                <option key={model.id} value={model.id} title={model.description}>
                  {model.name}
                </option>
              ))
            )}
          </select>
        )}
      </div>
    </div>
  );
}
