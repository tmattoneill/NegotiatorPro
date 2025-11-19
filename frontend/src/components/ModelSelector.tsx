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

  return (
    <div className="model-selector">
      {/* Provider Dropdown */}
      <div className="selector-group">
        <label htmlFor="provider-select">Provider</label>
        <select
          id="provider-select"
          value={selectedProvider || ''}
          onChange={(e) => setProvider(e.target.value)}
          className="selector-dropdown"
        >
          {availableBackends.map((backendId) => (
            <option key={backendId} value={backendId}>
              {availableModels[backendId].name}
            </option>
          ))}
        </select>
      </div>

      {/* Model Dropdown */}
      <div className="selector-group">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={selectedModel || ''}
          onChange={(e) => setModel(e.target.value)}
          className="selector-dropdown"
          disabled={currentProviderModels.length === 0}
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
      </div>

      {/* Model Description */}
      {selectedModel && currentProviderModels.length > 0 && (
        <div className="model-description">
          {currentProviderModels.find((m) => m.id === selectedModel)?.description}
        </div>
      )}
    </div>
  );
}
