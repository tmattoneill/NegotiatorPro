/**
 * ModelSelector Component
 *
 * Allows users to select LLM provider and specific model.
 * Dynamically updates model dropdown based on selected provider.
 */

import { useEffect } from 'react';
import { useSettingsStore } from '../store/settingsStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { useAuthStore } from '../store/authStore';

export default function ModelSelector() {
  const {
    selectedProvider,
    selectedModel,
    availableModels,
    isLoadingModels,
    modelsError,
    setProviderModel,
    setModel,
    loadAvailableModels,
  } = useSettingsStore();

  const { user } = useAuthStore();
  const { negotiations, currentNegotiationId, updateNegotiation } = useNegotiationStore();
  const currentNegotiation = negotiations.find((n) => n.id === currentNegotiationId) || null;

  // Persist a user-initiated provider/model choice onto the active negotiation
  // so it survives reloads and the backend resolves it (negotiation.settings).
  // Only the onChange handlers call this — the seeding effect in Sidebar uses
  // setProviderModel directly, so there are no spurious writes.
  const persistChoice = (provider: string | null, model: string | null) => {
    if (!user?.id || !currentNegotiation?.id) return;
    updateNegotiation(user.id, currentNegotiation.id, {
      settings: { ...(currentNegotiation.settings ?? {}), provider, model },
    });
  };

  const handleProviderChange = (provider: string) => {
    const firstModel = availableModels[provider]?.models?.[0]?.id ?? null;
    setProviderModel(provider, firstModel);
    persistChoice(provider, firstModel);
  };

  const handleModelChange = (model: string) => {
    setModel(model);
    persistChoice(selectedProvider, model);
  };

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

  const labelStyle = {
    display: 'block',
    fontSize: '11px',
    fontWeight: 600,
    color: '#9fadbd',
    marginBottom: '6px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  };

  return (
    <div style={{ marginBottom: '8px' }}>
      {/* Provider Dropdown */}
      <div style={{ marginBottom: '10px' }}>
        <label htmlFor="provider-select" style={labelStyle}>
          Provider
        </label>
        <div className="relative">
          <select
            id="provider-select"
            value={selectedProvider || ''}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="np-select w-full appearance-none bg-none px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white text-[13px] pr-8 cursor-pointer"
          >
            {availableBackends.map((id) => (
              <option key={id} value={id}>
                {availableModels[id]?.name || id}
              </option>
            ))}
          </select>
          <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-white/70">
            <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
              <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.08 1.04l-4.25 4.25a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
            </svg>
          </span>
        </div>
      </div>

      {/* Model Dropdown */}
      <div>
        <label htmlFor="model-select" style={labelStyle}>
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
          <div className="relative">
            <select
              id="model-select"
              value={selectedModel || ''}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={currentProviderModels.length === 0}
              className={`np-select w-full appearance-none bg-none px-3 py-2 bg-black/30 border border-white/20 rounded-md text-white text-[13px] pr-8 ${currentProviderModels.length === 0 ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
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
            <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-white/70">
              <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.08 1.04l-4.25 4.25a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z" clipRule="evenodd" />
              </svg>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
