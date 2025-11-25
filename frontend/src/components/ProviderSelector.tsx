/**
 * ProviderSelector Component
 *
 * Reusable component for selecting LLM provider and model.
 * Filters providers based on user's available API keys.
 * Can be used in User Settings, Negotiation settings, and Persona settings.
 */

import { useState, useEffect, useCallback } from 'react';
import { useAuthStore } from '../store/authStore';
import { fetchAvailableProviders } from '../services/api';
import type { AvailableProvidersResponse, ProviderInfo, ModelInfo } from '../types';

interface ProviderSelectorProps {
  /** Currently selected provider ID */
  selectedProvider: string | null;
  /** Currently selected model ID */
  selectedModel: string | null;
  /** Callback when provider changes */
  onProviderChange: (providerId: string | null) => void;
  /** Callback when model changes */
  onModelChange: (modelId: string | null) => void;
  /** Optional label for the provider dropdown */
  providerLabel?: string;
  /** Optional label for the model dropdown */
  modelLabel?: string;
  /** Show "Use Default" option */
  showUseDefault?: boolean;
  /** Compact mode for inline use */
  compact?: boolean;
  /** Disabled state */
  disabled?: boolean;
  /** Additional CSS class */
  className?: string;
}

export default function ProviderSelector({
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
  providerLabel = 'Provider',
  modelLabel = 'Model',
  showUseDefault = true,
  compact = false,
  disabled = false,
  className = '',
}: ProviderSelectorProps) {
  const { user } = useAuthStore();
  const [providersData, setProvidersData] = useState<AvailableProvidersResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load available providers
  const loadProviders = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchAvailableProviders(user?.id);
      setProvidersData(data);
    } catch (err) {
      console.error('Failed to load providers:', err);
      setError('Failed to load providers');
    } finally {
      setIsLoading(false);
    }
  }, [user?.id]);

  useEffect(() => {
    loadProviders();
  }, [loadProviders]);

  // Get available providers sorted by priority
  const getAvailableProviders = (): Array<{ id: string; info: ProviderInfo }> => {
    if (!providersData) return [];

    const providers = Object.entries(providersData.providers)
      .filter(([_, info]) => info.available || info.is_fallback)
      .map(([id, info]) => ({ id, info }))
      .sort((a, b) => {
        // Sort order: available first, then fallback
        if (a.info.available && !b.info.available) return -1;
        if (!a.info.available && b.info.available) return 1;
        // Put runpod (fallback) at the end
        if (a.info.is_fallback && !b.info.is_fallback) return 1;
        if (!a.info.is_fallback && b.info.is_fallback) return -1;
        return 0;
      });

    return providers;
  };

  // Get models for selected provider
  const getModelsForProvider = (): ModelInfo[] => {
    if (!providersData || !selectedProvider) return [];
    const provider = providersData.providers[selectedProvider];
    return provider?.models || [];
  };

  // Handle provider change
  const handleProviderChange = (providerId: string) => {
    if (providerId === '') {
      onProviderChange(null);
      onModelChange(null);
      return;
    }

    onProviderChange(providerId);

    // Auto-select first model when provider changes
    const provider = providersData?.providers[providerId];
    if (provider?.models?.length) {
      onModelChange(provider.models[0].id);
    } else {
      onModelChange(null);
    }
  };

  // Handle model change
  const handleModelChange = (modelId: string) => {
    onModelChange(modelId === '' ? null : modelId);
  };

  const availableProviders = getAvailableProviders();
  const availableModels = getModelsForProvider();

  if (isLoading) {
    return (
      <div className={`provider-selector ${className}`}>
        <div className="text-sm text-chat-muted-foreground">Loading providers...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`provider-selector ${className}`}>
        <div className="text-sm text-danger">{error}</div>
        <button
          onClick={loadProviders}
          className="text-xs text-chat-primary hover:underline mt-1"
        >
          Retry
        </button>
      </div>
    );
  }

  const containerClass = compact
    ? 'flex gap-3 items-center'
    : 'flex flex-col gap-4';

  const selectClass = `w-full px-3 py-2 border border-chat-border rounded text-sm outline-none
    focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10
    bg-chat-card text-chat-foreground
    ${disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`;

  return (
    <div className={`provider-selector ${className}`}>
      <div className={containerClass}>
        {/* Provider Dropdown */}
        <div className={compact ? 'flex-1' : ''}>
          <label className="block text-sm font-medium text-chat-foreground mb-1.5">
            {providerLabel}
          </label>
          <select
            value={selectedProvider || ''}
            onChange={(e) => handleProviderChange(e.target.value)}
            disabled={disabled}
            className={selectClass}
          >
            {showUseDefault && (
              <option value="">Use System Default</option>
            )}
            {availableProviders.map(({ id, info }) => (
              <option key={id} value={id}>
                {info.name}
                {info.is_fallback ? ' (Fallback)' : ''}
                {!info.available && info.requires_key ? ' (API Key Required)' : ''}
              </option>
            ))}
          </select>
          {/* Show message if no providers available */}
          {availableProviders.length === 0 && (
            <p className="text-xs text-chat-muted-foreground mt-1">
              No providers available. Add API keys in your profile settings.
            </p>
          )}
        </div>

        {/* Model Dropdown - only shown when provider is selected */}
        {selectedProvider && (
          <div className={compact ? 'flex-1' : ''}>
            <label className="block text-sm font-medium text-chat-foreground mb-1.5">
              {modelLabel}
            </label>
            <select
              value={selectedModel || ''}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={disabled || availableModels.length === 0}
              className={selectClass}
            >
              {availableModels.length === 0 ? (
                <option value="">No models available</option>
              ) : (
                availableModels.map((model) => (
                  <option key={model.id} value={model.id} title={model.description}>
                    {model.name}
                  </option>
                ))
              )}
            </select>
          </div>
        )}
      </div>

      {/* Provider status info */}
      {providersData && !compact && (
        <div className="mt-3 text-xs text-chat-muted-foreground">
          {providersData.ollama_local_available ? (
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-success inline-block"></span>
              Ollama running locally
            </span>
          ) : (
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-chat-muted-foreground inline-block"></span>
              Ollama not detected
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Simplified provider selector for use in forms
 * Returns just provider/model as a combined value
 */
interface SimpleProviderSelectorProps {
  value: { provider: string | null; model: string | null };
  onChange: (value: { provider: string | null; model: string | null }) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
}

export function SimpleProviderSelector({
  value,
  onChange,
  label = 'AI Provider',
  disabled = false,
  className = '',
}: SimpleProviderSelectorProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-chat-foreground mb-2">
          {label}
        </label>
      )}
      <ProviderSelector
        selectedProvider={value.provider}
        selectedModel={value.model}
        onProviderChange={(provider) => onChange({ ...value, provider })}
        onModelChange={(model) => onChange({ ...value, model })}
        providerLabel=""
        modelLabel=""
        showUseDefault={true}
        compact={true}
        disabled={disabled}
      />
    </div>
  );
}
