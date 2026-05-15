/**
 * LLM Configuration Tab Component
 * Admin UI for managing LLM backends and model selection
 */
import { useState, useEffect } from 'react';
import * as api from '../services/api';
import type { LLMBackendStatus, LLMConfigResponse, ActiveModelConfig } from '../services/api';
import Button from './ui/Button';
import Badge from './ui/Badge';

interface ModelSelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelType: 'default' | 'premium';
  backends: LLMBackendStatus[];
  currentConfig: ActiveModelConfig;
  onSelect: (backend: string, model: string) => void;
}

const ModelSelectModal = ({ isOpen, onClose, modelType, backends, currentConfig, onSelect }: ModelSelectModalProps) => {
  const [selectedBackend, setSelectedBackend] = useState(currentConfig.backend);
  const [selectedModel, setSelectedModel] = useState(currentConfig.model);

  useEffect(() => {
    setSelectedBackend(currentConfig.backend);
    setSelectedModel(currentConfig.model);
  }, [currentConfig.backend, currentConfig.model, isOpen]);

  if (!isOpen) return null;

  const availableBackends = backends.filter(b => b.is_available);
  const currentBackendData = backends.find(b => b.id === selectedBackend);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[80vh] overflow-hidden">
        <div className="p-6 border-b border-chat-border">
          <h3 className="text-lg font-semibold">
            Select {modelType === 'default' ? 'Default' : 'Premium'} Model
          </h3>
          <p className="text-sm text-chat-muted-foreground mt-1">
            Choose the {modelType} model for all users
          </p>
        </div>

        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {/* Backend Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Backend Provider</label>
            <select
              value={selectedBackend}
              onChange={(e) => {
                setSelectedBackend(e.target.value);
                const newBackend = backends.find(b => b.id === e.target.value);
                if (newBackend && newBackend.models.length > 0) {
                  setSelectedModel(newBackend.models[0].id);
                }
              }}
              className="w-full px-3 py-2 border border-chat-border rounded-md focus:ring-2 focus:ring-chat-primary/20"
            >
              {availableBackends.map(backend => (
                <option key={backend.id} value={backend.id}>
                  {backend.name} ({backend.models.length} models)
                </option>
              ))}
            </select>
          </div>

          {/* Model Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Model</label>
            {currentBackendData && currentBackendData.models.length > 0 ? (
              <div className="space-y-2">
                {currentBackendData.models.map(model => (
                  <label
                    key={model.id}
                    className={`flex items-start p-3 border rounded-md cursor-pointer transition-colors ${
                      selectedModel === model.id
                        ? 'border-chat-primary bg-chat-primary/5'
                        : 'border-chat-border hover:bg-chat-muted'
                    }`}
                  >
                    <input
                      type="radio"
                      name="model"
                      value={model.id}
                      checked={selectedModel === model.id}
                      onChange={() => setSelectedModel(model.id)}
                      className="mt-1 mr-3"
                    />
                    <div>
                      <div className="font-medium">{model.name}</div>
                      <div className="text-sm text-chat-muted-foreground">{model.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-chat-muted-foreground italic">
                No models available for this backend
              </p>
            )}
          </div>
        </div>

        <div className="p-6 border-t border-chat-border flex justify-end gap-3">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button
            variant="primary"
            onClick={() => {
              onSelect(selectedBackend, selectedModel);
              onClose();
            }}
            disabled={!selectedModel}
          >
            Save Selection
          </Button>
        </div>
      </div>
    </div>
  );
};

const LLMConfigTab = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [config, setConfig] = useState<LLMConfigResponse | null>(null);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState<'default' | 'premium'>('default');

  const loadConfig = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.adminGetLLMConfig();
      setConfig(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load LLM configuration');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const handleSetModel = async (backend: string, model: string) => {
    try {
      setLoading(true);
      setError(null);
      await api.adminSetModel({ model_type: modalType, backend, model });
      setSuccess(`${modalType === 'default' ? 'Default' : 'Premium'} model updated successfully`);
      await loadConfig();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update model');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleBackend = async (backendId: string, currentEnabled: boolean) => {
    try {
      setLoading(true);
      setError(null);
      await api.adminEnableBackend(backendId, !currentEnabled);
      setSuccess(`Backend ${!currentEnabled ? 'enabled' : 'disabled'} successfully`);
      await loadConfig();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update backend');
    } finally {
      setLoading(false);
    }
  };

  const openModelSelector = (type: 'default' | 'premium') => {
    setModalType(type);
    setModalOpen(true);
  };

  const getBackendName = (backendId: string) => {
    return config?.backends.find(b => b.id === backendId)?.name || backendId;
  };

  const getModelName = (backendId: string, modelId: string) => {
    const backend = config?.backends.find(b => b.id === backendId);
    return backend?.models.find(m => m.id === modelId)?.name || modelId;
  };

  if (loading && !config) {
    return <div className="p-8 text-center text-chat-muted-foreground">Loading LLM configuration...</div>;
  }

  return (
    <div>
      <h2 className="text-xl font-semibold text-chat-foreground">LLM Backend Configuration</h2>
      <p className="text-sm text-chat-muted-foreground mb-6">
        Configure system-wide default and premium models
      </p>

      {/* Status Messages */}
      {error && (
        <div className="px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger mb-4">
          {error}
        </div>
      )}
      {success && (
        <div className="px-4 py-3 rounded-md border border-green-500/30 bg-green-500/10 text-green-700 mb-4">
          {success}
        </div>
      )}

      {/* Current Active Models */}
      <div className="mb-8">
        <h3 className="text-lg font-medium mb-4">Active Models</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Default Model */}
          <div className="p-4 border border-chat-border rounded-lg bg-chat-muted">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-chat-muted-foreground uppercase">Default Model</span>
              <Badge variant="outline">Standard</Badge>
            </div>
            <div className="mb-3">
              <div className="text-lg font-semibold">
                {config?.active_models.default ? getModelName(config.active_models.default.backend, config.active_models.default.model) : 'Not set'}
              </div>
              <div className="text-sm text-chat-muted-foreground">
                {config?.active_models.default ? getBackendName(config.active_models.default.backend) : ''}
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={() => openModelSelector('default')} disabled={loading}>
              Change
            </Button>
          </div>

          {/* Premium Model */}
          <div className="p-4 border border-chat-primary/30 rounded-lg bg-chat-primary/5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-chat-muted-foreground uppercase">Premium Model</span>
              <Badge variant="success">Premium</Badge>
            </div>
            <div className="mb-3">
              <div className="text-lg font-semibold">
                {config?.active_models.premium ? getModelName(config.active_models.premium.backend, config.active_models.premium.model) : 'Not set'}
              </div>
              <div className="text-sm text-chat-muted-foreground">
                {config?.active_models.premium ? getBackendName(config.active_models.premium.backend) : ''}
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={() => openModelSelector('premium')} disabled={loading}>
              Change
            </Button>
          </div>
        </div>
      </div>

      {/* Available Backends */}
      <div>
        <h3 className="text-lg font-medium mb-4">Available Backends</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {config?.backends.map(backend => (
            <div
              key={backend.id}
              className={`p-4 border rounded-lg ${
                backend.is_available
                  ? 'border-green-500/30 bg-green-500/5'
                  : 'border-chat-border bg-chat-muted'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="font-semibold">{backend.name}</h4>
                  <div className="text-sm text-chat-muted-foreground">
                    {backend.models.length} models
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  {backend.is_available ? (
                    <Badge variant="success">Ready</Badge>
                  ) : backend.enabled && !backend.has_api_key ? (
                    <Badge variant="warning">No API Key</Badge>
                  ) : (
                    <Badge variant="muted">Disabled</Badge>
                  )}
                </div>
              </div>

              {/* Status indicators */}
              <div className="text-xs space-y-1 mb-3">
                <div className="flex items-center gap-2">
                  <span className={backend.enabled ? 'text-green-600' : 'text-chat-muted-foreground'}>
                    {backend.enabled ? '✓' : '○'} Enabled
                  </span>
                </div>
                {backend.requires_api_key && (
                  <div className="flex items-center gap-2">
                    <span className={backend.has_api_key ? 'text-green-600' : 'text-amber-600'}>
                      {backend.has_api_key ? '✓' : '!'} API Key {backend.has_api_key ? 'Present' : 'Missing'}
                    </span>
                  </div>
                )}
              </div>

              {/* Toggle button */}
              <Button
                variant={backend.enabled ? 'outline' : 'primary'}
                size="sm"
                onClick={() => handleToggleBackend(backend.id, backend.enabled)}
                disabled={loading}
                className="w-full"
              >
                {backend.enabled ? 'Disable' : 'Enable'}
              </Button>
            </div>
          ))}
        </div>
      </div>

      {/* Model Selection Modal */}
      {config && (
        <ModelSelectModal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          modelType={modalType}
          backends={config.backends}
          currentConfig={config.active_models[modalType]}
          onSelect={handleSetModel}
        />
      )}
    </div>
  );
};

export default LLMConfigTab;
