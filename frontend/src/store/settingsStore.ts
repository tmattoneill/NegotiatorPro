/**
 * Settings Store
 *
 * Manages user settings for LLM selection, premium model toggle, and text preprocessing.
 */

import { create } from 'zustand';
import type { ModelsResponse } from '../types';
import { fetchAvailableProviders } from '../services/api';
import { useAuthStore } from './authStore';

interface SettingsState {
  // Model selection
  selectedProvider: string | null;
  selectedModel: string | null;
  availableModels: ModelsResponse;

  // Feature toggles
  usePremiumModel: boolean;
  usePreprocessing: boolean;

  // Loading state
  isLoadingModels: boolean;
  modelsError: string | null;

  // Actions
  setProvider: (provider: string) => void;
  setModel: (model: string) => void;
  setProviderModel: (provider: string | null, model: string | null) => void;
  setPremiumModel: (use: boolean) => void;
  setPreprocessing: (use: boolean) => void;
  loadAvailableModels: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  // Initial state
  selectedProvider: null,
  selectedModel: null,
  availableModels: {},
  usePremiumModel: false,
  usePreprocessing: true,
  isLoadingModels: false,
  modelsError: null,

  // Actions
  setProvider: (provider: string) => {
    set({ selectedProvider: provider });

    // Auto-select first model when provider changes
    const models = get().availableModels[provider]?.models || [];
    if (models.length > 0) {
      set({ selectedModel: models[0].id });
    } else {
      set({ selectedModel: null });
    }
  },

  setModel: (model: string) => {
    set({ selectedModel: model });
  },

  // Set provider and model together without the auto-first-model behaviour —
  // used when loading a negotiation's saved choice so a stored model is kept.
  setProviderModel: (provider: string | null, model: string | null) => {
    set({ selectedProvider: provider, selectedModel: model });
  },

  setPremiumModel: (use: boolean) => {
    set({ usePremiumModel: use });
  },

  setPreprocessing: (use: boolean) => {
    set({ usePreprocessing: use });
  },

  loadAvailableModels: async () => {
    set({ isLoadingModels: true, modelsError: null });

    try {
      // Users can choose any LLM they have a key for — this is independent of
      // the admin's backend-processor enablement. Source the picker from the
      // key-aware endpoint, not the admin-gated /api/models.
      const userId = useAuthStore.getState().user?.id;
      const { providers } = await fetchAvailableProviders(userId);

      // Map the provider response into the ModelsResponse shape the pickers
      // expect. Only include providers that are actually usable (available) or
      // offered as a fallback. Skip unavailable providers even if they carry
      // an error message — showing a broken option just confuses users.
      const models: ModelsResponse = {};
      for (const [id, info] of Object.entries(providers)) {
        if (info.available || info.is_fallback) {
          models[id] = {
            name: info.name,
            enabled: true,
            models: info.models || [],
          };
        }
      }
      set({ availableModels: models, isLoadingModels: false });

      // Initialise the selection if none is set yet this session. Honour the
      // user's saved preference (preferred_provider/preferred_model) when it's
      // still available; otherwise fall back to the first available provider.
      if (!get().selectedProvider) {
        const user = useAuthStore.getState().user;
        const prefProvider = user?.preferred_provider;
        const prefModel = user?.preferred_model;

        if (prefProvider && models[prefProvider]) {
          const providerModels = models[prefProvider].models;
          const modelExists = prefModel && providerModels.some(m => m.id === prefModel);
          set({
            selectedProvider: prefProvider,
            selectedModel: modelExists ? prefModel : (providerModels[0]?.id ?? null),
          });
        } else {
          const firstBackend = Object.keys(models)[0];
          if (firstBackend) {
            set({
              selectedProvider: firstBackend,
              selectedModel: models[firstBackend].models[0]?.id ?? null,
            });
          }
        }
      }
    } catch (error) {
      console.error('Failed to load models:', error);
      set({
        modelsError: error instanceof Error ? error.message : 'Failed to load models',
        isLoadingModels: false
      });
    }
  },
}));
