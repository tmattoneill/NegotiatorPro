/**
 * Settings Store
 *
 * Manages user settings for LLM selection, premium model toggle, and text preprocessing.
 */

import { create } from 'zustand';
import type { ModelsResponse } from '../types';
import { fetchAvailableModels } from '../services/api';

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

  setPremiumModel: (use: boolean) => {
    set({ usePremiumModel: use });
  },

  setPreprocessing: (use: boolean) => {
    set({ usePreprocessing: use });
  },

  loadAvailableModels: async () => {
    set({ isLoadingModels: true, modelsError: null });

    try {
      const models = await fetchAvailableModels();
      set({ availableModels: models, isLoadingModels: false });

      // Auto-select first available provider and model if none selected
      if (!get().selectedProvider) {
        const firstBackend = Object.keys(models)[0];
        if (firstBackend) {
          const firstModel = models[firstBackend].models[0];
          set({
            selectedProvider: firstBackend,
            selectedModel: firstModel?.id || null
          });
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
