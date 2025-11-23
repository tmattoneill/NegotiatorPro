/**
 * Negotiation Store
 *
 * Manages negotiation state using Zustand with persistence
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Negotiation, NegotiationCreate, NegotiationUpdate } from '../types/negotiation';
import api from '../services/api';

interface NegotiationState {
  // State
  negotiations: Negotiation[];
  currentNegotiationId: string | null;
  isLoading: boolean;
  error: string | null;

  // Computed
  currentNegotiation: Negotiation | null;

  // Actions
  loadNegotiations: (userId: string) => Promise<void>;
  createNegotiation: (userId: string, data: NegotiationCreate) => Promise<Negotiation | null>;
  updateNegotiation: (userId: string, negotiationId: string, data: NegotiationUpdate) => Promise<void>;
  deleteNegotiation: (userId: string, negotiationId: string) => Promise<void>;
  setCurrentNegotiation: (negotiationId: string | null) => void;
  clearError: () => void;
}

export const useNegotiationStore = create<NegotiationState>()(
  persist(
    (set, get) => ({
      // Initial state
      negotiations: [],
      currentNegotiationId: null,
      isLoading: false,
      error: null,

      // Computed property
      get currentNegotiation() {
        const { negotiations, currentNegotiationId } = get();
        return negotiations.find(n => n.id === currentNegotiationId) || null;
      },

      // Load all negotiations for a user
      loadNegotiations: async (userId: string) => {
        set({ isLoading: true, error: null });

        try {
          const response = await api.get(`/negotiations/?user_id=${userId}`);
          const negotiations = response.data;

          // Check if the currently selected negotiation is in the loaded list
          const currentId = get().currentNegotiationId;
          const isCurrentValid = currentId && negotiations.some((n: Negotiation) => n.id === currentId);

          set({
            negotiations,
            isLoading: false,
            // Only keep current negotiation if it's valid for this user, otherwise select first
            currentNegotiationId: isCurrentValid ? currentId : (negotiations[0]?.id ?? null),
          });
        } catch (error: any) {
          console.error('Failed to load negotiations:', error);
          set({
            error: error.response?.data?.detail || 'Failed to load negotiations',
            isLoading: false,
          });
        }
      },

      // Create a new negotiation
      createNegotiation: async (userId: string, data: NegotiationCreate) => {
        set({ isLoading: true, error: null });

        try {
          const response = await api.post(`/negotiations/?user_id=${userId}`, data);
          const newNegotiation = response.data;

          set((state) => ({
            negotiations: [newNegotiation, ...state.negotiations],
            currentNegotiationId: newNegotiation.id,
            isLoading: false,
          }));

          return newNegotiation;
        } catch (error: any) {
          console.error('Failed to create negotiation:', error);
          set({
            error: error.response?.data?.detail || 'Failed to create negotiation',
            isLoading: false,
          });
          return null;
        }
      },

      // Update an existing negotiation
      updateNegotiation: async (userId: string, negotiationId: string, data: NegotiationUpdate) => {
        set({ isLoading: true, error: null });

        try {
          const response = await api.put(`/negotiations/${negotiationId}?user_id=${userId}`, data);
          const updatedNegotiation = response.data;

          set((state) => ({
            negotiations: state.negotiations.map((n) =>
              n.id === negotiationId ? updatedNegotiation : n
            ),
            isLoading: false,
          }));
        } catch (error: any) {
          console.error('Failed to update negotiation:', error);
          set({
            error: error.response?.data?.detail || 'Failed to update negotiation',
            isLoading: false,
          });
        }
      },

      // Delete a negotiation
      deleteNegotiation: async (userId: string, negotiationId: string) => {
        set({ isLoading: true, error: null });

        try {
          await api.delete(`/negotiations/${negotiationId}?user_id=${userId}`);

          set((state) => {
            const updatedNegotiations = state.negotiations.filter((n) => n.id !== negotiationId);

            return {
              negotiations: updatedNegotiations,
              // If deleted negotiation was current, select first available
              currentNegotiationId:
                state.currentNegotiationId === negotiationId
                  ? (updatedNegotiations[0]?.id ?? null)
                  : state.currentNegotiationId,
              isLoading: false,
            };
          });
        } catch (error: any) {
          console.error('Failed to delete negotiation:', error);
          set({
            error: error.response?.data?.detail || 'Failed to delete negotiation',
            isLoading: false,
          });
        }
      },

      // Set current negotiation
      setCurrentNegotiation: (negotiationId: string | null) => {
        set({ currentNegotiationId: negotiationId });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'negotiation-storage', // localStorage key
      // Only persist negotiations list and current selection
      partialize: (state) => ({
        negotiations: state.negotiations,
        currentNegotiationId: state.currentNegotiationId,
      }),
    }
  )
);
