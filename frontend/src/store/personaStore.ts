/**
 * Persona Store
 *
 * Zustand store for managing user and partner personas
 */
import { create } from 'zustand';
import type { UserPersona, UserPersonaCreate, PartnerPersona, PartnerPersonaCreate } from '../types/personas';
import * as api from '../services/api';

interface PersonaState {
  // User Personas
  userPersonas: UserPersona[];
  selectedUserPersona: UserPersona | null;
  userPersonasLoading: boolean;

  // Partner Personas
  partnerPersonas: PartnerPersona[];
  selectedPartnerPersona: PartnerPersona | null;
  partnerPersonasLoading: boolean;

  error: string | null;

  // User Persona Actions
  fetchUserPersonas: (userId: string) => Promise<void>;
  createUserPersona: (userId: string, data: UserPersonaCreate) => Promise<UserPersona>;
  updateUserPersona: (userId: string, id: string, data: Partial<UserPersonaCreate>) => Promise<void>;
  deleteUserPersona: (userId: string, id: string) => Promise<void>;
  selectUserPersona: (persona: UserPersona | null) => void;

  // Partner Persona Actions
  fetchPartnerPersonas: (userId: string, includeShared?: boolean) => Promise<void>;
  createPartnerPersona: (userId: string, data: PartnerPersonaCreate) => Promise<PartnerPersona>;
  updatePartnerPersona: (userId: string, id: string, data: Partial<PartnerPersonaCreate>) => Promise<void>;
  deletePartnerPersona: (userId: string, id: string) => Promise<void>;
  selectPartnerPersona: (persona: PartnerPersona | null) => void;

  clearError: () => void;
}

export const usePersonaStore = create<PersonaState>((set, get) => ({
  userPersonas: [],
  selectedUserPersona: null,
  userPersonasLoading: false,

  partnerPersonas: [],
  selectedPartnerPersona: null,
  partnerPersonasLoading: false,

  error: null,

  // User Persona Actions
  fetchUserPersonas: async (userId) => {
    set({ userPersonasLoading: true, error: null });
    try {
      const personas = await api.getUserPersonas(userId);
      set({ userPersonas: personas, userPersonasLoading: false });
    } catch (err) {
      set({ error: 'Failed to fetch user personas', userPersonasLoading: false });
    }
  },

  createUserPersona: async (userId, data) => {
    try {
      const persona = await api.createUserPersona(userId, data);
      set((state) => ({ userPersonas: [...state.userPersonas, persona] }));
      return persona;
    } catch (err) {
      set({ error: 'Failed to create user persona' });
      throw err;
    }
  },

  updateUserPersona: async (userId, id, data) => {
    try {
      const updated = await api.updateUserPersona(userId, id, data);
      set((state) => ({
        userPersonas: state.userPersonas.map((p) => (p.id === id ? updated : p)),
        selectedUserPersona: state.selectedUserPersona?.id === id ? updated : state.selectedUserPersona,
      }));
    } catch (err) {
      set({ error: 'Failed to update user persona' });
      throw err;
    }
  },

  deleteUserPersona: async (userId, id) => {
    try {
      await api.deleteUserPersona(userId, id);
      set((state) => ({
        userPersonas: state.userPersonas.filter((p) => p.id !== id),
        selectedUserPersona: state.selectedUserPersona?.id === id ? null : state.selectedUserPersona,
      }));
    } catch (err) {
      set({ error: 'Failed to delete user persona' });
      throw err;
    }
  },

  selectUserPersona: (persona) => {
    set({ selectedUserPersona: persona });
  },

  // Partner Persona Actions
  fetchPartnerPersonas: async (userId, includeShared = true) => {
    set({ partnerPersonasLoading: true, error: null });
    try {
      const personas = await api.getPartnerPersonas(userId, includeShared);
      set({ partnerPersonas: personas, partnerPersonasLoading: false });
    } catch (err) {
      set({ error: 'Failed to fetch partner personas', partnerPersonasLoading: false });
    }
  },

  createPartnerPersona: async (userId, data) => {
    try {
      const persona = await api.createPartnerPersona(userId, data);
      set((state) => ({ partnerPersonas: [...state.partnerPersonas, persona] }));
      return persona;
    } catch (err) {
      set({ error: 'Failed to create partner persona' });
      throw err;
    }
  },

  updatePartnerPersona: async (userId, id, data) => {
    try {
      const updated = await api.updatePartnerPersona(userId, id, data);
      set((state) => ({
        partnerPersonas: state.partnerPersonas.map((p) => (p.id === id ? updated : p)),
        selectedPartnerPersona: state.selectedPartnerPersona?.id === id ? updated : state.selectedPartnerPersona,
      }));
    } catch (err) {
      set({ error: 'Failed to update partner persona' });
      throw err;
    }
  },

  deletePartnerPersona: async (userId, id) => {
    try {
      await api.deletePartnerPersona(userId, id);
      set((state) => ({
        partnerPersonas: state.partnerPersonas.filter((p) => p.id !== id),
        selectedPartnerPersona: state.selectedPartnerPersona?.id === id ? null : state.selectedPartnerPersona,
      }));
    } catch (err) {
      set({ error: 'Failed to delete partner persona' });
      throw err;
    }
  },

  selectPartnerPersona: (persona) => {
    set({ selectedPartnerPersona: persona });
  },

  clearError: () => set({ error: null }),
}));
