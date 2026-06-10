/**
 * Context Store
 *
 * Holds the negotiation-level gutter data (leverage, parties, vitals) fetched
 * lazily from the context endpoint. Fetched when the gutter is expanded and
 * refreshed when new turns land; the backend caches the costly leverage/vitals
 * analysis, so repeat fetches are cheap.
 */
import { create } from 'zustand';
import type { NegotiationLevelContext } from '../types/negotiationContext';
import { getNegotiationContext } from '../services/api';

interface ContextState {
  data: NegotiationLevelContext | null;
  loading: boolean;
  // Guards against refetching the same (negotiation, message-count) pair.
  fetchedKey: string | null;

  fetchContext: (
    negotiationId: string,
    userId: string,
    messageCount: number,
  ) => Promise<void>;
  clear: () => void;
}

export const useContextStore = create<ContextState>((set, get) => ({
  data: null,
  loading: false,
  fetchedKey: null,

  fetchContext: async (negotiationId, userId, messageCount) => {
    const key = `${negotiationId}:${messageCount}`;
    if (get().loading || get().fetchedKey === key) return;

    set({ loading: true });
    try {
      const data = await getNegotiationContext(negotiationId, userId);
      set({ data, fetchedKey: key, loading: false });
    } catch (error) {
      console.error('Failed to load negotiation context:', error);
      set({ loading: false });
    }
  },

  clear: () => set({ data: null, fetchedKey: null, loading: false }),
}));
