/**
 * Admin state management using Zustand
 */
import { create } from 'zustand';

type AdminView = 'none' | 'system-prompt';

interface AdminState {
  currentView: AdminView;
  setView: (view: AdminView) => void;
}

export const useAdminStore = create<AdminState>()((set) => ({
  currentView: 'none',
  setView: (view: AdminView) => set({ currentView: view }),
}));
