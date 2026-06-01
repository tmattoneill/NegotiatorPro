/**
 * Authentication state management using Zustand
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  role: string;
  has_openai_key?: boolean;
  has_anthropic_key?: boolean;
  has_deepseek_key?: boolean;
  is_super_admin?: boolean;
  preferred_provider?: string | null;
  preferred_model?: string | null;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;

  // Actions
  login: (user: User) => void;
  logout: () => void;
  updateUser: (user: User) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,

      login: (user: User) => {
        set({ user, isAuthenticated: true });
      },

      logout: () => {
        // Clear JWT token from localStorage
        localStorage.removeItem('token');
        set({ user: null, isAuthenticated: false });
      },

      updateUser: (user: User) => {
        set({ user });
      },
    }),
    {
      name: 'auth-storage', // localStorage key
    }
  )
);
