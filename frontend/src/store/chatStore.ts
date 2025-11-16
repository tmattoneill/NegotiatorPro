/**
 * Chat state management using Zustand - with session support
 */
import { create } from 'zustand';
import type { Message } from '../types';

interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  messageCount: number;
}

interface ChatState {
  sessions: Session[];
  currentSessionId: string | null;
  isLoading: boolean;

  // Actions
  createNewSession: () => void;
  switchSession: (sessionId: string) => void;
  getCurrentSession: () => Session | undefined;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  isLoading: false,

  createNewSession: () => {
    const newSession: Session = {
      id: Date.now().toString(),
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      messageCount: 0,
    };
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSessionId: newSession.id,
    }));
  },

  switchSession: (sessionId: string) => {
    set({ currentSessionId: sessionId });
  },

  getCurrentSession: () => {
    const { sessions, currentSessionId } = get();
    return sessions.find((s) => s.id === currentSessionId);
  },

  addMessage: (message: Message) => {
    set((state) => {
      const { currentSessionId, sessions } = state;

      // Create new session if none exists
      if (!currentSessionId) {
        const newSession: Session = {
          id: Date.now().toString(),
          title: message.content.slice(0, 50) + (message.content.length > 50 ? '...' : ''),
          messages: [message],
          createdAt: new Date(),
          messageCount: 1,
        };
        return {
          sessions: [newSession, ...sessions],
          currentSessionId: newSession.id,
        };
      }

      // Add message to current session
      const updatedSessions = sessions.map((session) => {
        if (session.id === currentSessionId) {
          const updatedMessages = [...session.messages, message];
          // Update title from first user message
          const title =
            session.messages.length === 0 && message.role === 'user'
              ? message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
              : session.title;

          return {
            ...session,
            messages: updatedMessages,
            messageCount: updatedMessages.length,
            title,
          };
        }
        return session;
      });

      return { sessions: updatedSessions };
    });
  },

  setLoading: (loading) => set({ isLoading: loading }),
}));
