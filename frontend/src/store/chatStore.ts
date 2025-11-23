/**
 * Chat state management using Zustand - integrated with database conversations
 */
import { create } from 'zustand';
import type { Message } from '../types';
import * as api from '../services/api';

interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  messageCount: number;
  negotiationId?: string;
}

interface ChatState {
  sessions: Session[];
  currentSessionId: string | null;
  isLoading: boolean;

  // Actions
  loadConversations: (negotiationId: string, userId: string) => Promise<void>;
  loadConversationMessages: (conversationId: string, userId: string) => Promise<void>;
  createNewSession: (negotiationId: string, userId: string) => Promise<void>;
  switchSession: (sessionId: string, userId?: string) => Promise<void>;
  getCurrentSession: () => Session | undefined;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  isLoading: false,

  // Load all conversations for a negotiation from database
  loadConversations: async (negotiationId: string, userId: string) => {
    try {
      const conversations = await api.getNegotiationConversations(negotiationId, userId);

      const sessions: Session[] = conversations.map(conv => ({
        id: conv.id,
        title: conv.title || 'Untitled Conversation',
        messages: [],
        createdAt: new Date(conv.created_at),
        messageCount: conv.message_count || 0,
        negotiationId: negotiationId,
      }));

      set({ sessions, currentSessionId: sessions[0]?.id || null });

      // Load messages for first conversation if it has messages
      if (sessions[0] && sessions[0].messageCount > 0) {
        await get().loadConversationMessages(sessions[0].id, userId);
      }
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  },

  // Load messages for a specific conversation from database
  loadConversationMessages: async (conversationId: string, userId: string) => {
    try {
      const messages = await api.getConversationMessages(conversationId, userId);

      // Convert database messages to Message type
      const formattedMessages: Message[] = messages.map(msg => ({
        role: msg.role,
        content: msg.content,
      }));

      set((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === conversationId
            ? { ...session, messages: formattedMessages, messageCount: formattedMessages.length }
            : session
        ),
      }));
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  },

  // Create new conversation in database
  createNewSession: async (negotiationId: string, userId: string) => {
    console.log('createNewSession called with:', { negotiationId, userId });
    try {
      const newConversation = await api.createConversation({
        negotiation_id: negotiationId,
        title: 'New Conversation',
        user_id: userId,
      });

      console.log('Conversation created successfully:', newConversation);

      const newSession: Session = {
        id: newConversation.id,
        title: newConversation.title || 'New Conversation',
        messages: [],
        createdAt: new Date(newConversation.created_at),
        messageCount: 0,
        negotiationId: negotiationId,
      };

      console.log('Adding session to state:', newSession);

      set((state) => ({
        sessions: [newSession, ...state.sessions],
        currentSessionId: newSession.id,
      }));

      console.log('Session added, currentSessionId:', newSession.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
      throw error; // Re-throw so caller knows it failed
    }
  },

  // Switch to a different conversation and load its messages
  switchSession: async (sessionId: string, userId?: string) => {
    const session = get().sessions.find(s => s.id === sessionId);

    // Load messages if not already loaded (for database sessions)
    if (session && session.messages.length === 0 && session.messageCount > 0 && userId) {
      await get().loadConversationMessages(sessionId, userId);
    }

    set({ currentSessionId: sessionId });
  },

  getCurrentSession: () => {
    const { sessions, currentSessionId } = get();
    return sessions.find((s) => s.id === currentSessionId);
  },

  // Add message to current session (used for optimistic updates)
  addMessage: (message: Message) => {
    set((state) => {
      const { currentSessionId, sessions } = state;

      if (!currentSessionId) {
        return state;
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
