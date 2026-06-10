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
  isCreatingSession: boolean;

  // Actions
  loadConversations: (negotiationId: string, userId: string) => Promise<void>;
  loadConversationMessages: (conversationId: string, userId: string) => Promise<void>;
  createNewSession: (negotiationId: string, userId: string) => Promise<void>;
  renameSession: (sessionId: string, userId: string, newTitle: string) => Promise<void>;
  deleteSession: (sessionId: string, userId: string) => Promise<void>;
  switchSession: (sessionId: string, userId?: string) => Promise<void>;
  getCurrentSession: () => Session | undefined;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  createLocalSession: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  isLoading: false,
  isCreatingSession: false,

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
      const formattedMessages: Message[] = messages.map((msg: any) => ({
        id: msg.id?.toString() || crypto.randomUUID(),
        role: msg.role,
        content: msg.content,
        timestamp: msg.created_at ? new Date(msg.created_at) : new Date(),
        model_used: msg.model_used || msg.model,
        processing_time: msg.processing_time,
        detected_intent: msg.detected_intent || undefined,
        please: msg.please_score || null,
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
    // Prevent duplicate creation if already in progress
    const { isCreatingSession } = get();
    if (isCreatingSession) {
      return;
    }

    try {
      set({ isCreatingSession: true });

      const newConversation = await api.createConversation({
        negotiation_id: negotiationId,
        title: 'New Conversation',
        user_id: userId,
      });

      const newSession: Session = {
        id: newConversation.id,
        title: newConversation.title || 'New Conversation',
        messages: [],
        createdAt: new Date(newConversation.created_at),
        messageCount: 0,
        negotiationId: negotiationId,
      };

      set((state) => ({
        sessions: [newSession, ...state.sessions],
        currentSessionId: newSession.id,
        isCreatingSession: false,
      }));
    } catch (error) {
      console.error('Failed to create conversation:', error);
      set({ isCreatingSession: false });
      throw error; // Re-throw so caller knows it failed
    }
  },

  // Rename a conversation
  renameSession: async (sessionId: string, userId: string, newTitle: string) => {
    try {
      // Trim and limit to 64 characters
      const trimmedTitle = newTitle.trim().slice(0, 64);

      if (!trimmedTitle) {
        throw new Error('Title cannot be empty');
      }

      await api.updateConversation(sessionId, userId, { title: trimmedTitle });

      // Update local state
      set((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === sessionId
            ? { ...session, title: trimmedTitle }
            : session
        ),
      }));
    } catch (error) {
      console.error('Failed to rename conversation:', error);
      throw error;
    }
  },

  // Delete a conversation
  deleteSession: async (sessionId: string, userId: string) => {
    try {
      await api.deleteConversation(sessionId, userId);

      const { sessions, currentSessionId } = get();
      const updatedSessions = sessions.filter(s => s.id !== sessionId);

      // If we deleted the current session, switch to the first available one
      const newCurrentId = currentSessionId === sessionId
        ? (updatedSessions[0]?.id || null)
        : currentSessionId;

      set({
        sessions: updatedSessions,
        currentSessionId: newCurrentId,
      });
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      throw error;
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

  createLocalSession: () => {
    const localSession: Session = {
      id: `local-${Date.now()}`,
      title: 'Testing Session',
      messages: [],
      createdAt: new Date(),
      messageCount: 0,
    };
    set((state) => ({
      sessions: [localSession, ...state.sessions],
      currentSessionId: localSession.id,
    }));
  },
}));
