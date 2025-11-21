/**
 * Conversation Types
 *
 * TypeScript interfaces for conversation data structures
 */

export interface Conversation {
  id: string;
  negotiation_id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ConversationCreate {
  negotiation_id: string;
  title?: string;
}

export interface ConversationUpdate {
  title?: string;
}
