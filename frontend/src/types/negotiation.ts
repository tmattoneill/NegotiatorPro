/**
 * Negotiation Types
 *
 * TypeScript interfaces for negotiation data structures
 */

export interface KeyPlayer {
  name: string;
  role: string;
  notes?: string;
}

export interface Negotiation {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  visibility: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
  status: 'active' | 'archived' | 'completed';
  created_at: string;
  updated_at: string;
  conversation_count?: number;
  document_count?: number;
  url_count?: number;
}

export interface NegotiationCreate {
  name: string;
  description?: string;
  visibility?: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
}

export interface NegotiationUpdate {
  name?: string;
  description?: string;
  visibility?: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
  status?: 'active' | 'archived' | 'completed';
}
