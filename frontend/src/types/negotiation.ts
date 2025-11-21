/**
 * Negotiation Types
 *
 * TypeScript interfaces for negotiation data structures
 */

import type { UserPersona, PartnerPersona } from './personas';

export type NegotiationStatus = 'active' | 'paused' | 'closed' | 'won' | 'lost';

// Legacy - kept for backwards compatibility during migration
export interface KeyPlayer {
  name: string;
  role: string;
  notes?: string;
}

export interface Negotiation {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  status: NegotiationStatus;
  user_persona_id?: string;
  settings?: Record<string, unknown>;
  created_at: string;
  updated_at: string;

  // From detail endpoint
  user_persona?: UserPersona;
  partners?: PartnerPersona[];
  conversation_count?: number;

  // Legacy fields - will be removed after migration
  name?: string;  // Alias for title
  visibility?: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
  document_count?: number;
  url_count?: number;
}

export interface NegotiationCreate {
  title: string;
  description?: string;
  user_persona_id?: string;
  partner_persona_ids: string[];  // Required: at least one
  primary_partner_id?: string;
  settings?: Record<string, unknown>;

  // Legacy - will be removed
  name?: string;
  visibility?: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
}

export interface NegotiationUpdate {
  title?: string;
  description?: string;
  status?: NegotiationStatus;
  user_persona_id?: string;
  settings?: Record<string, unknown>;

  // Legacy - will be removed
  name?: string;
  visibility?: 'private' | 'company';
  preferred_backend?: string;
  preferred_model?: string;
  background?: string;
  key_players?: KeyPlayer[];
}
