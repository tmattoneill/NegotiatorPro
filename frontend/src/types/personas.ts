/**
 * Persona Types
 *
 * TypeScript interfaces for user and partner persona data structures
 */

export interface UserPersona {
  id: string;
  user_id: string;
  name: string;
  role_title?: string;
  organization?: string;
  communication_style?: string;
  negotiation_strengths?: string;
  notes?: string;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserPersonaCreate {
  name: string;
  role_title?: string;
  organization?: string;
  communication_style?: string;
  negotiation_strengths?: string;
  notes?: string;
  is_default?: boolean;
}

export interface UserPersonaUpdate {
  name?: string;
  role_title?: string;
  organization?: string;
  communication_style?: string;
  negotiation_strengths?: string;
  notes?: string;
  is_default?: boolean;
}

export interface PartnerPersona {
  id: string;
  created_by?: string;
  name: string;
  role_title?: string;
  company?: string;
  communication_style?: string;
  known_interests?: string;
  batna_estimate?: string;
  relationship_notes?: string;
  is_shared: boolean;
  created_at: string;
  updated_at: string;
}

export interface PartnerPersonaCreate {
  name: string;
  role_title?: string;
  company?: string;
  communication_style?: string;
  known_interests?: string;
  batna_estimate?: string;
  relationship_notes?: string;
  is_shared?: boolean;
}

export interface PartnerPersonaUpdate {
  name?: string;
  role_title?: string;
  company?: string;
  communication_style?: string;
  known_interests?: string;
  batna_estimate?: string;
  relationship_notes?: string;
  is_shared?: boolean;
}
