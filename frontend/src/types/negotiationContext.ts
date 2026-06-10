/**
 * Negotiation Context Types
 *
 * The data behind the right-hand stats gutter. PLEASE and sources arrive per
 * assistant response (on the chat payload); leverage, parties, and vitals are
 * negotiation-level and fetched lazily from the context endpoint.
 *
 * Mirrors the backend Pydantic models in backend/api/models/responses.py.
 */

export type PleaseLetter = 'P' | 'L' | 'E1' | 'A' | 'S' | 'E2';

export type ScoreValue = 1 | 2 | 3 | 4 | 5;

export interface PleaseScore {
  polite: ScoreValue;
  logical: ScoreValue;
  empathetic: ScoreValue;
  assertive: ScoreValue;
  strategic: ScoreValue;
  engaging: ScoreValue;
  total: number;        // sum, 6–30
  weakest: string[];    // letter codes of the lowest-scoring elements
}

export interface LeverageEdge {
  label: string;
  confidence: 'high' | 'low';
}

export interface Leverage {
  mineWeight: number;   // 0–1, sums with theirsWeight to 1
  theirsWeight: number;
  mine: LeverageEdge[];
  theirs: LeverageEdge[];
  summary: string;      // e.g. "slight tilt — you"
}

export interface PartyCard {
  name: string;
  initial: string;
  summary: string;
}

export interface Parties {
  me: PartyCard;
  them: PartyCard;
}

export interface Vitals {
  zopa: string | null;                          // e.g. "£42–58k"
  batna: 'strong' | 'moderate' | 'weak' | null;
  time: 'urgent' | 'tight' | 'loose' | null;
}

export interface SourceCitation {
  title: string;        // e.g. "Voss · Never Split the Difference"
  sub: string;          // e.g. "Mirroring · ch. 3"
  quote?: string;       // pulled passage for the active source
  active: boolean;
}

/**
 * The full gutter payload. `please` and `sources` come from the latest assistant
 * message; `leverage`, `parties`, and `vitals` come from the context endpoint.
 */
export interface NegotiationContext {
  please: PleaseScore | null;
  leverage: Leverage | null;
  parties: Parties | null;
  vitals: Vitals | null;
  sources: SourceCitation[];
}
