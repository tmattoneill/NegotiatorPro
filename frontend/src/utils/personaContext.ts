/**
 * Persona context measurement — mirror of backend/persona_text.py.
 *
 * The backend is the authoritative gate (returns 422 below the minimum); this is
 * for live UX only (a character counter and submit-gating). Keep the placeholder
 * set, the field lists, and MIN_PERSONA_CHARS in sync with the server.
 */
export const MIN_PERSONA_CHARS = 240;

const PLACEHOLDER_VALUES = new Set([
  'n/a', 'na', 'n.a.', '-', '—', 'none', 'tbd', 'nil', '?',
]);

function meaningful(value?: string | null): string {
  if (!value) return '';
  const v = value.trim();
  return PLACEHOLDER_VALUES.has(v.toLowerCase()) ? '' : v;
}

/** Total length of the meaningful (non-placeholder) values, trimmed. */
export function meaningfulLen(...values: Array<string | null | undefined>): number {
  return values.reduce((sum: number, v) => sum + meaningful(v).length, 0);
}
