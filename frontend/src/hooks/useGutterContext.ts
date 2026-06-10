/**
 * useGutterContext
 *
 * Assembles the NegotiationContext that feeds the right gutter. PLEASE is live
 * and conversation-level: it is the running mean of every scored assistant turn
 * in the current session, so it reads as the overall health of the conversation
 * rather than a single reply. Leverage, parties, vitals, and sources are still
 * mocked; they go live in later phases, at which point their mock fallbacks come
 * out.
 */
import { useMemo } from 'react';
import { useChatStore } from '../store/chatStore';
import type { NegotiationContext, PleaseScore, ScoreValue } from '../types/negotiationContext';
import { MOCK_CONTEXT } from '../components/RightGutter/mockContext';

const LETTER_CODE: Record<keyof RunningTotals, string> = {
  polite: 'P',
  logical: 'L',
  empathetic: 'E',
  assertive: 'A',
  strategic: 'S',
  engaging: 'E',
};

type RunningTotals = {
  polite: number;
  logical: number;
  empathetic: number;
  assertive: number;
  strategic: number;
  engaging: number;
};

const DIMENSIONS = Object.keys(LETTER_CODE) as (keyof RunningTotals)[];

const clampScore = (value: number): ScoreValue =>
  Math.min(5, Math.max(1, Math.round(value))) as ScoreValue;

/** Average each PLEASE dimension across every scored turn, then derive the
 * total and soft spots from the rounded means. Returns null if nothing scored. */
function runningAverage(scores: PleaseScore[]): PleaseScore | null {
  if (scores.length === 0) return null;

  const sums: RunningTotals = {
    polite: 0, logical: 0, empathetic: 0, assertive: 0, strategic: 0, engaging: 0,
  };
  for (const score of scores) {
    for (const dim of DIMENSIONS) sums[dim] += score[dim];
  }

  const means = {} as Record<keyof RunningTotals, ScoreValue>;
  for (const dim of DIMENSIONS) means[dim] = clampScore(sums[dim] / scores.length);

  const total = DIMENSIONS.reduce((acc, dim) => acc + means[dim], 0);

  // Soft spots: the dimensions sitting in the lowest two score tiers, weakest
  // first, deduplicated by letter. Empty when every dimension is equal.
  const distinct = [...new Set(DIMENSIONS.map((d) => means[d]))].sort((a, b) => a - b);
  let weakest: string[] = [];
  if (distinct.length > 1) {
    const softTiers = new Set(distinct.slice(0, 2));
    for (const tier of distinct) {
      if (!softTiers.has(tier)) continue;
      for (const dim of DIMENSIONS) {
        const code = LETTER_CODE[dim];
        if (means[dim] === tier && !weakest.includes(code)) weakest.push(code);
      }
    }
  }

  return { ...means, total, weakest };
}

export function useGutterContext(): NegotiationContext {
  const sessions = useChatStore((s) => s.sessions);
  const currentSessionId = useChatStore((s) => s.currentSessionId);

  const runningPlease = useMemo<PleaseScore | null>(() => {
    const session = sessions.find((s) => s.id === currentSessionId);
    if (!session) return null;
    const scores = session.messages
      .filter((m) => m.role === 'assistant' && m.please)
      .map((m) => m.please as PleaseScore);
    return runningAverage(scores);
  }, [sessions, currentSessionId]);

  return {
    ...MOCK_CONTEXT,
    please: runningPlease,
  };
}
