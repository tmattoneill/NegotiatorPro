/**
 * useGutterContext
 *
 * Assembles the NegotiationContext that feeds the right gutter, all live:
 * - PLEASE: the running mean of every scored assistant turn in the session.
 * - sources: the citations from the most recent turn that retrieved any.
 * - leverage / parties / vitals: fetched from the context endpoint, lazily when
 *   the gutter is expanded and refreshed as new turns land.
 */
import { useEffect, useMemo } from 'react';
import { useChatStore } from '../store/chatStore';
import { useContextStore } from '../store/contextStore';
import { useUIStore } from '../store/uiStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { useAuthStore } from '../store/authStore';
import type {
  NegotiationContext,
  PleaseScore,
  ScoreValue,
  SourceCitation,
} from '../types/negotiationContext';

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
  const currentNegotiationId = useNegotiationStore((s) => s.currentNegotiationId);
  const userId = useAuthStore((s) => s.user?.id);
  const gutterExpanded = useUIStore((s) => s.gutterExpanded);

  const contextData = useContextStore((s) => s.data);
  const fetchContext = useContextStore((s) => s.fetchContext);
  const clearContext = useContextStore((s) => s.clear);

  // Drop the previous negotiation's leverage/parties/vitals on switch so they
  // don't linger while the new one loads.
  useEffect(() => {
    clearContext();
  }, [currentNegotiationId, clearContext]);

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentSessionId),
    [sessions, currentSessionId],
  );
  const messageCount = currentSession?.messages.length ?? 0;

  // Fetch leverage/parties/vitals only when the gutter is expanded (that is the
  // only state that shows them). Re-runs when the turn count changes; the store
  // and backend both dedupe, so repeats are cheap.
  useEffect(() => {
    if (gutterExpanded && currentNegotiationId && userId) {
      fetchContext(currentNegotiationId, userId, messageCount);
    }
  }, [gutterExpanded, currentNegotiationId, userId, messageCount, fetchContext]);

  const runningPlease = useMemo<PleaseScore | null>(() => {
    if (!currentSession) return null;
    const scores = currentSession.messages
      .filter((m) => m.role === 'assistant' && m.please)
      .map((m) => m.please as PleaseScore);
    return runningAverage(scores);
  }, [currentSession]);

  // Sources are per-turn: the citations from the most recent assistant turn
  // that retrieved any.
  const liveSources = useMemo<SourceCitation[]>(() => {
    if (!currentSession) return [];
    for (let i = currentSession.messages.length - 1; i >= 0; i -= 1) {
      const message = currentSession.messages[i];
      if (message.role === 'assistant' && message.sources && message.sources.length > 0) {
        return message.sources;
      }
    }
    return [];
  }, [currentSession]);

  return {
    please: runningPlease,
    sources: liveSources,
    leverage: contextData?.leverage ?? null,
    parties: contextData?.parties ?? null,
    vitals: contextData?.vitals ?? null,
  };
}
