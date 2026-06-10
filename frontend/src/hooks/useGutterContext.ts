/**
 * useGutterContext
 *
 * Assembles the NegotiationContext that feeds the right gutter. PLEASE is live —
 * it reads the most recent assistant turn that carried a score. Leverage,
 * parties, vitals, and sources are still mocked; they go live in later phases,
 * at which point their mock fallbacks here come out.
 */
import { useMemo } from 'react';
import { useChatStore } from '../store/chatStore';
import type { NegotiationContext, PleaseScore } from '../types/negotiationContext';
import { MOCK_CONTEXT } from '../components/RightGutter/mockContext';

export function useGutterContext(): NegotiationContext {
  const sessions = useChatStore((s) => s.sessions);
  const currentSessionId = useChatStore((s) => s.currentSessionId);

  const livePlease = useMemo<PleaseScore | null>(() => {
    const session = sessions.find((s) => s.id === currentSessionId);
    if (!session) return null;
    // Walk back to the latest assistant message that carried a PLEASE score.
    for (let i = session.messages.length - 1; i >= 0; i -= 1) {
      const message = session.messages[i];
      if (message.role === 'assistant' && message.please) {
        return message.please;
      }
    }
    return null;
  }, [sessions, currentSessionId]);

  return {
    ...MOCK_CONTEXT,
    please: livePlease,
  };
}
