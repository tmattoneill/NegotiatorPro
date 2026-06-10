/**
 * Mocked gutter data for Phase 1, matching the values in the HTML mockup. Real
 * data replaces this as each backend phase lands (PLEASE, sources, then the
 * negotiation-level leverage/parties/vitals).
 */
import type { NegotiationContext } from '../../types/negotiationContext';

export const MOCK_CONTEXT: NegotiationContext = {
  please: {
    polite: 4,
    logical: 5,
    empathetic: 3,
    assertive: 2,
    strategic: 4,
    engaging: 3,
    total: 21,
    weakest: ['A', 'E'],
  },
  leverage: {
    mineWeight: 0.58,
    theirsWeight: 0.42,
    summary: 'slight tilt — you',
    mine: [
      { label: 'Strong BATNA', confidence: 'high' },
      { label: 'Time on your side', confidence: 'high' },
      { label: 'Domain expertise', confidence: 'high' },
    ],
    theirs: [
      { label: 'Budget authority', confidence: 'high' },
      { label: 'Multiple vendors', confidence: 'high' },
      { label: 'Internal champion', confidence: 'low' },
    ],
  },
  parties: {
    me: { name: 'Matt', initial: 'M', summary: 'Founder. Strong BATNA, willing to walk.' },
    them: { name: 'Partner Test', initial: 'P', summary: 'Procurement-led. Comparing vendors.' },
  },
  vitals: {
    zopa: '£42–58k',
    batna: 'strong',
    time: 'loose',
  },
  sources: [
    {
      title: 'Voss · Never Split the Difference',
      sub: 'Mirroring · ch. 3',
      quote: '"Repeat the last three words. Then shut up."',
      active: true,
    },
    {
      title: 'Fisher & Ury · Getting to Yes',
      sub: 'Separate the people from the problem · p. 84',
      active: false,
    },
    {
      title: 'Ury · Getting Past No',
      sub: 'ch. 2',
      active: false,
    },
  ],
};
