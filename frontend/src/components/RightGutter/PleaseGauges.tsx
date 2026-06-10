/**
 * PleaseGauges
 *
 * Six radial gauges for the PLEASE self-assessment. Collapsed: 3×2 grid.
 * Expanded: a single row of six, plus a /30 threshold bar with a tick at 20
 * (the rewrite line) and a one-line read of the weakest letters.
 */
import type { PleaseScore, ScoreValue } from '../../types/negotiationContext';

const SCORE_TOKEN: Record<ScoreValue, string> = {
  5: 'var(--score-5)',
  4: 'var(--score-4)',
  3: 'var(--score-3)',
  2: 'var(--score-2)',
  1: 'var(--score-1)',
};

// The mockup draws each arc as score×16 against a 107-unit circumference, which
// leaves a deliberate gap at full score. Kept identical so the gauge reads the
// same as the reference.
const ARC_MAX = 107;
const arcFor = (score: ScoreValue) => `${score * 16} ${ARC_MAX}`;

interface LetterGauge {
  code: string;
  label: string;
  score: ScoreValue;
}

function Gauge({ gauge, size }: { gauge: LetterGauge; size: number }) {
  return (
    <div>
      <svg viewBox="0 0 44 44" width={size} height={size} className="mx-auto block">
        <circle cx="22" cy="22" r="17" fill="none" stroke="var(--color-border)" strokeWidth="3" />
        <circle
          cx="22"
          cy="22"
          r="17"
          fill="none"
          stroke={SCORE_TOKEN[gauge.score]}
          strokeWidth="3"
          strokeDasharray={arcFor(gauge.score)}
          strokeLinecap="round"
          transform="rotate(-90 22 22)"
        />
        <text
          x="22"
          y="25"
          textAnchor="middle"
          className="text-[11px] font-medium"
          fill="var(--color-heading)"
        >
          {gauge.score}
        </text>
      </svg>
      <p className="mt-1 text-center text-[10px] font-medium" style={{ color: 'var(--color-heading)' }}>
        {gauge.code}
      </p>
      <p className="text-center text-[9px]" style={{ color: 'var(--color-muted)' }}>
        {gauge.label}
      </p>
    </div>
  );
}

interface PleaseGaugesProps {
  please: PleaseScore;
  expanded: boolean;
}

export default function PleaseGauges({ please, expanded }: PleaseGaugesProps) {
  const gauges: LetterGauge[] = [
    { code: 'P', label: 'Polite', score: please.polite },
    { code: 'L', label: 'Logical', score: please.logical },
    { code: 'E', label: expanded ? 'Empathetic' : 'Empath.', score: please.empathetic },
    { code: 'A', label: expanded ? 'Assertive' : 'Assert.', score: please.assertive },
    { code: 'S', label: 'Strategic', score: please.strategic },
    { code: 'E', label: 'Engaging', score: please.engaging },
  ];

  // Threshold bar (expanded only): fill is total/30, the tick sits at 20/30.
  const fillPct = Math.min(100, (please.total / 30) * 100);
  const thresholdPct = (20 / 30) * 100;
  const aboveThreshold = please.total >= 20;
  const weakestLabels = please.weakest.join(' and ');

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <p
          className="m-0 text-[10px] uppercase tracking-[0.08em]"
          style={{ color: 'var(--color-muted)' }}
        >
          PLEASE
        </p>
        <span className="text-[11px] font-medium" style={{ color: 'var(--color-heading)' }}>
          {please.total} / 30
        </span>
      </div>

      {expanded && (
        <div
          className="relative mb-3 h-1 overflow-hidden rounded-sm"
          style={{ background: 'var(--color-surface)' }}
        >
          <div
            className="absolute left-0 top-0 h-full"
            style={{ width: `${fillPct}%`, background: 'var(--score-4)' }}
          />
          <div
            className="absolute top-[-2px] h-2 w-px"
            style={{ left: `${thresholdPct}%`, background: 'var(--threshold)' }}
            title="health line: 20"
          />
        </div>
      )}

      <div
        className={expanded ? 'grid grid-cols-6 gap-1' : 'grid grid-cols-3 gap-1.5'}
      >
        {gauges.map((g, i) => (
          <Gauge key={`${g.code}-${i}`} gauge={g} size={expanded ? 46 : 42} />
        ))}
      </div>

      {expanded && (
        <p className="mt-2.5 text-[10px] leading-snug" style={{ color: 'var(--color-muted)' }}>
          {aboveThreshold ? 'Healthy overall (above 20).' : 'Below the 20 health line.'}
          {please.weakest.length > 0 && ` ${weakestLabels} ${please.weakest.length > 1 ? 'are' : 'is'} the soft ${please.weakest.length > 1 ? 'spots' : 'spot'}.`}
        </p>
      )}
    </div>
  );
}
