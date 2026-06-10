/**
 * LeveragePanel (expanded only)
 *
 * A split bar showing where leverage sits — left is the user, right is the
 * partner, with a centre tick for parity. Below it, two columns of inferred
 * edges. Low-confidence edges render muted rather than in the strong colour.
 */
import { TbArrowUpRight } from 'react-icons/tb';
import type { Leverage, LeverageEdge } from '../../types/negotiationContext';

function EdgeRow({ edge, side }: { edge: LeverageEdge; side: 'mine' | 'theirs' }) {
  const strong = side === 'mine' ? 'var(--leverage-mine)' : 'var(--leverage-theirs)';
  const color = edge.confidence === 'low' ? 'var(--color-muted)' : strong;
  return (
    <p className="m-0 mb-0.5 flex items-center gap-1 text-[11px]" style={{ color: 'var(--color-body)' }}>
      <TbArrowUpRight
        className="text-[11px]"
        style={{ color, transform: side === 'theirs' ? 'scaleX(-1)' : undefined }}
      />
      {edge.label}
    </p>
  );
}

export default function LeveragePanel({
  leverage,
  meName,
  themName,
}: {
  leverage: Leverage;
  meName: string;
  themName: string;
}) {
  const minePct = Math.round(leverage.mineWeight * 100);
  const theirsPct = 100 - minePct;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <p className="m-0 text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--color-muted)' }}>
          Leverage
        </p>
        <p className="m-0 text-[10px]" style={{ color: 'var(--color-muted)' }}>
          {leverage.summary}
        </p>
      </div>

      <div className="mb-1 flex justify-between text-[11px]">
        <span className="font-medium" style={{ color: 'var(--color-heading)' }}>{meName}</span>
        <span style={{ color: 'var(--color-muted)' }}>{themName}</span>
      </div>

      <div className="relative h-2 overflow-hidden rounded" style={{ background: 'var(--color-surface)' }}>
        <div
          className="absolute left-0 top-0 h-full"
          style={{ width: `${minePct}%`, background: 'var(--leverage-mine)' }}
        />
        <div
          className="absolute right-0 top-0 h-full opacity-65"
          style={{ width: `${theirsPct}%`, background: 'var(--leverage-theirs)' }}
        />
        <div
          className="absolute top-[-2px] left-1/2 h-3 w-px"
          style={{ background: 'var(--color-heading)' }}
        />
      </div>

      <div className="mt-2.5 grid grid-cols-2 gap-3">
        <div>
          <p className="m-0 mb-1 text-[10px] uppercase tracking-[0.05em]" style={{ color: 'var(--color-muted)' }}>
            Your edge
          </p>
          {leverage.mine.map((e, i) => (
            <EdgeRow key={`mine-${i}`} edge={e} side="mine" />
          ))}
        </div>
        <div>
          <p className="m-0 mb-1 text-[10px] uppercase tracking-[0.05em]" style={{ color: 'var(--color-muted)' }}>
            Their edge
          </p>
          {leverage.theirs.map((e, i) => (
            <EdgeRow key={`theirs-${i}`} edge={e} side="theirs" />
          ))}
        </div>
      </div>
    </div>
  );
}
