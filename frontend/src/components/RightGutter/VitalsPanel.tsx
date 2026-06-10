/**
 * VitalsPanel (expanded only)
 *
 * Three small tiles: ZOPA, BATNA strength, time pressure. Empty values show a
 * dash so the layout holds before the context call has inferred them.
 */
import type { Vitals } from '../../types/negotiationContext';

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function Tile({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-lg p-2" style={{ background: 'var(--color-surface)' }}>
      <p className="m-0 text-[9px] uppercase tracking-[0.05em]" style={{ color: 'var(--color-muted)' }}>
        {label}
      </p>
      <p className="mt-0.5 text-[13px] font-medium" style={{ color: 'var(--color-heading)' }}>
        {value ?? '—'}
      </p>
    </div>
  );
}

export default function VitalsPanel({ vitals }: { vitals: Vitals }) {
  return (
    <div>
      <p className="m-0 mb-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--color-muted)' }}>
        Vitals
      </p>
      <div className="grid grid-cols-3 gap-1.5">
        <Tile label="ZOPA" value={vitals.zopa} />
        <Tile label="BATNA" value={vitals.batna ? cap(vitals.batna) : null} />
        <Tile label="Time" value={vitals.time ? cap(vitals.time) : null} />
      </div>
    </div>
  );
}
