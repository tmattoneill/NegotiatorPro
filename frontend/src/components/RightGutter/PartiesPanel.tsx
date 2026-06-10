/**
 * PartiesPanel (expanded only)
 *
 * Two compact cards, one per side: avatar initial, name, and a one-line summary.
 */
import type { Parties, PartyCard } from '../../types/negotiationContext';

function Card({ party, avatarColor }: { party: PartyCard; avatarColor: string }) {
  return (
    <div className="rounded-lg p-2.5" style={{ background: 'var(--color-surface)' }}>
      <div className="mb-1.5 flex items-center gap-1.5">
        <div
          className="flex h-[22px] w-[22px] items-center justify-center rounded-full text-[10px] font-medium text-white"
          style={{ background: avatarColor }}
        >
          {party.initial}
        </div>
        <p className="m-0 text-[12px] font-medium" style={{ color: 'var(--color-heading)' }}>
          {party.name}
        </p>
      </div>
      <p className="m-0 text-[10px] leading-snug" style={{ color: 'var(--color-body)' }}>
        {party.summary}
      </p>
    </div>
  );
}

export default function PartiesPanel({ parties }: { parties: Parties }) {
  return (
    <div>
      <p className="m-0 mb-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--color-muted)' }}>
        Parties
      </p>
      <div className="grid grid-cols-2 gap-2">
        <Card party={parties.me} avatarColor="var(--leverage-mine)" />
        <Card party={parties.them} avatarColor="var(--leverage-theirs)" />
      </div>
    </div>
  );
}
