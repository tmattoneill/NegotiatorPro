/**
 * RightGutter
 *
 * The persistent stats gutter. Two widths: collapsed (~240px, PLEASE + Sources)
 * and expanded (~420px, adds Leverage, Parties, Vitals). A chevron on the
 * leading edge toggles between them. Presentational — fed a NegotiationContext.
 * Mirrors the np-gutter block in mockups-ui/dialogue-gutter.html.
 */
import { TbChevronLeft, TbChevronRight } from 'react-icons/tb';
import type { NegotiationContext } from '../../types/negotiationContext';
import { useUIStore } from '../../store/uiStore';
import PleaseGauges from './PleaseGauges';
import LeveragePanel from './LeveragePanel';
import PartiesPanel from './PartiesPanel';
import VitalsPanel from './VitalsPanel';
import SourcesPanel from './SourcesPanel';

interface RightGutterProps {
  context: NegotiationContext;
  hoveredSource?: string | null;
  onHoverSource?: (title: string | null) => void;
}

const COLLAPSED_WIDTH = 240;
const EXPANDED_WIDTH = 420;

export default function RightGutter({ context, hoveredSource, onHoverSource }: RightGutterProps) {
  const expanded = useUIStore((s) => s.gutterExpanded);
  const toggleGutter = useUIStore((s) => s.toggleGutter);

  const width = expanded ? EXPANDED_WIDTH : COLLAPSED_WIDTH;

  return (
    // Outer frame stays overflow-visible so the toggle can sit on the leading
    // edge without being clipped; the inner wrapper owns the scroll.
    <div
      className="relative shrink-0 transition-[width] duration-200 ease-out"
      style={{
        width,
        background: 'var(--color-background)',
        borderLeft: '1px solid var(--color-border)',
        boxShadow: expanded ? '-8px 0 24px rgba(0,0,0,0.04)' : 'none',
      }}
    >
      <button
        type="button"
        onClick={toggleGutter}
        title={expanded ? 'Collapse panel' : 'Expand panel'}
        className="absolute left-[-11px] top-3.5 z-10 flex h-[22px] w-[22px] items-center justify-center rounded-full"
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-muted)',
        }}
      >
        {expanded ? <TbChevronRight className="text-[13px]" /> : <TbChevronLeft className="text-[13px]" />}
      </button>

      <div className="flex h-full flex-col gap-4 overflow-y-auto px-4 py-4">
        {context.please ? (
          <PleaseGauges please={context.please} expanded={expanded} />
        ) : (
          <div>
            <p
              className="m-0 mb-2 text-[10px] uppercase tracking-[0.08em]"
              style={{ color: 'var(--color-muted)' }}
            >
              PLEASE
            </p>
            <p className="text-[11px] leading-snug" style={{ color: 'var(--color-muted)' }}>
              Send a message to start the running PLEASE score for this conversation.
            </p>
          </div>
        )}

        {expanded && context.leverage && (
          <LeveragePanel
            leverage={context.leverage}
            meName={context.parties?.me.name ?? 'You'}
            themName={context.parties?.them.name ?? 'Partner'}
          />
        )}

        {expanded && context.parties && <PartiesPanel parties={context.parties} />}

        {expanded && context.vitals && <VitalsPanel vitals={context.vitals} />}

        <SourcesPanel
          sources={context.sources}
          hoveredTitle={hoveredSource}
          onHoverSource={onHoverSource}
        />
      </div>
    </div>
  );
}
