/**
 * SourcesPanel (both states)
 *
 * The RAG citations for the latest turn. The active source gets an amber left
 * rail and shows a one-line quote; hovering any source expands its passage.
 * Hover is shared through the UI store, so hovering a citation chip in chat
 * highlights the matching source here, and vice versa.
 */
import type { SourceCitation } from '../../types/negotiationContext';
import { useUIStore } from '../../store/uiStore';

interface SourcesPanelProps {
  sources: SourceCitation[];
}

export default function SourcesPanel({ sources }: SourcesPanelProps) {
  const hoveredSource = useUIStore((s) => s.hoveredSource);
  const setHoveredSource = useUIStore((s) => s.setHoveredSource);

  return (
    <div>
      <p className="m-0 mb-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--color-muted)' }}>
        Sources in play
      </p>
      {sources.map((src, i) => {
        // Hover wins over the resting active source for the highlight, so only
        // one row is emphasised at a time.
        const isActive = hoveredSource ? src.title === hoveredSource : src.active;
        // But keep the resting-active quote visible even while another row is
        // hovered — collapsing it would shift the list under the cursor and set
        // off a hover enter/leave loop. Only the hovered row grows, downward.
        const showQuote = Boolean(src.quote) && (src.active || src.title === hoveredSource);
        return (
          <div
            key={`${src.title}-${i}`}
            onMouseEnter={() => setHoveredSource(src.title)}
            onMouseLeave={() => setHoveredSource(null)}
            className="mb-1.5 py-1.5 pl-2.5 transition-colors"
            style={{
              borderLeft: '2px solid',
              borderLeftColor: isActive ? 'var(--score-3)' : 'var(--color-border)',
              background: isActive ? 'color-mix(in oklch, var(--score-3) 10%, transparent)' : 'transparent',
              borderRadius: isActive ? '0 8px 8px 0' : '0',
            }}
          >
            <p className="m-0 text-[11px] font-medium" style={{ color: 'var(--color-heading)' }}>
              {src.title}
            </p>
            {src.sub && (
              <p className="mt-px text-[10px]" style={{ color: 'var(--color-muted)' }}>
                {src.sub}
              </p>
            )}
            {showQuote && (
              <p className="mt-1 text-[10px] italic leading-snug" style={{ color: 'var(--color-muted)' }}>
                {src.quote}
              </p>
            )}
          </div>
        );
      })}
      {sources.length === 0 && (
        <p className="text-[11px] italic" style={{ color: 'var(--color-muted)' }}>
          No sources cited yet.
        </p>
      )}
    </div>
  );
}
