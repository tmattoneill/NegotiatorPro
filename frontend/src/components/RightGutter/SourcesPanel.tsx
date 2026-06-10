/**
 * SourcesPanel (both states)
 *
 * Recent RAG citations. The active source gets an amber left rail and shows a
 * one-line quote; hovering any source expands its passage. Hovering a source
 * also reports back so the matching citation chip in chat can highlight, and a
 * chip hovered in chat highlights the matching source here.
 */
import { useState } from 'react';
import type { SourceCitation } from '../../types/negotiationContext';

interface SourcesPanelProps {
  sources: SourceCitation[];
  hoveredTitle?: string | null;
  onHoverSource?: (title: string | null) => void;
}

export default function SourcesPanel({ sources, hoveredTitle, onHoverSource }: SourcesPanelProps) {
  const [localHover, setLocalHover] = useState<string | null>(null);

  // Hover takes precedence: while a source is hovered (here or via a chat chip),
  // only that one is active, so the default active source doesn't double up.
  const activeHover = localHover ?? hoveredTitle ?? null;

  return (
    <div>
      <p className="m-0 mb-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--color-muted)' }}>
        Sources in play
      </p>
      {sources.map((src, i) => {
        const isActive = activeHover ? src.title === activeHover : src.active;
        const showQuote = src.quote && isActive;
        return (
          <div
            key={`${src.title}-${i}`}
            onMouseEnter={() => {
              setLocalHover(src.title);
              onHoverSource?.(src.title);
            }}
            onMouseLeave={() => {
              setLocalHover(null);
              onHoverSource?.(null);
            }}
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
            <p className="mt-px text-[10px]" style={{ color: 'var(--color-muted)' }}>
              {src.sub}
            </p>
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
