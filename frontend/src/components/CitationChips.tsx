/**
 * CitationChips
 *
 * The RAG sources behind an assistant reply, shown as small chips under the
 * bubble. Hovering a chip highlights the matching entry in the gutter's sources
 * list (and vice versa) through the shared hoveredSource in the UI store.
 */
import type { SourceCitation } from '../types/negotiationContext';
import { useUIStore } from '../store/uiStore';

interface CitationChipsProps {
  sources: SourceCitation[];
}

export default function CitationChips({ sources }: CitationChipsProps) {
  const hoveredSource = useUIStore((s) => s.hoveredSource);
  const setHoveredSource = useUIStore((s) => s.setHoveredSource);

  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {sources.map((src, i) => {
        const isHot = hoveredSource === src.title;
        return (
          <span
            key={`${src.title}-${i}`}
            onMouseEnter={() => setHoveredSource(src.title)}
            onMouseLeave={() => setHoveredSource(null)}
            title={src.quote || src.sub || src.title}
            className="cursor-default rounded px-1.5 py-0.5 text-[10px] transition-colors"
            style={{
              background: isHot
                ? 'color-mix(in oklch, var(--score-3) 18%, transparent)'
                : 'var(--color-surface)',
              color: isHot ? 'var(--color-heading)' : 'var(--color-muted)',
              border: isHot ? '1px solid var(--score-3)' : '1px solid transparent',
            }}
          >
            {src.title}
          </span>
        );
      })}
    </div>
  );
}
