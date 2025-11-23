/**
 * Mermaid diagram renderer component
 * Renders flowcharts, sequence diagrams, and other Mermaid visualizations
 */
import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
}

// Initialize Mermaid with configuration
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'monospace',
});

export default function MermaidDiagram({ chart }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [rendered, setRendered] = useState(false);

  useEffect(() => {
    const renderDiagram = async () => {
      if (!containerRef.current || rendered) return;

      try {
        // Generate unique ID for this diagram
        const id = `mermaid-${Math.random().toString(36).substring(7)}`;

        // Render the diagram
        const { svg } = await mermaid.render(id, chart);

        // Insert the SVG into the container
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
          setRendered(true);
          setError(null);
        }
      } catch (err) {
        console.error('Error rendering Mermaid diagram:', err);
        setError('Failed to render diagram');
      }
    };

    renderDiagram();
  }, [chart, rendered]);

  if (error) {
    return (
      <div className="mermaid-error">
        <div className="mermaid-error-icon">⚠️</div>
        <div className="mermaid-error-text">{error}</div>
        <pre className="mermaid-error-code">{chart}</pre>
      </div>
    );
  }

  return <div ref={containerRef} className="mermaid-diagram" />;
}
