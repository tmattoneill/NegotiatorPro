/**
 * Portal — renders children at document.body, outside the component tree.
 *
 * Modals must not inherit styles from wherever they're declared (e.g. the
 * sidebar sets text-white, which bled into modal inputs). Rendering through a
 * portal at the document root keeps every modal visually self-contained.
 */
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

interface PortalProps {
  children: React.ReactNode;
}

export default function Portal({ children }: PortalProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!mounted) return null;
  return createPortal(children, document.body);
}
