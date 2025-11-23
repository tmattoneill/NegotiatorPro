/**
 * Advanced copy button with format options (Markdown, HTML, Plain Text)
 * Shows dropdown menu on hover after 0.5s delay
 */
import { useState, useRef } from 'react';
import { Remarkable } from 'remarkable';

interface CopyButtonProps {
  content: string;
  className?: string;
}

type CopyFormat = 'markdown' | 'html' | 'text';

const md = new Remarkable({ html: true, breaks: true });

export default function CopyButton({ content, className = '' }: CopyButtonProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [copied, setCopied] = useState<CopyFormat | null>(null);
  const [hoverTimer, setHoverTimer] = useState<ReturnType<typeof setTimeout> | null>(null);
  const buttonRef = useRef<HTMLDivElement>(null);

  const convertToHTML = (markdown: string): string => {
    return md.render(markdown);
  };

  const convertToPlainText = (markdown: string): string => {
    // Remove markdown formatting
    let text = markdown;

    // Remove code blocks
    text = text.replace(/```[\s\S]*?```/g, (match) => {
      return match.replace(/```\w*\n?/g, '').replace(/```/g, '');
    });

    // Remove inline code
    text = text.replace(/`([^`]+)`/g, '$1');

    // Remove headers
    text = text.replace(/^#{1,6}\s+/gm, '');

    // Remove bold/italic
    text = text.replace(/(\*\*|__)(.*?)\1/g, '$2');
    text = text.replace(/(\*|_)(.*?)\1/g, '$2');

    // Remove links but keep text
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

    // Remove images
    text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');

    // Remove blockquotes
    text = text.replace(/^>\s+/gm, '');

    // Remove horizontal rules
    text = text.replace(/^(\*{3,}|-{3,}|_{3,})$/gm, '');

    // Remove list markers
    text = text.replace(/^\s*[-*+]\s+/gm, '');
    text = text.replace(/^\s*\d+\.\s+/gm, '');

    // Clean up extra whitespace
    text = text.replace(/\n{3,}/g, '\n\n');

    return text.trim();
  };

  const copyToClipboard = async (format: CopyFormat) => {
    let textToCopy: string;

    switch (format) {
      case 'html':
        textToCopy = convertToHTML(content);
        break;
      case 'text':
        textToCopy = convertToPlainText(content);
        break;
      case 'markdown':
      default:
        textToCopy = content;
        break;
    }

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(format);
      setShowDropdown(false);
      setTimeout(() => setCopied(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleMouseEnter = () => {
    const timer = setTimeout(() => {
      setShowDropdown(true);
    }, 500); // 0.5 second delay
    setHoverTimer(timer);
  };

  const handleMouseLeave = () => {
    if (hoverTimer) {
      clearTimeout(hoverTimer);
      setHoverTimer(null);
    }
    // Keep dropdown visible for 2 seconds after appearing, then fade out over 0.2s
    setTimeout(() => setShowDropdown(false), 2000);
  };

  const handleQuickCopy = () => {
    // Quick copy defaults to markdown
    if (!showDropdown) {
      copyToClipboard('markdown');
    }
  };

  return (
    <div
      ref={buttonRef}
      className={`copy-button-container ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <button
        className="copy-button"
        onClick={handleQuickCopy}
        title="Copy message"
        aria-label="Copy message to clipboard"
      >
        {copied ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
        )}
      </button>

      {showDropdown && !copied && (
        <div className="copy-dropdown">
          <button
            className="copy-dropdown-item"
            onClick={() => copyToClipboard('markdown')}
            title="Copy as Markdown"
          >
            <i className="fa-sharp fa-light fa-hashtag"></i>
            <span>Markdown</span>
          </button>
          <button
            className="copy-dropdown-item"
            onClick={() => copyToClipboard('html')}
            title="Copy as HTML"
          >
            <i className="fa-light fa-code"></i>
            <span>HTML</span>
          </button>
          <button
            className="copy-dropdown-item"
            onClick={() => copyToClipboard('text')}
            title="Copy as Plain Text"
          >
            <i className="fa-light fa-text-size"></i>
            <span>Plain Text</span>
          </button>
        </div>
      )}
    </div>
  );
}
