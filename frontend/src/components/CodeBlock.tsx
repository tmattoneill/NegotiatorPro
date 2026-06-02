/**
 * Advanced code block component with syntax highlighting, themes, collapse, and save
 */
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import {
  vscDarkPlus,
  materialDark,
  atomDark,
  oneDark,
  coldarkDark,
  duotoneDark,
} from 'react-syntax-highlighter/dist/esm/styles/prism';
import { FiCopy, FiCheck, FiChevronDown, FiChevronUp, FiDownload } from 'react-icons/fi';

interface CodeBlockProps {
  language: string;
  value: string;
  inline?: boolean;
  theme?: string;
}

const THEMES = {
  'VS Code Dark': vscDarkPlus,
  'Material Dark': materialDark,
  'Atom Dark': atomDark,
  'One Dark': oneDark,
  'Coldark Dark': coldarkDark,
  'Duotone Dark': duotoneDark,
};

const CODE_COLLAPSE_THRESHOLD = 20; // Collapse code blocks with more than 20 lines

export default function CodeBlock({ language, value, inline = false, theme = 'VS Code Dark' }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [currentTheme, setCurrentTheme] = useState(theme);

  const lineCount = value.split('\n').length;
  const shouldShowCollapse = lineCount > CODE_COLLAPSE_THRESHOLD;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const extension = getFileExtension(language);
    const blob = new Blob([value], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `code.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getFileExtension = (lang: string): string => {
    const extensions: { [key: string]: string } = {
      javascript: 'js',
      typescript: 'ts',
      python: 'py',
      java: 'java',
      cpp: 'cpp',
      c: 'c',
      csharp: 'cs',
      ruby: 'rb',
      go: 'go',
      rust: 'rs',
      php: 'php',
      swift: 'swift',
      kotlin: 'kt',
      sql: 'sql',
      html: 'html',
      css: 'css',
      json: 'json',
      yaml: 'yaml',
      markdown: 'md',
      bash: 'sh',
      shell: 'sh',
    };
    return extensions[lang.toLowerCase()] || 'txt';
  };

  const getLanguageLabel = (lang: string): string => {
    const labels: { [key: string]: string } = {
      javascript: 'JavaScript',
      typescript: 'TypeScript',
      python: 'Python',
      java: 'Java',
      cpp: 'C++',
      c: 'C',
      csharp: 'C#',
      ruby: 'Ruby',
      go: 'Go',
      rust: 'Rust',
      php: 'PHP',
      swift: 'Swift',
      kotlin: 'Kotlin',
      sql: 'SQL',
      html: 'HTML',
      css: 'CSS',
      json: 'JSON',
      yaml: 'YAML',
      markdown: 'Markdown',
      bash: 'Bash',
      shell: 'Shell',
    };
    return labels[lang.toLowerCase()] || lang.toUpperCase();
  };

  if (inline) {
    return <code className="inline-code">{value}</code>;
  }

  const selectedTheme = THEMES[currentTheme as keyof typeof THEMES] || vscDarkPlus;
  const displayValue = isCollapsed ? value.split('\n').slice(0, 10).join('\n') + '\n...' : value;

  return (
    <div className="code-block-wrapper">
      {/* Code block header */}
      <div className="code-block-header">
        <div className="code-language-info">
          <span className="code-language-badge">{getLanguageLabel(language)}</span>
          {lineCount > 1 && (
            <span className="code-line-count">{lineCount} lines</span>
          )}
        </div>

        <div className="code-actions">
          {/* Theme selector */}
          <select
            className="code-theme-selector"
            value={currentTheme}
            onChange={(e) => setCurrentTheme(e.target.value)}
            title="Change theme"
          >
            {Object.keys(THEMES).map((themeName) => (
              <option key={themeName} value={themeName}>
                {themeName}
              </option>
            ))}
          </select>

          {/* Collapse/Expand button */}
          {shouldShowCollapse && (
            <button
              className="code-action-button"
              onClick={() => setIsCollapsed(!isCollapsed)}
              title={isCollapsed ? 'Expand code' : 'Collapse code'}
            >
              {isCollapsed ? <FiChevronDown size={16} /> : <FiChevronUp size={16} />}
            </button>
          )}

          {/* Download button */}
          <button
            className="code-action-button"
            onClick={handleDownload}
            title="Download code"
          >
            <FiDownload size={16} />
          </button>

          {/* Copy button */}
          <button
            className="code-action-button code-copy-btn"
            onClick={handleCopy}
            title="Copy code"
          >
            {copied ? (
              <>
                <FiCheck size={16} />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <FiCopy size={16} />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code content */}
      <div className={`code-block-content ${isCollapsed ? 'collapsed' : ''}`}>
        <SyntaxHighlighter
          language={language}
          style={selectedTheme}
          showLineNumbers={!isCollapsed && lineCount > 1}
          wrapLines={true}
          wrapLongLines={true}
          customStyle={{
            margin: 0,
            borderRadius: '0 0 8px 8px',
            fontSize: '13px',
            lineHeight: '1.5',
            maxWidth: '100%',
            whiteSpace: 'pre-wrap',
            overflowWrap: 'anywhere',
          }}
        >
          {displayValue}
        </SyntaxHighlighter>
      </div>

      {/* Show more/less indicator */}
      {shouldShowCollapse && isCollapsed && (
        <div className="code-collapse-indicator">
          <button
            className="code-expand-button"
            onClick={() => setIsCollapsed(false)}
          >
            Show {lineCount - 10} more lines
          </button>
        </div>
      )}
    </div>
  );
}
