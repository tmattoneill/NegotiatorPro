/**
 * Chat message component - clean bubble design with Markdown support, copy button, and advanced code highlighting
 */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import CodeBlock from './CodeBlock';
import MermaidDiagram from './MermaidDiagram';
import CopyButton from './CopyButton';
import type { Message } from '../types';
import 'katex/dist/katex.min.css';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  const formatTimestamp = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - new Date(date).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins === 1) return '1 minute ago';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours === 1) return '1 hour ago';
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays === 1) return 'yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;

    return new Date(date).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-header-row">
        <div className="message-header-left">
          <div className="message-header">{isUser ? 'You' : 'NegotiatorPro'}</div>
          <div className="message-timestamp" title={new Date(message.timestamp).toLocaleString()}>
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
        <CopyButton content={message.content} />
      </div>
      <div className="message-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            code(props) {
              const { children, className } = props;
              const match = /language-(\w+)/.exec(className || '');
              const language = match ? match[1] : '';
              const isInline = !className;
              const codeValue = String(children).replace(/\n$/, '');

              // Handle Mermaid diagrams
              if (!isInline && (language === 'mermaid' || language === 'mmd')) {
                return <MermaidDiagram chart={codeValue} />;
              }

              // Handle regular code blocks
              if (!isInline && language) {
                return (
                  <CodeBlock
                    language={language}
                    value={codeValue}
                  />
                );
              }

              // Handle inline code
              return (
                <CodeBlock
                  language="text"
                  value={String(children)}
                  inline={true}
                />
              );
            },
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
      {!isUser && message.model_used && (
        <div className="message-meta">
          <div className="message-meta-item">
            <span className="message-meta-label">Model:</span>
            <span className="message-meta-value">{message.model_used}</span>
          </div>
          {message.processing_time && (
            <div className="message-meta-item">
              <span className="message-meta-label">Time:</span>
              <span className="message-meta-value">{message.processing_time.toFixed(2)}s</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
