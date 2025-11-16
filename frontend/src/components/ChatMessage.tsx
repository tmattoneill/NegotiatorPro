/**
 * Chat message component - clean bubble design with Markdown support
 */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-header">{isUser ? 'You' : 'NegotiatorPro'}</div>
      <div className="message-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>
      </div>
      {!isUser && message.model_used && (
        <div className="message-meta">
          Model: {message.model_used}
          {message.processing_time && ` • ${message.processing_time}s`}
        </div>
      )}
    </div>
  );
}
