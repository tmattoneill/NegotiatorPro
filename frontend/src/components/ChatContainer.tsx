/**
 * Main chat container component - clean layout
 */
import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { sendChatMessage } from '../services/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import type { Message } from '../types';

export default function ChatContainer() {
  const { getCurrentSession, addMessage, isLoading, setLoading } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentSession = getCurrentSession();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages]);

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    addMessage(userMessage);
    setLoading(true);

    try {
      // Call API - using defaults for now
      const response = await sendChatMessage({
        question: content,
        use_premium_model: false,
        use_preprocessing: true,
      });

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        model_used: response.model_used,
        processing_time: response.processing_time,
      };
      addMessage(assistantMessage);
    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
        timestamp: new Date(),
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>{currentSession?.title || 'Select or create a conversation'}</h2>
      </div>

      <div className="messages-container">
        {!currentSession || currentSession.messages.length === 0 ? (
          <div className="welcome-message">
            <div>
              <h3>Welcome to NegotiatorPro</h3>
              <p>Start a conversation to get expert negotiation guidance</p>
            </div>
          </div>
        ) : (
          <>
            {currentSession.messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="message-header">NegotiatorPro</div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
    </div>
  );
}
