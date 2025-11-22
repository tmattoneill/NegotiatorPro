/**
 * Main chat container component - clean layout
 */
import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { sendChatMessage } from '../services/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import type { Message } from '../types';

export default function ChatContainer() {
  const { getCurrentSession, addMessage, isLoading, setLoading } = useChatStore();
  const { selectedProvider, selectedModel, usePremiumModel, usePreprocessing, availableModels } = useSettingsStore();

  // Get display names for provider and model
  const providerDisplayName = selectedProvider && availableModels[selectedProvider]?.name
    ? availableModels[selectedProvider].name
    : selectedProvider || 'Unknown';
  const modelDisplayName = selectedModel || 'Unknown';
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentSession = getCurrentSession();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages]);

  const handleSendMessage = async (content: string) => {
    // Don't send if no active conversation
    if (!currentSession?.id) {
      return;
    }

    // Add user message (optimistic update)
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    addMessage(userMessage);
    setLoading(true);

    try {
      // Call API with settings from store and conversation_id
      const response = await sendChatMessage({
        question: content,
        conversation_id: currentSession.id,  // Pass conversation ID to save to database
        use_premium_model: usePremiumModel,
        use_preprocessing: usePreprocessing,
        provider: usePremiumModel ? undefined : selectedProvider || undefined,
        model: usePremiumModel ? undefined : selectedModel || undefined,
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
      // Extract user-friendly error message from backend response
      let errorContent = 'Failed to get response. Please try again.';

      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as { response?: { data?: { detail?: string | string[] } } };
        const detail = axiosError.response?.data?.detail;

        if (detail) {
          // Handle both single string and array of errors
          errorContent = Array.isArray(detail) ? detail.join('\n') : detail;
        }
      } else if (error instanceof Error) {
        errorContent = error.message;
      }

      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ ${errorContent}`,
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
        <span className="provider-model-display">{providerDisplayName} / {modelDisplayName}</span>
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
