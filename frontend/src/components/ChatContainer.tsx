/**
 * Main chat container component - clean layout
 */
import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { useAuthStore } from '../store/authStore';
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

  const handleSendMessage = async (content: string, files?: File[]) => {
    // Auto-create conversation if none exists but we have a negotiation
    const { negotiations, currentNegotiationId } = useNegotiationStore.getState();
    const currentNegotiation = negotiations.find(n => n.id === currentNegotiationId);
    const { user } = useAuthStore.getState();

    if (!currentSession?.id) {
      // Check if we have a negotiation to create a conversation for
      if (!currentNegotiation?.id) {
        console.warn('No negotiation selected - cannot send message');
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: '⚠️ Please select a negotiation from the dropdown first.',
          timestamp: new Date(),
        };
        addMessage(errorMessage);
        return;
      }

      // Auto-create a new conversation
      console.log('Auto-creating conversation for negotiation:', currentNegotiation.id);
      const { createNewSession } = useChatStore.getState();
      await createNewSession(currentNegotiation.id, user?.id || '');

      // Give it a moment for state to update
      await new Promise(resolve => setTimeout(resolve, 100));

      // Get the newly created session
      const updatedSession = getCurrentSession();
      if (!updatedSession?.id) {
        console.error('Failed to create conversation');
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: '❌ Failed to create conversation. Please try clicking "New Conversation" button.',
          timestamp: new Date(),
        };
        addMessage(errorMessage);
        return;
      }
    }

    // Build user message content with file info
    let userContent = content;
    if (files && files.length > 0) {
      const fileInfo = files.map(f => `[${f.name}]`).join(' ');
      userContent = `${fileInfo}\n\n${content}`;
    }

    // Add user message (optimistic update)
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userContent,
      timestamp: new Date(),
    };
    addMessage(userMessage);
    setLoading(true);

    try {
      // Call API with settings, conversation_id, and files
      const response = await sendChatMessage({
        question: content,
        conversation_id: currentSession.id,  // Pass conversation ID to save to database
        use_premium_model: usePremiumModel,
        use_preprocessing: usePreprocessing,
        provider: usePremiumModel ? undefined : selectedProvider || undefined,
        model: usePremiumModel ? undefined : selectedModel || undefined,
      }, files);

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
