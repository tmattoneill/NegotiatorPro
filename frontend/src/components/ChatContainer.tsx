/**
 * Main chat container component - clean layout
 */
import { useEffect, useRef, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { useAuthStore } from '../store/authStore';
import { sendChatMessage } from '../services/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import type { Message } from '../types';

export default function ChatContainer() {
  const { getCurrentSession, addMessage, isLoading, setLoading, createNewSession, renameSession, createLocalSession } = useChatStore();
  const { selectedProvider, selectedModel, usePremiumModel, usePreprocessing, availableModels } = useSettingsStore();
  const { negotiations, currentNegotiationId } = useNegotiationStore();
  const { user } = useAuthStore();
  const isSuperAdmin = user?.is_super_admin === true;

  // Get display names for provider and model
  const providerDisplayName = selectedProvider && availableModels[selectedProvider]?.name
    ? availableModels[selectedProvider].name
    : selectedProvider || 'Unknown';
  const modelDisplayName = selectedModel || 'Unknown';
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentSession = getCurrentSession();
  const currentNegotiation = negotiations.find(n => n.id === currentNegotiationId);

  // State for editing conversation title
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editingTitle, setEditingTitle] = useState('');

  // Reset editing state when session changes
  useEffect(() => {
    setIsEditingTitle(false);
    setEditingTitle('');
  }, [currentSession?.id]);

  // Note: conversations are created lazily on first message (see handleSendMessage)
  // and explicitly via the "New Conversation" button — never on mount/refresh,
  // which would otherwise spawn an empty conversation on every page load.

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages]);

  const handleSendMessage = async (content: string, files?: File[]) => {
    if (isSuperAdmin) {
      // Testing mode: create a local in-memory session on first message
      if (!getCurrentSession()?.id) {
        createLocalSession();
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    } else {
      if (!currentSession?.id && currentNegotiation?.id && user?.id) {
        await createNewSession(currentNegotiation.id, user.id);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (!getCurrentSession()?.id) {
        console.error('No session available');
        return;
      }
    }

    const session = getCurrentSession();

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
      // Call API — super admin testing mode sends without a conversation_id
      const response = await sendChatMessage({
        question: content,
        conversation_id: session?.id,
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

  const startEditingTitle = () => {
    if (currentSession) {
      setEditingTitle(currentSession.title || 'New Conversation');
      setIsEditingTitle(true);
    }
  };

  const handleTitleSubmit = async () => {
    if (currentSession && editingTitle.trim() && user?.id) {
      try {
        await renameSession(currentSession.id, user.id, editingTitle);
        setIsEditingTitle(false);
        setEditingTitle('');
      } catch (error) {
        console.error('Failed to rename conversation:', error);
        alert('Failed to rename conversation. Please try again.');
      }
    } else {
      setIsEditingTitle(false);
      setEditingTitle('');
    }
  };

  const cancelTitleEdit = () => {
    setIsEditingTitle(false);
    setEditingTitle('');
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        {isEditingTitle ? (
          <input
            type="text"
            value={editingTitle}
            onChange={(e) => setEditingTitle(e.target.value.slice(0, 64))}
            onBlur={handleTitleSubmit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleTitleSubmit();
              } else if (e.key === 'Escape') {
                cancelTitleEdit();
              }
            }}
            autoFocus
            maxLength={64}
            style={{
              margin: 0,
              fontSize: '20px',
              fontWeight: '600',
              background: 'transparent',
              border: '1px solid #fafafa',
              borderRadius: '4px',
              padding: '2px 4px',
              color: '#191919',
              outline: 'none',
              fontFamily: 'inherit',
              lineHeight: 'normal',
            }}
          />
        ) : (
          <h2
            onClick={startEditingTitle}
            style={{ cursor: currentSession ? 'pointer' : 'default' }}
            title={currentSession ? 'Click to rename' : ''}
          >
            {currentSession?.title || 'Select or create a conversation'}
          </h2>
        )}
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
