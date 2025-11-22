/**
 * Chat input component with file upload and image paste support
 */
import { useState, useRef } from 'react';

interface ChatInputProps {
  onSend: (message: string, files?: File[]) => void;
  isLoading: boolean;
}

interface AttachedFile {
  file: File;
  preview?: string;
}

export default function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [input, setInput] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((input.trim() || attachedFiles.length > 0) && !isLoading) {
      onSend(input, attachedFiles.map(af => af.file));
      setInput('');
      setAttachedFiles([]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    const imageItems = Array.from(items).filter(item => item.type.startsWith('image/'));

    if (imageItems.length > 0) {
      e.preventDefault();

      for (const item of imageItems) {
        const file = item.getAsFile();
        if (file) {
          addFile(file);
        }
      }
    }
  };

  const addFile = (file: File) => {
    // Create preview for images
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setAttachedFiles(prev => [...prev, { file, preview: e.target?.result as string }]);
      };
      reader.readAsDataURL(file);
    } else {
      setAttachedFiles(prev => [...prev, { file }]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      // Validate file types
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'text/plain', 'text/csv'];
      if (validTypes.includes(file.type) || file.name.endsWith('.txt') || file.name.endsWith('.csv')) {
        addFile(file);
      }
    });

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer?.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'text/plain', 'text/csv'];
      if (validTypes.includes(file.type) || file.name.endsWith('.txt') || file.name.endsWith('.csv')) {
        addFile(file);
      }
    });
  };

  return (
    <div className="input-container">
      <form onSubmit={handleSubmit}>
        {/* File attachments preview */}
        {attachedFiles.length > 0 && (
          <div className="attachments-preview">
            {attachedFiles.map((attached, index) => (
              <div key={index} className="attachment-item">
                {attached.preview ? (
                  <img src={attached.preview} alt={attached.file.name} className="attachment-image" />
                ) : (
                  <div className="attachment-file">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                      <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                    <span className="attachment-name">{attached.file.name}</span>
                  </div>
                )}
                <button
                  type="button"
                  className="remove-attachment"
                  onClick={() => removeFile(index)}
                  aria-label="Remove file"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        <div
          className={`input-wrapper ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <button
            type="button"
            className="attachment-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            title="Attach files (images, txt, csv)"
            aria-label="Attach files"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/png,image/jpeg,image/jpg,image/gif,image/webp,.txt,.csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={isDragging ? "Drop files here..." : "Ask a negotiation question... (Ctrl+V to paste images)"}
            disabled={isLoading}
            rows={2}
          />
          <button
            type="submit"
            className="send-button"
            disabled={isLoading || (!input.trim() && attachedFiles.length === 0)}
          >
            {isLoading ? (
              <>
                <span className="loading-spinner-small"></span>
                Sending...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
                Send
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
