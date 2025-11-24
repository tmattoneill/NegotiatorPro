/**
 * Enhanced chat input component with multi-file upload, compression, and PDF support
 */
import { useState, useRef } from 'react';
import imageCompression from 'browser-image-compression';
import FilePreview from './FilePreview';
import type { FileWithMetadata } from '../types/files';

interface ChatInputProps {
  onSend: (message: string, files?: File[]) => void;
  isLoading: boolean;
}

const MAX_FILE_SIZE_MB = 10;
const COMPRESSION_OPTIONS = {
  maxSizeMB: 1,
  maxWidthOrHeight: 1920,
  useWebWorker: true,
};

export default function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [input, setInput] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<FileWithMetadata[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((input.trim() || attachedFiles.length > 0) && !isLoading) {
      const files = attachedFiles
        .filter(af => !af.error)
        .map(af => af.file);

      onSend(input, files);
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
          await addFile(file);
        }
      }
    }
  };

  const compressImage = async (file: File): Promise<{ file: File; compressed: boolean }> => {
    if (!file.type.startsWith('image/')) {
      return { file, compressed: false };
    }

    try {
      const compressedFile = await imageCompression(file, COMPRESSION_OPTIONS);
      const compressionRatio = (compressedFile.size / file.size) * 100;

      // Only use compressed version if it's significantly smaller (< 80% of original)
      if (compressionRatio < 80) {
        return { file: compressedFile, compressed: true };
      }
      return { file, compressed: false };
    } catch (error) {
      console.error('Error compressing image:', error);
      return { file, compressed: false };
    }
  };

  const addFile = async (file: File) => {
    // Validate file size
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setAttachedFiles(prev => [...prev, {
        file,
        error: `File exceeds ${MAX_FILE_SIZE_MB}MB limit`,
      }]);
      return;
    }

    // Create temporary entry with progress
    const tempIndex = attachedFiles.length;
    setAttachedFiles(prev => [...prev, { file, progress: 0 }]);

    // Compress image if applicable
    const { file: processedFile, compressed } = await compressImage(file);

    // Update progress
    setAttachedFiles(prev => prev.map((af, i) =>
      i === tempIndex ? { ...af, progress: 50 } : af
    ));

    // Create preview for images
    if (processedFile.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setAttachedFiles(prev => prev.map((af, i) =>
          i === tempIndex
            ? { file: processedFile, preview: e.target?.result as string, progress: 100, compressed }
            : af
        ));
      };
      reader.readAsDataURL(processedFile);
    } else {
      setAttachedFiles(prev => prev.map((af, i) =>
        i === tempIndex
          ? { file: processedFile, progress: 100, compressed }
          : af
      ));
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const validTypes = [
      'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp',
      'application/pdf',
      'text/plain', 'text/csv',
      'application/msword', // .doc
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
      'application/vnd.ms-excel', // .xls
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
    ];

    for (const file of Array.from(files)) {
      const isValidType = validTypes.includes(file.type) ||
        file.name.match(/\.(txt|csv|doc|docx|xls|xlsx|pdf)$/i);

      if (isValidType) {
        await addFile(file);
      } else {
        setAttachedFiles(prev => [...prev, {
          file,
          error: 'Unsupported file type',
        }]);
      }
    }

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

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer?.files;
    if (!files) return;

    const validTypes = [
      'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp',
      'application/pdf',
      'text/plain', 'text/csv',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ];

    for (const file of Array.from(files)) {
      const isValidType = validTypes.includes(file.type) ||
        file.name.match(/\.(txt|csv|doc|docx|xls|xlsx|pdf)$/i);

      if (isValidType) {
        await addFile(file);
      }
    }
  };

  return (
    <div className="input-container">
      <form onSubmit={handleSubmit}>
        {/* Enhanced file preview with PDF support */}
        <FilePreview
          files={attachedFiles}
          onRemove={removeFile}
          maxFileSize={MAX_FILE_SIZE_MB}
        />

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
            title="Attach files (images, PDFs, documents)"
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
            accept="image/*,.pdf,.txt,.csv,.doc,.docx,.xls,.xlsx"
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
            className=""
          />
          <button
            type="submit"
            className="send-button"
            disabled={isLoading || (!input.trim() && attachedFiles.filter(af => !af.error).length === 0)}
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
