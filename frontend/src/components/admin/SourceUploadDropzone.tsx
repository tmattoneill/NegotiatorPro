import { useRef, useState } from 'react';
import Button from '../ui/Button';

interface Props {
  onUpload: (files: File[]) => Promise<void>;
  uploading: boolean;
}

export default function SourceUploadDropzone({ onUpload, uploading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    await onUpload(Array.from(files));
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
        dragging
          ? 'border-chat-primary bg-chat-primary/10'
          : 'border-chat-border bg-chat-muted/40 hover:border-chat-primary/50'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
    >
      <p className="text-chat-muted-foreground text-sm mb-3">
        Drag &amp; drop PDF, TXT, or DOCX files here
      </p>
      <Button
        variant="outline"
        size="sm"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
      >
        {uploading ? 'Uploading…' : 'Browse files'}
      </Button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.docx"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
