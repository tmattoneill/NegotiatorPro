/**
 * File preview component with support for images and documents
 * Displays file thumbnails, metadata, and upload progress
 */
import { FiFile, FiImage, FiFileText, FiX, FiCheck } from 'react-icons/fi';
import type { FileWithMetadata } from '../types/files';

interface FilePreviewProps {
  files: FileWithMetadata[];
  onRemove: (index: number) => void;
  maxFileSize?: number; // in MB
}

export default function FilePreview({ files, onRemove, maxFileSize = 10 }: FilePreviewProps) {
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (fileType: string, fileName: string) => {
    if (fileType.startsWith('image/')) return <FiImage size={24} />;
    if (fileType === 'application/pdf') return <FiFileText size={24} />;
    if (fileType.startsWith('text/') || fileName.endsWith('.txt') || fileName.endsWith('.csv')) {
      return <FiFileText size={24} />;
    }
    if (fileName.match(/\.(doc|docx)$/i)) return <FiFileText size={24} />;
    if (fileName.match(/\.(xls|xlsx)$/i)) return <FiFileText size={24} />;
    return <FiFile size={24} />;
  };

  const getFileTypeLabel = (file: File): string => {
    if (file.type === 'application/pdf') return 'PDF';
    if (file.type.startsWith('image/')) return 'Image';
    if (file.name.match(/\.(doc|docx)$/i)) return 'Word';
    if (file.name.match(/\.(xls|xlsx)$/i)) return 'Excel';
    if (file.type.startsWith('text/') || file.name.endsWith('.txt')) return 'Text';
    if (file.name.endsWith('.csv')) return 'CSV';
    return 'File';
  };

  if (files.length === 0) return null;

  return (
    <div className="file-preview-container">
      {files.map((fileData, index) => {
        const { file, preview, progress, error, compressed } = fileData;
        const isImage = file.type.startsWith('image/');
        const fileSizeExceeded = file.size > maxFileSize * 1024 * 1024;

        return (
          <div key={index} className={`file-preview-item ${error || fileSizeExceeded ? 'error' : ''}`}>
            {/* Preview area */}
            <div className="file-preview-thumbnail">
              {isImage && preview ? (
                <img src={preview} alt={file.name} />
              ) : (
                <div className="file-icon-placeholder">
                  {getFileIcon(file.type, file.name)}
                </div>
              )}

              {/* Progress overlay */}
              {progress !== undefined && progress < 100 && (
                <div className="file-progress-overlay">
                  <div className="progress-circle">
                    <span>{progress}%</span>
                  </div>
                </div>
              )}

              {/* Completed overlay */}
              {progress === 100 && (
                <div className="file-complete-overlay">
                  <FiCheck size={24} color="#4caf50" />
                </div>
              )}
            </div>

            {/* File metadata */}
            <div className="file-metadata">
              <div className="file-type-badge">{getFileTypeLabel(file)}</div>
              <div className="file-name" title={file.name}>
                {file.name.length > 25 ? file.name.substring(0, 22) + '...' : file.name}
              </div>
              <div className="file-size">
                {formatFileSize(file.size)}
                {compressed && <span className="compressed-badge">Compressed</span>}
              </div>
              {fileSizeExceeded && (
                <div className="file-error">Exceeds {maxFileSize}MB limit</div>
              )}
              {error && <div className="file-error">{error}</div>}
            </div>

            {/* Remove button */}
            <button
              type="button"
              className="file-remove-button"
              onClick={() => onRemove(index)}
              aria-label={`Remove ${file.name}`}
            >
              <FiX size={18} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
