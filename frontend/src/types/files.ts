/**
 * File-related type definitions
 */

export type FileWithMetadata = {
  file: File;
  preview?: string;
  progress?: number;
  error?: string;
  compressed?: boolean;
};
