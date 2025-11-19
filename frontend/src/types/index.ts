/**
 * TypeScript type definitions for the application
 */

export interface ChatRequest {
  question: string;
  partner_info?: string;
  use_premium_model: boolean;
  use_preprocessing: boolean;
  provider?: string;
  model?: string;
}

export interface ChatResponse {
  answer: string;
  model_used: string;
  tokens_used?: number;
  processing_time?: number;
}

export interface LoginRequest {
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model_used?: string;
  processing_time?: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
}

export interface BackendInfo {
  name: string;
  enabled: boolean;
  models: ModelInfo[];
}

export interface ModelsResponse {
  [backendId: string]: BackendInfo;
}
