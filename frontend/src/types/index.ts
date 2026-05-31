/**
 * TypeScript type definitions for the application
 */

export interface ChatRequest {
  question: string;
  conversation_id?: string;
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
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: {
    id: string;
    username: string;
    email: string;
    role: string;
    first_name?: string | null;
    last_name?: string | null;
    preferred_provider?: string | null;
    preferred_model?: string | null;
  };
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
  error?: string;
}

export interface ModelsResponse {
  [backendId: string]: BackendInfo;
}

// Provider availability response from /api/models/available-for-user
export interface ProviderInfo {
  name: string;
  available: boolean;
  models: ModelInfo[];
  is_fallback?: boolean;
  requires_key?: boolean;
  error?: string;
}

export interface AvailableProvidersResponse {
  providers: {
    [providerId: string]: ProviderInfo;
  };
  user_api_keys: {
    has_openai: boolean;
    has_anthropic: boolean;
    has_deepseek: boolean;
  };
  system_api_keys: {
    has_openai: boolean;
    has_anthropic: boolean;
    has_deepseek: boolean;
    has_runpod: boolean;
  };
  ollama_local_available: boolean;
}

// Provider selection preference
export interface ProviderPreference {
  provider: string | null;
  model: string | null;
}
