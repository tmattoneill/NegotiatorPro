/**
 * API service layer for communicating with FastAPI backend
 */
import axios from 'axios';
import type { ChatRequest, ChatResponse, LoginRequest, LoginResponse, ModelsResponse, AvailableProvidersResponse } from '../types';
import type {
  UserPersona, UserPersonaCreate, UserPersonaUpdate,
  PartnerPersona, PartnerPersonaCreate, PartnerPersonaUpdate
} from '../types/personas';
import type { Conversation, ConversationCreate, ConversationUpdate } from '../types/conversations';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    // Debug: log the full URL being requested
    console.log('Axios request:', config.method?.toUpperCase(), config.baseURL ?? '', config.url ?? '', '→', (config.baseURL ?? '') + (config.url ?? ''));
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      // Optionally redirect to login
    }
    return Promise.reject(error);
  }
);

/**
 * Send a chat message to the API
 * Supports file uploads (images, txt, csv)
 */
export const sendChatMessage = async (request: ChatRequest, files?: File[]): Promise<ChatResponse> => {
  // Always use FormData since backend now expects form fields
  const formData = new FormData();
  formData.append('question', request.question);

  if (request.conversation_id) {
    formData.append('conversation_id', request.conversation_id);
  }

  if (request.partner_info) {
    formData.append('partner_info', request.partner_info);
  }

  formData.append('use_premium_model', String(request.use_premium_model || false));
  formData.append('use_preprocessing', String(request.use_preprocessing !== false));

  if (request.provider) {
    formData.append('provider', request.provider);
  }

  if (request.model) {
    formData.append('model', request.model);
  }

  // Append files if provided
  if (files && files.length > 0) {
    files.forEach((file) => {
      formData.append('files', file);
    });
  }

  const response = await api.post<ChatResponse>('/chat', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Login as admin
 */
export const login = async (request: LoginRequest): Promise<LoginResponse> => {
  const response = await api.post<LoginResponse>('/auth/login', request);
  return response.data;
};

/**
 * Check API health
 */
export const checkHealth = async (): Promise<{ status: string }> => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Fetch available LLM models organized by backend
 */
export const fetchAvailableModels = async (): Promise<ModelsResponse> => {
  const response = await api.get<ModelsResponse>('/models');
  return response.data;
};

/**
 * Fetch available providers filtered by user's API keys
 * Only returns providers the user can actually use
 */
export const fetchAvailableProviders = async (userId?: string): Promise<AvailableProvidersResponse> => {
  const params = userId ? `?user_id=${userId}` : '';
  const response = await api.get<AvailableProvidersResponse>(`/models/available-for-user${params}`);
  return response.data;
};

// ============================================================================
// USER PERSONAS
// ============================================================================

export const createUserPersona = async (userId: string, data: UserPersonaCreate): Promise<UserPersona> => {
  const response = await api.post<UserPersona>(`/personas/user?user_id=${userId}`, data);
  return response.data;
};

export const getUserPersonas = async (userId: string): Promise<UserPersona[]> => {
  const response = await api.get<UserPersona[]>(`/personas/user?user_id=${userId}`);
  return response.data;
};

export const getUserPersona = async (userId: string, id: string): Promise<UserPersona> => {
  const response = await api.get<UserPersona>(`/personas/user/${id}?user_id=${userId}`);
  return response.data;
};

export const updateUserPersona = async (userId: string, id: string, data: UserPersonaUpdate): Promise<UserPersona> => {
  const response = await api.patch<UserPersona>(`/personas/user/${id}?user_id=${userId}`, data);
  return response.data;
};

export const deleteUserPersona = async (userId: string, id: string): Promise<void> => {
  await api.delete(`/personas/user/${id}?user_id=${userId}`);
};

// ============================================================================
// PARTNER PERSONAS
// ============================================================================

export const createPartnerPersona = async (userId: string, data: PartnerPersonaCreate): Promise<PartnerPersona> => {
  const response = await api.post<PartnerPersona>(`/personas/partner?user_id=${userId}`, data);
  return response.data;
};

export const getPartnerPersonas = async (userId: string, includeShared = true): Promise<PartnerPersona[]> => {
  const response = await api.get<PartnerPersona[]>(`/personas/partner?user_id=${userId}&include_shared=${includeShared}`);
  return response.data;
};

export const getPartnerPersona = async (userId: string, id: string): Promise<PartnerPersona> => {
  const response = await api.get<PartnerPersona>(`/personas/partner/${id}?user_id=${userId}`);
  return response.data;
};

export const updatePartnerPersona = async (userId: string, id: string, data: PartnerPersonaUpdate): Promise<PartnerPersona> => {
  const response = await api.patch<PartnerPersona>(`/personas/partner/${id}?user_id=${userId}`, data);
  return response.data;
};

export const deletePartnerPersona = async (userId: string, id: string): Promise<void> => {
  await api.delete(`/personas/partner/${id}?user_id=${userId}`);
};

// ============================================================================
// CONVERSATIONS
// ============================================================================

export const createConversation = async (data: ConversationCreate): Promise<Conversation> => {
  const { user_id, ...requestData } = data;
  const response = await api.post<Conversation>(`/conversations/?user_id=${user_id}`, requestData);
  return response.data;
};

export const getNegotiationConversations = async (negotiationId: string, userId: string): Promise<Conversation[]> => {
  const response = await api.get<Conversation[]>(`/conversations/negotiation/${negotiationId}?user_id=${userId}`);
  return response.data;
};

export const getConversation = async (id: string): Promise<Conversation> => {
  const response = await api.get<Conversation>(`/conversations/${id}`);
  return response.data;
};

export const updateConversation = async (id: string, userId: string, data: ConversationUpdate): Promise<Conversation> => {
  const response = await api.patch<Conversation>(`/conversations/${id}?user_id=${userId}`, data);
  return response.data;
};

export const deleteConversation = async (id: string, userId: string): Promise<void> => {
  await api.delete(`/conversations/${id}?user_id=${userId}`);
};

export const getConversationMessages = async (conversationId: string, userId: string, limit?: number): Promise<any[]> => {
  const params = new URLSearchParams({ user_id: userId });
  if (limit) {
    params.append('limit', limit.toString());
  }
  const response = await api.get(`/conversations/${conversationId}/messages?${params.toString()}`);
  return response.data;
};

// ============================================================================
// ADMIN ENDPOINTS (require admin role)
// ============================================================================

export interface AdminUser {
  id: string;
  username: string;
  email: string;
  role: string;
  created_at: string;
  last_login: string | null;
  is_active: boolean;
}

export interface AdminNegotiation {
  id: string;
  user_id: string;
  username: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface AdminUsageStats {
  user_id: string;
  username: string;
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  models_used: string[];
  last_activity: string | null;
}

export interface DatabaseResetResponse {
  success: boolean;
  users_deleted: number;
  negotiations_deleted: number;
  conversations_deleted: number;
  messages_deleted: number;
  message: string;
}

/**
 * Admin: List all users
 */
export const adminListUsers = async (): Promise<AdminUser[]> => {
  const response = await api.get<AdminUser[]>('/admin/users');
  return response.data;
};

/**
 * Admin: Delete a user
 */
export const adminDeleteUser = async (userId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/admin/users/${userId}`);
  return response.data;
};

/**
 * Admin: List all negotiations
 */
export const adminListNegotiations = async (): Promise<AdminNegotiation[]> => {
  const response = await api.get<AdminNegotiation[]>('/admin/negotiations');
  return response.data;
};

/**
 * Admin: List negotiations for a specific user
 */
export const adminListUserNegotiations = async (userId: string): Promise<AdminNegotiation[]> => {
  const response = await api.get<AdminNegotiation[]>(`/admin/negotiations/user/${userId}`);
  return response.data;
};

/**
 * Admin: Delete a negotiation
 */
export const adminDeleteNegotiation = async (negotiationId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/admin/negotiations/${negotiationId}`);
  return response.data;
};

/**
 * Admin: Get usage summary for all users
 */
export const adminGetUsageSummary = async (): Promise<AdminUsageStats[]> => {
  const response = await api.get<AdminUsageStats[]>('/admin/usage/summary');
  return response.data;
};

/**
 * Admin: Get usage stats for a specific user
 */
export const adminGetUserUsage = async (userId: string): Promise<AdminUsageStats> => {
  const response = await api.get<AdminUsageStats>(`/admin/usage/user/${userId}`);
  return response.data;
};

/**
 * Admin: Reset the entire database (except admin users)
 */
export const adminResetDatabase = async (): Promise<DatabaseResetResponse> => {
  const response = await api.post<DatabaseResetResponse>('/admin/database/reset');
  return response.data;
};

// ============================================================================
// ADMIN LLM CONFIGURATION
// ============================================================================

export interface LLMModelInfo {
  id: string;
  name: string;
  description: string;
}

export interface LLMBackendStatus {
  id: string;
  name: string;
  enabled: boolean;
  has_api_key: boolean;
  requires_api_key: boolean;
  models: LLMModelInfo[];
  is_available: boolean;
}

export interface ActiveModelConfig {
  backend: string;
  model: string;
}

export interface LLMConfigResponse {
  backends: LLMBackendStatus[];
  active_models: {
    default: ActiveModelConfig;
    premium: ActiveModelConfig;
  };
}

export interface SetModelRequest {
  model_type: 'default' | 'premium';
  backend: string;
  model: string;
}

/**
 * Admin: Get full LLM backend configuration
 */
export const adminGetLLMConfig = async (): Promise<LLMConfigResponse> => {
  const response = await api.get<LLMConfigResponse>('/admin/llm-config');
  return response.data;
};

/**
 * Admin: Set active model for default or premium tier
 */
export const adminSetModel = async (request: SetModelRequest): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/admin/llm-config/set-model', request);
  return response.data;
};

/**
 * Admin: Enable or disable an LLM backend
 */
export const adminEnableBackend = async (backend: string, enabled: boolean): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/admin/llm-config/enable-backend', { backend, enabled });
  return response.data;
};

// ---------------------------------------------------------------------------
// Admin RAG — sources and vectorstore
// ---------------------------------------------------------------------------

export interface AdminSource {
  id: string;
  filename: string;
  sha256: string;
  title: string | null;
  author: string | null;
  year: number | null;
  tags: string[];
  enabled: boolean;
  page_count: number | null;
  word_count: number | null;
  size_bytes: number;
  extension: string;
  added_at: string;
  last_indexed_at: string | null;
}

export interface VectorstoreStatus {
  embedding_model: string | null;
  chunks_count: number | null;
  last_build: string | null;
  size_bytes: number;
  dimensions: number | null;
  metadata_path: string;
  index_exists: boolean;
}

export interface RebuildJob {
  id: string;
  status: 'pending' | 'running' | 'done' | 'error';
  percent: number;
  current_file: string | null;
  errors: string[];
  params: Record<string, unknown>;
  started_at: string;
  finished_at: string | null;
}

export interface TestQueryResult {
  content: string;
  score: number;
  source_file: string;
  title: string | null;
  tags: string[];
  page: number | null;
}

export const adminListSources = async (): Promise<AdminSource[]> => {
  const response = await api.get<AdminSource[]>('/admin/sources');
  return response.data;
};

export const adminUploadSource = async (file: File): Promise<{ success: boolean; source: AdminSource }> => {
  const form = new FormData();
  form.append('file', file);
  const response = await api.post<{ success: boolean; source: AdminSource }>('/admin/sources/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const adminUpdateSource = async (
  filename: string,
  updates: { title?: string; author?: string; year?: number; tags?: string[]; enabled?: boolean }
): Promise<AdminSource> => {
  const response = await api.patch<AdminSource>(`/admin/sources/${encodeURIComponent(filename)}`, updates);
  return response.data;
};

export const adminDeleteSource = async (filename: string): Promise<{ success: boolean }> => {
  const response = await api.delete<{ success: boolean }>(`/admin/sources/${encodeURIComponent(filename)}`);
  return response.data;
};

export const adminGetVectorstoreStatus = async (): Promise<VectorstoreStatus> => {
  const response = await api.get<VectorstoreStatus>('/admin/vectorstore/status');
  return response.data;
};

export const adminTriggerRebuild = async (params: {
  chunk_size?: number;
  chunk_overlap?: number;
  embedding_model?: string;
  tags_filter?: string[];
}): Promise<{ job_id: string; status: string }> => {
  const response = await api.post<{ job_id: string; status: string }>('/admin/vectorstore/rebuild', params);
  return response.data;
};

export const adminGetRebuildJob = async (jobId: string): Promise<RebuildJob> => {
  const response = await api.get<RebuildJob>(`/admin/vectorstore/jobs/${jobId}`);
  return response.data;
};

export const adminTestQuery = async (params: {
  query: string;
  k?: number;
  target?: 'live' | 'staging';
  tags_filter?: string[];
}): Promise<TestQueryResult[]> => {
  const response = await api.post<TestQueryResult[]>('/admin/vectorstore/test-query', params);
  return response.data;
};

export default api;
