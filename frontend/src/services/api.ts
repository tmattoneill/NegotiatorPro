/**
 * API service layer for communicating with FastAPI backend
 */
import axios from 'axios';
import type { ChatRequest, ChatResponse, LoginRequest, LoginResponse, ModelsResponse } from '../types';
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
    console.log('Axios request:', config.method?.toUpperCase(), config.baseURL, config.url, '→', config.baseURL + config.url);
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

export const updateConversation = async (id: string, data: ConversationUpdate): Promise<Conversation> => {
  const response = await api.patch<Conversation>(`/conversations/${id}`, data);
  return response.data;
};

export const deleteConversation = async (id: string): Promise<void> => {
  await api.delete(`/conversations/${id}`);
};

export const getConversationMessages = async (conversationId: string, userId: string, limit?: number): Promise<any[]> => {
  const params = new URLSearchParams({ user_id: userId });
  if (limit) {
    params.append('limit', limit.toString());
  }
  const response = await api.get(`/conversations/${conversationId}/messages?${params.toString()}`);
  return response.data;
};

export default api;
