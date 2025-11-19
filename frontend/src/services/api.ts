/**
 * API service layer for communicating with FastAPI backend
 */
import axios from 'axios';
import type { ChatRequest, ChatResponse, LoginRequest, LoginResponse, ModelsResponse } from '../types';

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
 */
export const sendChatMessage = async (request: ChatRequest): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/chat', request);
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

export default api;
