-- Initial Database Schema for NegotiatorPro
-- PostgreSQL Migration 001
-- Created: 2024-11-15

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USER MANAGEMENT
-- ============================================================================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user', -- 'admin', 'user', 'viewer'
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT valid_role CHECK (role IN ('admin', 'user', 'viewer'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- ============================================================================
-- SESSION MANAGEMENT
-- ============================================================================

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45), -- IPv6 support
    user_agent TEXT,

    CONSTRAINT valid_session_role CHECK (role IN ('admin', 'user', 'viewer'))
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);

-- ============================================================================
-- SYSTEM CONFIGURATION
-- ============================================================================

-- System configuration table
CREATE TABLE system_config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX idx_system_config_updated_at ON system_config(updated_at);

-- ============================================================================
-- LLM CONFIGURATION
-- ============================================================================

-- LLM backend configuration
CREATE TABLE llm_config (
    id SERIAL PRIMARY KEY,
    backend VARCHAR(50) NOT NULL, -- 'openai', 'anthropic', 'ollama', 'ollama-cloud'
    model VARCHAR(100) NOT NULL,
    config_type VARCHAR(50) NOT NULL, -- 'default' or 'premium'
    parameters JSONB, -- Model-specific parameters
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_backend CHECK (backend IN ('openai', 'anthropic', 'ollama', 'ollama-cloud')),
    CONSTRAINT valid_config_type CHECK (config_type IN ('default', 'premium'))
);

CREATE INDEX idx_llm_config_backend ON llm_config(backend);
CREATE INDEX idx_llm_config_is_active ON llm_config(is_active);
CREATE INDEX idx_llm_config_type ON llm_config(config_type);

-- ============================================================================
-- USAGE STATISTICS
-- ============================================================================

-- Usage logs table
CREATE TABLE usage_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    model VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DECIMAL(10,6),
    response_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX idx_usage_logs_timestamp ON usage_logs(timestamp DESC);
CREATE INDEX idx_usage_logs_user_id ON usage_logs(user_id);
CREATE INDEX idx_usage_logs_model ON usage_logs(model);

-- Partition by month for performance (optional, for high-volume deployments)
-- CREATE TABLE usage_logs_YYYY_MM PARTITION OF usage_logs
--     FOR VALUES FROM ('YYYY-MM-01') TO ('YYYY-MM+1-01');

-- ============================================================================
-- DOCUMENT MANAGEMENT
-- ============================================================================

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL, -- 'pdf', 'docx', 'txt', 'doc'
    file_size INTEGER NOT NULL,
    file_hash VARCHAR(64) NOT NULL, -- SHA-256 for deduplication
    upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
    metadata JSONB, -- Additional metadata (author, pages, etc.)
    is_processed BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at TIMESTAMP,

    CONSTRAINT unique_file_hash UNIQUE (file_hash)
);

CREATE INDEX idx_documents_upload_date ON documents(upload_date DESC);
CREATE INDEX idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX idx_documents_file_type ON documents(file_type);
CREATE INDEX idx_documents_is_processed ON documents(is_processed);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);

-- ============================================================================
-- PROMPT MANAGEMENT
-- ============================================================================

-- Prompts table with versioning
CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    prompt_type VARCHAR(50) NOT NULL, -- 'system', 'user_default', etc.
    content TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    notes TEXT,

    CONSTRAINT unique_active_prompt UNIQUE (prompt_type, is_active) WHERE is_active = TRUE
);

CREATE INDEX idx_prompts_type ON prompts(prompt_type);
CREATE INDEX idx_prompts_is_active ON prompts(is_active);
CREATE INDEX idx_prompts_created_at ON prompts(created_at DESC);

-- ============================================================================
-- CHAT HISTORY
-- ============================================================================

-- Chat messages table
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    model VARCHAR(100), -- Model used for assistant responses
    tokens_used INTEGER,
    preprocessing_applied BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_message_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at DESC);

-- ============================================================================
-- EMBEDDING CONFIGURATION
-- ============================================================================

-- Embedding configuration table
CREATE TABLE embedding_config (
    id SERIAL PRIMARY KEY,
    model VARCHAR(100) NOT NULL,
    dimensions INTEGER NOT NULL,
    provider VARCHAR(50) NOT NULL, -- 'openai', 'anthropic', etc.
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_embedding_config_is_active ON embedding_config(is_active);

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- Insert default admin user (password: admin123)
-- Note: This is hashed with bcrypt, change password immediately after first login
INSERT INTO users (id, username, email, password_hash, role)
VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@negotiatorpro.local',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oDWP.Fl4hN3K', -- admin123
    'admin'
);

-- Insert default system configuration
INSERT INTO system_config (key, value) VALUES
    ('admin_password', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oDWP.Fl4hN3K'),
    ('session_timeout_hours', '24'),
    ('max_upload_size_mb', '50');

-- Insert default prompts
INSERT INTO prompts (prompt_type, content, version, created_by) VALUES
    ('system', 'You are a negotiation expert advisor...', 1, (SELECT id FROM users WHERE username = 'admin' LIMIT 1)),
    ('user_default', 'Based on the following context, please provide negotiation advice...', 1, (SELECT id FROM users WHERE username = 'admin' LIMIT 1));

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active sessions view
CREATE VIEW active_sessions AS
SELECT s.*, u.username, u.email
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.expires_at > CURRENT_TIMESTAMP;

-- Usage statistics summary view
CREATE VIEW usage_summary AS
SELECT
    DATE(timestamp) as date,
    model,
    COUNT(*) as requests,
    SUM(total_tokens) as total_tokens,
    SUM(cost) as total_cost,
    AVG(response_time_ms) as avg_response_time
FROM usage_logs
GROUP BY DATE(timestamp), model
ORDER BY date DESC, model;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions
    WHERE expires_at < CURRENT_TIMESTAMP;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update timestamp on record update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at trigger to relevant tables
CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_llm_config_updated_at
    BEFORE UPDATE ON llm_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_embedding_config_updated_at
    BEFORE UPDATE ON embedding_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE users IS 'User accounts for multi-user support';
COMMENT ON TABLE sessions IS 'Active user sessions with expiration tracking';
COMMENT ON TABLE system_config IS 'System-wide configuration key-value store';
COMMENT ON TABLE llm_config IS 'LLM backend and model configurations';
COMMENT ON TABLE usage_logs IS 'API usage statistics and token tracking';
COMMENT ON TABLE documents IS 'Uploaded source documents for RAG';
COMMENT ON TABLE prompts IS 'System and user prompt templates with versioning';
COMMENT ON TABLE chat_messages IS 'Chat conversation history';
COMMENT ON TABLE embedding_config IS 'Vector embedding model configuration';

-- End of migration 001
