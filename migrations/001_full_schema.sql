-- NegotiatorPro Full Database Schema
-- PostgreSQL Migration - Consolidated for Production
-- Created: 2025-11-21
--
-- This is a consolidated migration combining all schema requirements.
-- Run this on a fresh database for production deployment.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USER MANAGEMENT
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    openai_api_key TEXT,
    anthropic_api_key TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    profile_updated_at TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT valid_role CHECK (role IN ('admin', 'user', 'viewer'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_first_name ON users(first_name);
CREATE INDEX idx_users_last_name ON users(last_name);

-- ============================================================================
-- SESSION MANAGEMENT
-- ============================================================================

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,

    CONSTRAINT valid_session_role CHECK (role IN ('admin', 'user', 'viewer'))
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);

-- ============================================================================
-- SYSTEM CONFIGURATION
-- ============================================================================

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

CREATE TABLE llm_config (
    id SERIAL PRIMARY KEY,
    backend VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    config_type VARCHAR(50) NOT NULL,
    parameters JSONB,
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

-- ============================================================================
-- DOCUMENT MANAGEMENT
-- ============================================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
    metadata JSONB,
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

CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    prompt_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    notes TEXT
);

CREATE INDEX idx_prompts_type ON prompts(prompt_type);
CREATE INDEX idx_prompts_is_active ON prompts(is_active);
CREATE INDEX idx_prompts_created_at ON prompts(created_at DESC);

CREATE UNIQUE INDEX unique_active_prompt_per_type ON prompts(prompt_type) WHERE is_active = TRUE;

-- ============================================================================
-- EMBEDDING CONFIGURATION
-- ============================================================================

CREATE TABLE embedding_config (
    id SERIAL PRIMARY KEY,
    model VARCHAR(100) NOT NULL,
    dimensions INTEGER NOT NULL,
    provider VARCHAR(50) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_embedding_config_is_active ON embedding_config(is_active);

-- ============================================================================
-- USER PERSONAS - User's own negotiation identities
-- ============================================================================

CREATE TABLE user_personas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    role_title VARCHAR(255),
    organization VARCHAR(255),
    communication_style TEXT,
    negotiation_strengths TEXT,
    notes TEXT,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_personas_user_id ON user_personas(user_id);
CREATE INDEX idx_user_personas_is_default ON user_personas(is_default);
CREATE UNIQUE INDEX unique_default_user_persona ON user_personas(user_id) WHERE is_default = TRUE;

-- ============================================================================
-- PARTNER PERSONAS - Negotiation counterparts (shareable)
-- ============================================================================

CREATE TABLE partner_personas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    role_title VARCHAR(255),
    company VARCHAR(255),
    communication_style TEXT,
    known_interests TEXT,
    batna_estimate TEXT,
    relationship_notes TEXT,
    is_shared BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_partner_personas_created_by ON partner_personas(created_by);
CREATE INDEX idx_partner_personas_is_shared ON partner_personas(is_shared);
CREATE INDEX idx_partner_personas_name ON partner_personas(name);

-- ============================================================================
-- NEGOTIATIONS - Core negotiation entity
-- ============================================================================

CREATE TABLE negotiations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    user_persona_id UUID REFERENCES user_personas(id) ON DELETE SET NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_negotiation_status CHECK (status IN ('active', 'paused', 'closed', 'won', 'lost'))
);

CREATE INDEX idx_negotiations_user_id ON negotiations(user_id);
CREATE INDEX idx_negotiations_status ON negotiations(status);
CREATE INDEX idx_negotiations_user_persona_id ON negotiations(user_persona_id);
CREATE INDEX idx_negotiations_created_at ON negotiations(created_at DESC);

-- ============================================================================
-- NEGOTIATION PARTNERS - Join table (many-to-many)
-- ============================================================================

CREATE TABLE negotiation_partners (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    negotiation_id UUID NOT NULL REFERENCES negotiations(id) ON DELETE CASCADE,
    partner_persona_id UUID NOT NULL REFERENCES partner_personas(id) ON DELETE CASCADE,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_negotiation_partner UNIQUE (negotiation_id, partner_persona_id)
);

CREATE INDEX idx_negotiation_partners_negotiation_id ON negotiation_partners(negotiation_id);
CREATE INDEX idx_negotiation_partners_partner_persona_id ON negotiation_partners(partner_persona_id);
CREATE UNIQUE INDEX unique_primary_partner ON negotiation_partners(negotiation_id) WHERE is_primary = TRUE;

-- ============================================================================
-- CONVERSATIONS - Chat sessions within negotiations
-- ============================================================================

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    negotiation_id UUID NOT NULL REFERENCES negotiations(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_negotiation_id ON conversations(negotiation_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);

-- ============================================================================
-- CHAT MESSAGES
-- ============================================================================

CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    model VARCHAR(100),
    tokens_used INTEGER,
    preprocessing_applied BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_message_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at DESC);

-- ============================================================================
-- NEGOTIATION DOCUMENTS - Documents attached to negotiations
-- ============================================================================

CREATE TABLE negotiation_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    negotiation_id UUID NOT NULL REFERENCES negotiations(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,

    CONSTRAINT unique_negotiation_document UNIQUE (negotiation_id, document_id)
);

CREATE INDEX idx_negotiation_documents_negotiation_id ON negotiation_documents(negotiation_id);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;
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

-- Function to update profile timestamp
CREATE OR REPLACE FUNCTION update_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    IF (NEW.first_name IS DISTINCT FROM OLD.first_name OR
        NEW.last_name IS DISTINCT FROM OLD.last_name OR
        NEW.openai_api_key IS DISTINCT FROM OLD.openai_api_key OR
        NEW.anthropic_api_key IS DISTINCT FROM OLD.anthropic_api_key OR
        NEW.email IS DISTINCT FROM OLD.email) THEN
        NEW.profile_updated_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_llm_config_updated_at
    BEFORE UPDATE ON llm_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_embedding_config_updated_at
    BEFORE UPDATE ON embedding_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_profile_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_profile_timestamp();

CREATE TRIGGER update_user_personas_updated_at
    BEFORE UPDATE ON user_personas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_partner_personas_updated_at
    BEFORE UPDATE ON partner_personas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_negotiations_updated_at
    BEFORE UPDATE ON negotiations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE VIEW active_sessions AS
SELECT s.*, u.username, u.email
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.expires_at > CURRENT_TIMESTAMP;

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
-- SEED DATA
-- ============================================================================

-- Insert default admin user (password: admin123)
-- IMPORTANT: Change this password immediately after first login!
INSERT INTO users (id, username, email, password_hash, role)
VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@negotiatorpro.local',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oDWP.Fl4hN3K',
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
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE users IS 'User accounts with profile and API key storage';
COMMENT ON TABLE sessions IS 'Active user sessions with expiration tracking';
COMMENT ON TABLE system_config IS 'System-wide configuration key-value store';
COMMENT ON TABLE llm_config IS 'LLM backend and model configurations';
COMMENT ON TABLE usage_logs IS 'API usage statistics and token tracking';
COMMENT ON TABLE documents IS 'Uploaded source documents for RAG';
COMMENT ON TABLE prompts IS 'System and user prompt templates with versioning';
COMMENT ON TABLE embedding_config IS 'Vector embedding model configuration';
COMMENT ON TABLE user_personas IS 'User negotiation identities/roles';
COMMENT ON TABLE partner_personas IS 'Negotiation counterpart profiles (shareable across users)';
COMMENT ON TABLE negotiations IS 'Core negotiation tracking with status and settings';
COMMENT ON TABLE negotiation_partners IS 'Links negotiations to partner personas';
COMMENT ON TABLE conversations IS 'Chat sessions within a negotiation context';
COMMENT ON TABLE chat_messages IS 'Chat conversation history';
COMMENT ON TABLE negotiation_documents IS 'Documents attached to specific negotiations';

COMMENT ON COLUMN users.openai_api_key IS 'Encrypted OpenAI API key (optional)';
COMMENT ON COLUMN users.anthropic_api_key IS 'Encrypted Anthropic API key (optional)';

-- End of consolidated schema
