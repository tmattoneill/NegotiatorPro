-- Negotiations and Personas Schema
-- PostgreSQL Migration 002
-- Created: 2024-11-21

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

-- Only one default persona per user
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

-- Only one primary partner per negotiation
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
-- UPDATE CHAT_MESSAGES - Add conversation reference
-- ============================================================================

ALTER TABLE chat_messages
ADD COLUMN conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE;

CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE TRIGGER update_user_personas_updated_at
    BEFORE UPDATE ON user_personas
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_partner_personas_updated_at
    BEFORE UPDATE ON partner_personas
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_negotiations_updated_at
    BEFORE UPDATE ON negotiations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE user_personas IS 'User negotiation identities/roles';
COMMENT ON TABLE partner_personas IS 'Negotiation counterpart profiles (shareable across users)';
COMMENT ON TABLE negotiations IS 'Core negotiation tracking with status and settings';
COMMENT ON TABLE negotiation_partners IS 'Links negotiations to partner personas (min 1 required)';
COMMENT ON TABLE conversations IS 'Chat sessions within a negotiation context';
COMMENT ON TABLE negotiation_documents IS 'Documents attached to specific negotiations';

-- End of migration 002
