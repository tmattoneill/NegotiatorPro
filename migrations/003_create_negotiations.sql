-- Migration 003: Create negotiations table
-- This table stores negotiation instances for each user

CREATE TABLE IF NOT EXISTS negotiations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    visibility VARCHAR(20) NOT NULL DEFAULT 'private',

    -- Model preferences (override user defaults)
    preferred_backend VARCHAR(50),  -- e.g., 'openai', 'anthropic', 'ollama'
    preferred_model VARCHAR(100),   -- e.g., 'gpt-4o', 'claude-sonnet-4-5-20250929'

    -- Negotiation context
    background TEXT,                 -- Background information and key details
    key_players JSONB,               -- Array of {name, role, notes}

    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'active',  -- 'active', 'archived', 'completed'

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_visibility CHECK (visibility IN ('private', 'company')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'archived', 'completed'))
);

-- Create indexes for efficient queries
CREATE INDEX idx_negotiations_user_id ON negotiations(user_id);
CREATE INDEX idx_negotiations_status ON negotiations(status);
CREATE INDEX idx_negotiations_created_at ON negotiations(created_at DESC);
CREATE INDEX idx_negotiations_updated_at ON negotiations(updated_at DESC);

-- Create a trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_negotiations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_negotiations_updated_at
    BEFORE UPDATE ON negotiations
    FOR EACH ROW
    EXECUTE FUNCTION update_negotiations_updated_at();

-- Create a default "General" negotiation for each existing user
-- This ensures backwards compatibility with existing chat messages
INSERT INTO negotiations (user_id, name, description, status)
SELECT
    id,
    'General Negotiations',
    'Default negotiation for general conversations',
    'active'
FROM users
WHERE NOT EXISTS (
    SELECT 1 FROM negotiations WHERE negotiations.user_id = users.id
);

-- Add comments for documentation
COMMENT ON TABLE negotiations IS 'Stores negotiation instances for each user with context and preferences';
COMMENT ON COLUMN negotiations.key_players IS 'JSONB array of objects with shape: [{name: string, role: string, notes: string}]';
COMMENT ON COLUMN negotiations.visibility IS 'private = only user can see, company = team members can see (Phase 2)';
COMMENT ON COLUMN negotiations.preferred_backend IS 'Overrides user default model backend for this negotiation';
COMMENT ON COLUMN negotiations.preferred_model IS 'Overrides user default model for this negotiation';
