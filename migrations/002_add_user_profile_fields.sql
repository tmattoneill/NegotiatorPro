-- Migration 002: Add User Profile Fields
-- Created: 2025-11-20
-- Purpose: Extend users table with profile information and API keys

-- ============================================================================
-- ALTER USERS TABLE
-- ============================================================================

-- Add profile fields
ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR(100);

-- Add encrypted API keys (optional fields)
-- These will be encrypted before storage
ALTER TABLE users ADD COLUMN IF NOT EXISTS openai_api_key TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS anthropic_api_key TEXT;

-- Add profile metadata
ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_updated_at TIMESTAMP;

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_users_first_name ON users(first_name);
CREATE INDEX IF NOT EXISTS idx_users_last_name ON users(last_name);

-- ============================================================================
-- UPDATE EXISTING DATA
-- ============================================================================

-- Set profile_updated_at for existing users
UPDATE users
SET profile_updated_at = created_at
WHERE profile_updated_at IS NULL;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Function to update profile_updated_at timestamp
CREATE OR REPLACE FUNCTION update_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    -- Only update profile_updated_at when profile fields change
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

-- Apply trigger to users table
DROP TRIGGER IF EXISTS update_users_profile_timestamp ON users;
CREATE TRIGGER update_users_profile_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_profile_timestamp();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN users.first_name IS 'User first name';
COMMENT ON COLUMN users.last_name IS 'User last name';
COMMENT ON COLUMN users.openai_api_key IS 'Encrypted OpenAI API key (optional)';
COMMENT ON COLUMN users.anthropic_api_key IS 'Encrypted Anthropic API key (optional)';
COMMENT ON COLUMN users.profile_updated_at IS 'Timestamp of last profile update';

-- End of migration 002
