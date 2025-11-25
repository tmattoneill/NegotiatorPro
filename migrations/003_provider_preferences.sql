-- Provider Preferences Migration
-- Adds provider/model preferences at user, negotiation, and persona levels
-- Created: 2025-11-25

-- ============================================================================
-- USER PROVIDER PREFERENCES
-- Default provider/model settings for each user
-- ============================================================================

ALTER TABLE users
ADD COLUMN IF NOT EXISTS preferred_provider VARCHAR(50),
ADD COLUMN IF NOT EXISTS preferred_model VARCHAR(100);

COMMENT ON COLUMN users.preferred_provider IS 'User default LLM provider (openai, anthropic, ollama, runpod)';
COMMENT ON COLUMN users.preferred_model IS 'User default model ID within the preferred provider';

-- ============================================================================
-- NEGOTIATION PROVIDER SETTINGS
-- Override provider/model at negotiation level (stored in settings JSONB)
-- The settings column already exists, we just document the expected structure:
-- settings: { provider: string, model: string, ... }
-- ============================================================================

COMMENT ON COLUMN negotiations.settings IS 'Negotiation settings JSON including: provider, model, and other preferences';

-- ============================================================================
-- USER PERSONA PROVIDER PREFERENCES
-- ============================================================================

ALTER TABLE user_personas
ADD COLUMN IF NOT EXISTS preferred_provider VARCHAR(50),
ADD COLUMN IF NOT EXISTS preferred_model VARCHAR(100);

COMMENT ON COLUMN user_personas.preferred_provider IS 'Persona-level LLM provider preference';
COMMENT ON COLUMN user_personas.preferred_model IS 'Persona-level model ID preference';

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_users_preferred_provider ON users(preferred_provider);
CREATE INDEX IF NOT EXISTS idx_user_personas_preferred_provider ON user_personas(preferred_provider);

-- End of migration
