-- Migration 005: Replace per-provider API key columns with a single JSONB store
-- provider_keys stores {provider_id: fernet_encrypted_key} — add any provider without future migrations

ALTER TABLE users ADD COLUMN IF NOT EXISTS provider_keys JSONB NOT NULL DEFAULT '{}';

-- Migrate existing encrypted keys from individual columns into the new JSONB store
UPDATE users SET provider_keys = jsonb_strip_nulls(jsonb_build_object(
    'openai',     openai_api_key,
    'anthropic',  anthropic_api_key
))
WHERE openai_api_key IS NOT NULL OR anthropic_api_key IS NOT NULL;

ALTER TABLE users DROP COLUMN IF EXISTS openai_api_key;
ALTER TABLE users DROP COLUMN IF EXISTS anthropic_api_key;

-- The profile-timestamp trigger (migration 001) referenced the now-dropped
-- per-provider key columns. Without this, ANY update to a users row (including
-- the last_login write on login) fails with "record new has no field
-- openai_api_key". Redefine it to track provider_keys instead.
CREATE OR REPLACE FUNCTION update_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    IF (NEW.first_name IS DISTINCT FROM OLD.first_name OR
        NEW.last_name IS DISTINCT FROM OLD.last_name OR
        NEW.provider_keys IS DISTINCT FROM OLD.provider_keys OR
        NEW.email IS DISTINCT FROM OLD.email) THEN
        NEW.profile_updated_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
