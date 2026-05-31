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
