-- Store the RAG citations surfaced for an assistant turn so the stats gutter's
-- "sources in play" rehydrates when a conversation is reloaded. JSONB holds an
-- ordered list of {title, sub, quote, active}. Null on turns with no retrieval.
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS sources JSONB;
