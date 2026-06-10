-- Store the parsed PLEASE self-assessment on the assistant message so the stats
-- gutter rehydrates when a conversation is reloaded. JSONB holds the six 1-5
-- scores plus the computed total and weakest-letter list. Null on non-ANALYSIS
-- turns, which carry no PLEASE block.
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS please_score JSONB;
