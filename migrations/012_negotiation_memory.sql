-- Compact per-negotiation memory: state of play, offers on the table, agreed
-- facts, open questions, user instructions and next steps, distilled from
-- turns that have fallen out of the verbatim history window the chat sends
-- to the model. Refreshed in the background after a reply is saved, never on
-- every turn. Shape:
--   {"summary_md": str, "source_message_count": int, "updated_at": iso8601, "model": str}
ALTER TABLE negotiations ADD COLUMN IF NOT EXISTS memory JSONB;
