-- Cache the inferred negotiation context (leverage, parties, vitals) on the
-- negotiation so the stats gutter loads instantly and the structured LLM call
-- only reruns when new messages have landed. JSONB holds the computed context
-- plus a small _meta block (message_count) for cache invalidation.
ALTER TABLE negotiations ADD COLUMN IF NOT EXISTS context JSONB;
