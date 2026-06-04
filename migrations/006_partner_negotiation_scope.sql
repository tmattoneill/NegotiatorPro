-- 006_partner_negotiation_scope.sql
--
-- Per-negotiation isolated partner personas (copy-on-write).
--
-- A partner persona is either a reusable library template (negotiation_id IS NULL)
-- or a private copy scoped to a single negotiation (negotiation_id set). The first
-- time a negotiation's partner is edited, the backend clones the template into a
-- private copy bound here, so edits never leak into the shared template or other
-- negotiations. ON DELETE CASCADE removes the private copy with its negotiation.

ALTER TABLE partner_personas
  ADD COLUMN IF NOT EXISTS negotiation_id UUID REFERENCES negotiations(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_partner_personas_negotiation_id ON partner_personas(negotiation_id);
