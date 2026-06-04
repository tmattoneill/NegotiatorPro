-- 007_partner_cloned_from.sql
--
-- Track the template a private partner copy was cloned from, so an edit can
-- optionally be pushed back to that template ("update parent"). NULL means the
-- partner has no parent (a template itself, or a copy created before this column
-- existed). ON DELETE SET NULL keeps copies if their template is later removed.

ALTER TABLE partner_personas
  ADD COLUMN IF NOT EXISTS cloned_from UUID REFERENCES partner_personas(id) ON DELETE SET NULL;
