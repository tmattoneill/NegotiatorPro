-- Migration 002: Fix negotiations table schema mismatch
-- Issue: Database has legacy 'name' column (NOT NULL) but code uses 'title'
-- This migration consolidates to use 'title' only
--
-- 001_full_schema.sql was later consolidated to create the table without the
-- legacy columns, so on a fresh database there is no 'name' to copy from. The
-- copy step only runs when the column exists; every other step is already
-- safe to run against either shape.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'negotiations' AND column_name = 'name'
    ) THEN
        -- Keep title as the surviving identifier; backfill it from name where empty.
        UPDATE negotiations SET title = name WHERE (title IS NULL OR title = '') AND name IS NOT NULL;
    END IF;
END
$$;

ALTER TABLE negotiations DROP COLUMN IF EXISTS name;

ALTER TABLE negotiations ALTER COLUMN title SET NOT NULL;

ALTER TABLE negotiations DROP COLUMN IF EXISTS visibility;
ALTER TABLE negotiations DROP COLUMN IF EXISTS preferred_backend;
ALTER TABLE negotiations DROP COLUMN IF EXISTS preferred_model;
