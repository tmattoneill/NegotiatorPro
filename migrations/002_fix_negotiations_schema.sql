-- Migration 002: Fix negotiations table schema mismatch
-- Issue: Database has legacy 'name' column (NOT NULL) but code uses 'title'
-- This migration consolidates to use 'title' only

BEGIN;

-- Step 1: Copy any existing data from title to name (if title has data)
UPDATE negotiations
SET name = title
WHERE title IS NOT NULL AND title != '';

-- Step 2: For rows where title is null/empty, keep existing name
-- (No action needed, name already has data)

-- Step 3: Drop the old name column
ALTER TABLE negotiations DROP COLUMN IF EXISTS name;

-- Step 4: Make title NOT NULL (since it's now the primary identifier)
ALTER TABLE negotiations ALTER COLUMN title SET NOT NULL;

-- Step 5: Drop other legacy columns that aren't in current schema
ALTER TABLE negotiations DROP COLUMN IF EXISTS visibility;
ALTER TABLE negotiations DROP COLUMN IF EXISTS preferred_backend;
ALTER TABLE negotiations DROP COLUMN IF EXISTS preferred_model;

COMMIT;
