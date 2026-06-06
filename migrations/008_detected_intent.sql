-- Add detected_intent column to chat_messages for persisting intent classification
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS detected_intent VARCHAR(20);
