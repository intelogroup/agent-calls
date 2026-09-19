-- 0027: suspension flag. When true the voice webhook rejects inbound calls with
a billing message, independent of the user-controlled is_live toggle.
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS suspended boolean NOT NULL DEFAULT false;
