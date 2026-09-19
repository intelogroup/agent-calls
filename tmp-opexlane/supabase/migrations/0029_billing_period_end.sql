-- 0028 (part 2): billing-period end anchor.
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS current_period_end timestamptz;
