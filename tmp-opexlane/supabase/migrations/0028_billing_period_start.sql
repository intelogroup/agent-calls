-- 0028: billing-period anchors so minutes_used can reset each cycle.
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS current_period_start timestamptz;
