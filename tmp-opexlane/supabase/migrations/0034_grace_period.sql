-- Item: billing grace period. A failed payment no longer suspends the tenant
-- immediately — it starts a grace window (default 3 days, see
-- BILLING_GRACE_PERIOD_DAYS) during which the tenant keeps running. When the
-- window expires with billing_status still past_due, the tenant counts as
-- suspended everywhere (see isEffectivelySuspended in lib/billing/lifecycle.ts).
-- A successful payment clears the deadline.
alter table businesses add column grace_period_ends_at timestamptz;
