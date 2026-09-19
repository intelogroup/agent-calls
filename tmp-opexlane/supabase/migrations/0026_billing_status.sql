-- 0026: track Stripe subscription state on the tenant ("none" | "trialing" | "active" | "past_due" | "canceled").
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS billing_status text NOT NULL DEFAULT 'none';
