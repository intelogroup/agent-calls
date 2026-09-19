-- Item: Stripe event idempotency. Stripe redelivers events when we return
-- 5xx (and occasionally delivers the same event twice regardless), so every
-- event id is recorded before its handler runs. A redelivered id whose row
-- already says processed short-circuits to 200 without re-applying, and the
-- table doubles as the delivery audit trail (including poison events stuck
-- in failed). Business deletes cascade their event rows.
create table stripe_webhook_events (
  event_id text primary key,
  type text not null,
  business_id uuid references businesses (id) on delete cascade,
  status text not null default 'received',
  error text,
  received_at timestamptz not null default now(),
  processed_at timestamptz
);

alter table stripe_webhook_events enable row level security;

-- No policies: service-role client only. Authenticated users never read or
-- write webhook delivery records; RLS with zero policies denies them all.
