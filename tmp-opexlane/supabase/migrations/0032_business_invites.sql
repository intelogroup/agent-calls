-- Item: staff invitation lifecycle. The old settings invite immediately
-- inserted the invited user into business_users — no pending state, no
-- expiry, no way to decline or revoke. Invites are now first-class rows:
-- pending -> accepted | declined | revoked, with lazy expiry (expired is
-- marked when the invite is next read after expires_at passes).
create table business_invites (
  id uuid primary key default gen_random_uuid(),
  business_id uuid not null references businesses (id) on delete cascade,
  email text not null,
  role text not null default 'staff',
  token text not null unique,
  status text not null default 'pending',
  expires_at timestamptz not null default now() + interval '7 days',
  created_by uuid references auth.users (id) on delete set null,
  created_at timestamptz not null default now(),
  decided_at timestamptz
);

create index business_invites_business_id_idx on business_invites (business_id);
create index business_invites_token_idx on business_invites (token);

alter table business_invites enable row level security;

-- Members of the business (and platform admins) can see its invites. The
-- invitee themself is not a member yet — the /invite/[token] page reads via
-- the service-role client using the unguessable token, same pattern as the
-- delete cascade RPCs (app-code check, never end-user session).
create policy "tenant read" on business_invites for select to authenticated
  using (exists (select 1 from business_users bu where bu.business_id = business_invites.business_id and bu.user_id = auth.uid())
      or exists (select 1 from platform_admins pa where pa.user_id = auth.uid()));
