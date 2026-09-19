-- Item: tenant-deletion cleanup. The admin console's deleteBusiness deleted DB
-- rows but left private call recordings orphaned in the call-recordings
-- bucket — nobody tracking them, no way to find them again. This RPC does
-- the whole tenant wipe as one transaction (a partial delete defeats the
-- point, same reasoning as delete_contact_cascade in 0024) and returns the
-- recording_url paths so the caller (service-role client only) can remove
-- them from Storage — SQL can't touch the Storage API directly.
--
-- Tables that reference businesses WITHOUT on delete cascade must be
-- deleted explicitly first or the final businesses delete fails with an FK
-- violation: audit_logs, calls, bookings, contacts, students, cars,
-- instructors. audit_logs also FKs to calls.id, so it goes before calls.
-- Everything else (business_users, services, service_staff, business_faqs,
-- business_ai_config, business_hours, booking_events via bookings,
-- call_antispoof_scores via calls) cascades automatically.
create or replace function delete_business_cascade(p_business_id uuid)
returns text[]
language plpgsql
security definer
set search_path = public
as $$
declare
  v_recording_paths text[];
begin
  if not exists (select 1 from businesses where id = p_business_id) then
    raise exception 'business not found';
  end if;

  select coalesce(array_agg(recording_url), '{}')
    into v_recording_paths
    from calls
    where business_id = p_business_id and recording_url is not null;

  delete from audit_logs
    where business_id = p_business_id
       or call_id in (select id from calls where business_id = p_business_id);
  delete from call_antispoof_scores where business_id = p_business_id;
  delete from calls where business_id = p_business_id;
  delete from bookings where business_id = p_business_id; -- cascades booking_events
  delete from contacts where business_id = p_business_id;
  delete from students where business_id = p_business_id;
  delete from cars where business_id = p_business_id;
  delete from instructors where business_id = p_business_id;
  delete from business_invites where business_id = p_business_id;
  delete from businesses where id = p_business_id; -- cascades memberships, services, staff, faqs, config, hours

  return v_recording_paths;
end;
$$;

-- Same rule as delete_contact_cascade: only callable via the service-role
-- client with an app-code ownership check, never by an end-user session.
revoke execute on function delete_business_cascade(uuid) from public;
