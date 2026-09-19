-- Voiceprint enrollment marker. contacts.voice_embedding (vector(256), from
-- 0001) holds the enrolled embedding; presence of the embedding is what
-- makes a contact enrolled. This cheap scalar lets list views show
-- enrollment status (and when it happened) without selecting the 256-dim
-- vector for every contact.
alter table contacts add column voice_enrolled_at timestamptz;
