# agent-call-bus — async consultation protocol (voice-agent side)

Mirrors `BUS_PROTOCOL.md` in `intelogroup/agent-call-bus` (the authoritative
copy). The pre-existing `PROTOCOL.md` live JSONL design (`calls/live/…`) is
**not** used by this worker — the judgment-queue watcher (polling every
5 min) reads the async `requests/` layout below.

## Layout

```
requests/<uuid>.json    # voice agent -> Jett (one file per consult)
responses/<uuid>.json   # Jett -> voice-agent side (one file per answer)
deliveries/<uuid>.json  # delivery record (Jett's side, after delivering)
facts/facts.json        # synced facts snapshot (Track B; not part of consults)
```

`<uuid>` is a UUID v4. The `id` field inside MUST match the filename stem.

## requests/<uuid>.json — voice agent → Jett

Written by `consult_jett` in `server/live_agent.py`, committed + pushed
immediately (batch window ≤ 3 s).

```json
{
  "id": "3f9a2c1e-7b4d-4a8e-9c1f-2e5d6a7b8c9d",
  "ts": "2026-09-18T16:40:00Z",
  "call_id": "jett-20260918-163855-x7q2",
  "question": "Caller asks: did my National Grid payment go through?",
  "context": "Live voice call with Jim on Jett's line (sip:jett@129.159.189.244). Asked mid-call; the brief and facts snapshot couldn't answer it.",
  "priority": "normal"
}
```

- `id`, `ts`, `question`: REQUIRED. `ts` is UTC ISO-8601 with trailing `Z`.
- `call_id`, `context`, `priority` (`normal` | `urgent`): recommended.
- `priority: "urgent"` is for time-sensitive matters (money emergencies,
  same-day deadlines).

## responses/<uuid>.json — Jett → voice-agent side

Written by Jett's side (the real Jett runtime — never the watcher, never the
voice agent). The filename MUST equal the request's uuid it answers.

```json
{
  "id": "3f9a2c1e-7b4d-4a8e-9c1f-2e5d6a7b8c9d",
  "ts": "2026-09-18T16:45:00Z",
  "answer": "Yes — the $30.23 National Grid payment posted this morning.",
  "attribution": "Jett says",
  "answered_by": "jett-runtime"
}
```

- `id`, `ts`, `answer`: REQUIRED.
- `attribution` MUST be exactly `"Jett says"`.
- `answered_by` MUST be `"jett-runtime"`.
- `answer`: plain speakable text. Numbers, dates, amounts, statuses: relay
  EXACTLY — the worker validates them against the spoken relay and falls
  back to reading the answer literally on any mismatch.

## deliveries/<uuid>.json — delivery record

Written by Jett's side after the answer reaches the caller. The voice agent
NEVER writes `responses/` or `deliveries/`.

## Worker behavior (server/live_agent.py)

`consult_jett(question, priority)`:

1. Writes `requests/<uuid>.json`, commits, pushes immediately.
2. Speaks a holding line ("Let me check with Jett on that — one moment…").
3. Polls `responses/<uuid>.json` (git pull every 5 s) for up to
   `CONSULT_TIMEOUT_SEC` (default 180 s).
4. Answer lands → returned to the model, relayed live as "Jett says …"
   (attribution + number validation enforced in code).
5. Timeout → returns `FILED_ASYNC: …` — the request stays filed, Jett's
   5-minute watcher picks it up, and the answer is delivered afterward
   (voice callback / WhatsApp). The caller is told honestly the question
   was passed to Jett.
6. Bus unusable (e.g. deploy key not added yet) or push failed → returns
   `ERROR_CANT_REACH_JETT: …` — the caller is told honestly Jett can't be
   reached right now. Nothing is invented.

## Rules

1. Writer ownership: `requests/` ONLY the voice agent; `responses/` and
   `deliveries/` ONLY Jett's side. The watcher writes nothing.
2. Append-only: never edit a request/response/delivery after writing.
3. One JSON object per file, UTF-8.
4. Nothing secret ever lands here — no API keys, no credentials, no tokens.
   Keep caller PII to the minimum the question needs.
5. Answered + delivered pairs older than 7 days may be deleted by either side.
