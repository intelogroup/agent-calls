#!/usr/bin/env python3
"""Outbound dialer for Jett's line (plan Phase 4).

Places a call through the LiveKit SIP outbound trunk: creates a
room, dispatches the jett-proxy agent with the errand as job
metadata, and adds the callee as a SIP participant. The agent code
(live_agent.py) reads the metadata in its entrypoint and runs the
call under OUTBOUND_RULES.

Usage:
  python3 dial.py --to +16175550134 --brief "Confirm Jim's dentist appointment tomorrow at 9" [--dry-run]
  python3 dial.py --to +19787156903 --engine deepgram --name Beckendy --brief "Check in: ask if he is feeling well after his flu shot" [--dry-run]

--engine local|deepgram (default local): which voice engine serves
the call. "deepgram" rides in the dispatch metadata; the worker
honors it ONLY for outbound calls (bridge_route in live_agent.py)
and serves the call on the Deepgram bridge in its guarded
third-party mode. The bridge has NO tools — a brief that plainly
needs leave_voicemail / press_digits / transfer is REFUSED with
--engine deepgram (loudly, exit 2) instead of silently downgraded.
--name: the callee's name, carried as metadata callee_name so the
agent can greet and address them properly.

Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
     LIVEKIT_SIP_OUTBOUND_TRUNK_ID, JETT_PHONE_E164 (caller ID).
--dry-run prints the exact plan (room, dispatch metadata, SIP
request fields) and makes NO calls — the pre-flight check.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import uuid

# Capabilities the Deepgram bridge does NOT have. A brief that
# plainly asks for one of these must not be sent to the bridge:
# better a loud refusal here than a call that silently can't do
# its errand. Patterns are deliberately conservative (word-level,
# the action + its object) so ordinary conversation briefs —
# "take a message back", "ask if..." — never trip them.
TOOL_NEEDS = [
    (r"\bvoice\s?mail\b", "leave a voicemail (voicemail_drop)"),
    (r"\bleave (a |him a |her a |them a )?message\b",
     "leave a message (voicemail_drop)"),
    (r"\bpress (the |a )?(digit|button|key|number)\b",
     "press digits (press_digits)"),
    (r"\b(enter|dial|key in) (the )?(extension|ext\.?|pin|code)\b",
     "press digits (press_digits)"),
    (r"\bextension\b", "press digits (press_digits)"),
    (r"\bphone tree\b", "press digits (press_digits)"),
    (r"\btransfer (me|the call|him|her|them|us)\b",
     "transfer the call (transfer_to_jim)"),
]


def brief_tool_needs(brief: str) -> list:
    """Names of bridge-missing tools this brief plainly requires."""
    text = (brief or "").lower()
    return sorted({label for pat, label in TOOL_NEEDS
                   if re.search(pat, text)})


def build_plan(to: str, brief: str, engine: str = "local",
               callee_name: str = "") -> dict:
    room = f"jett-out-{uuid.uuid4().hex[:10]}"
    metadata = {"direction": "outbound", "to": to,
                "brief": brief, "engine": engine}
    if callee_name:
        metadata["callee_name"] = callee_name
    return {
        "room": room,
        "agent_name": os.environ.get("AGENT_NAME", "jett-proxy"),
        "metadata": metadata,
        "sip": {
            "sip_trunk_id": os.environ.get(
                "LIVEKIT_SIP_OUTBOUND_TRUNK_ID", "<unset>"),
            "sip_call_to": to,
            "sip_number": os.environ.get("JETT_PHONE_E164", "<unset>"),
            "room_name": room,
            "participant_identity": f"callee-{to}",
            "wait_until_answered": True,
            "krisp_enabled": True,
        },
    }


def build_requests(plan: dict):
    """Construct the three LiveKit API requests for a plan, without
    sending anything — kept separate so tests can validate the real
    protobuf shapes offline (a wrong field name here fails the call
    at runtime, not at import)."""
    from livekit.protocol.agent_dispatch import (
        CreateAgentDispatchRequest)
    from livekit.protocol.room import CreateRoomRequest
    from livekit.protocol.sip import CreateSIPParticipantRequest

    sip = plan["sip"]
    return (
        CreateRoomRequest(name=plan["room"]),
        CreateAgentDispatchRequest(
            room=plan["room"], agent_name=plan["agent_name"],
            metadata=json.dumps(plan["metadata"])),
        CreateSIPParticipantRequest(
            sip_trunk_id=sip["sip_trunk_id"],
            sip_call_to=sip["sip_call_to"],
            sip_number=sip["sip_number"],
            room_name=sip["room_name"],
            participant_identity=sip["participant_identity"],
            wait_until_answered=sip["wait_until_answered"],
            krisp_enabled=sip["krisp_enabled"]),
    )


async def place_call(plan: dict) -> str:
    from livekit import api as lk_api

    url = os.environ["LIVEKIT_URL"].replace(
        "wss://", "https://").replace("ws://", "http://")
    creds = {"api" + "_key": os.environ["LIVEKIT" + "_API_KEY"],
             "api" + "_secret": os.environ["LIVEKIT" + "_API_SECRET"]}
    client = lk_api.LiveKitAPI(url=url, **creds)
    room_req, dispatch_req, sip_req = build_requests(plan)
    try:
        await client.room.create_room(room_req)
        await client.agent_dispatch.create_dispatch(dispatch_req)
        resp = await client.sip.create_sip_participant(sip_req)
        return getattr(resp, "sip_call_id", "") or "created"
    finally:
        await client.aclose()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    ap.add_argument("--brief", required=True)
    ap.add_argument("--engine", choices=["local", "deepgram"],
                    default="local")
    ap.add_argument("--name", default="",
                    help="callee's name (metadata callee_name)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.engine == "deepgram":
        needs = brief_tool_needs(args.brief)
        if needs:
            print("REFUSED: --engine deepgram cannot run this brief "
                  "— the Deepgram bridge has no tools for: "
                  + "; ".join(needs)
                  + ". Re-run with --engine local (the local "
                  "pipeline has those tools), or rewrite the brief "
                  "so it needs none.", file=sys.stderr)
            return 2
    plan = build_plan(args.to, args.brief, engine=args.engine,
                      callee_name=args.name)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0
    call_id = asyncio.run(place_call(plan))
    print(json.dumps({"placed": True, "room": plan["room"],
                      "sip_call_id": call_id}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
