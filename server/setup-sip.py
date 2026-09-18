#!/usr/bin/env python3
"""Idempotent LiveKit SIP setup for Jett's voice proxy.

Creates (or replaces) exactly:
  1. one inbound trunk named "jett"  (direct dial sip:jett@<vm-ip>)
  2. one dispatch rule named "jett"  (individual, room prefix jett-,
     dispatches agent "jett-proxy")

Safe to re-run: existing trunk/rule with the same name are replaced, never
duplicated.

Reads credentials from /opt/agent/livekit.env (written by setup.sh).
"""

import asyncio
import os
import sys

sys.path.insert(0, "/opt/agent/venv/lib/python3.12/site-packages")

from livekit.api import LiveKitAPI  # noqa: E402
from livekit.protocol import sip as S  # noqa: E402
from livekit.protocol import room as R  # noqa: E402
from livekit.protocol.agent_dispatch import RoomAgentDispatch  # noqa: E402

LIVEKIT_ENV = "/opt/agent/livekit.env"
TRUNK_NAME = "jett"
RULE_NAME = "jett"
ROOM_PREFIX = "jett-"
AGENT_NAME = "jett-proxy"


def load_env(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


async def ensure_trunk(sip) -> str:
    """Return the trunk id, creating or replacing as needed."""
    existing = await sip.list_inbound_trunk(S.ListSIPInboundTrunkRequest())
    old = [t for t in existing.items if t.name == TRUNK_NAME]

    desired = S.SIPInboundTrunkInfo(
        name=TRUNK_NAME,
        numbers=["jett"],  # matches To: user when dialing sip:jett@<vm-ip>
    )
    if old:
        trunk_id = old[0].sip_trunk_id
        # preserve server-assigned fields, replace our config
        desired.sip_trunk_id = trunk_id
        info = await sip.update_inbound_trunk(trunk_id, desired)
        print(f"trunk '{TRUNK_NAME}' updated ({trunk_id})", flush=True)
        # delete any accidental duplicates
        for dup in old[1:]:
            await sip.delete_sip_trunk(
                S.DeleteSIPTrunkRequest(sip_trunk_id=dup.sip_trunk_id))
            print(f"duplicate trunk {dup.sip_trunk_id} deleted", flush=True)
        return info.sip_trunk_id

    info = await sip.create_inbound_trunk(
        S.CreateSIPInboundTrunkRequest(trunk=desired))
    print(f"trunk '{TRUNK_NAME}' created ({info.sip_trunk_id})", flush=True)
    return info.sip_trunk_id


async def ensure_rule(sip, trunk_id: str):
    existing = await sip.list_dispatch_rule(S.ListSIPDispatchRuleRequest())
    for rule in existing.items:
        if rule.name == RULE_NAME:
            await sip.delete_dispatch_rule(
                S.DeleteSIPDispatchRuleRequest(
                    sip_dispatch_rule_id=rule.sip_dispatch_rule_id))
            print(f"old rule '{RULE_NAME}' deleted", flush=True)

    rule = await sip.create_dispatch_rule(
        S.CreateSIPDispatchRuleRequest(
            dispatch_rule=S.SIPDispatchRule(
                dispatch_rule_individual=S.SIPDispatchRuleIndividual(
                    room_prefix=ROOM_PREFIX)),
            trunk_ids=[trunk_id],
            name=RULE_NAME,
            room_config=R.RoomConfiguration(
                empty_timeout=300,
                max_participants=10,
                agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
            ),
        ))
    print(f"rule '{RULE_NAME}' created -> room prefix '{ROOM_PREFIX}', "
          f"agent '{AGENT_NAME}' ({rule.sip_dispatch_rule_id})", flush=True)


async def main() -> int:
    env = load_env(LIVEKIT_ENV)
    url = env.get("LIVEKIT_URL", "ws://localhost:7880")
    key = env.get("LIVEKIT_API_KEY", "")
    secret = env.get("LIVEKIT_API_SECRET", "")
    if not (key and secret):
        print("LIVEKIT_API_KEY/SECRET missing from livekit.env", flush=True)
        return 1

    api = LiveKitAPI(url=url, api_key=key, api_secret=secret)
    try:
        trunk_id = await ensure_trunk(api.sip)
        await ensure_rule(api.sip, trunk_id)
        print("SIP setup complete: dial sip:jett@129.159.189.244", flush=True)
        return 0
    finally:
        await api.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
