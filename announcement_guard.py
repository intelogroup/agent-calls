"""Terminal machine-announcement detection for outbound calls.

Built 2026-11-02 after the 19:53Z test call: an outbound call to
Jim's own number was answered by the line's recorded busy
announcement ("Jett's line is on another call right now. Try again
in a bit."), and the agent treated it as a human turn, replying to
the recording 17 s later. A carrier/line announcement is not a
conversation: when the far end's FIRST turn is one of these
terminal announcements, the call should end promptly and the
record should say why.

Deliberately conservative, in the dialer's TOOL_NEEDS tradition:
only phrases a carrier or line recording says and a human first
turn essentially never does. Two classes are excluded outright,
checked BEFORE any pattern:

  - voicemail greetings ("leave a message", "after the beep", a
    mailbox prompt) — the existing voicemail path governs them
    (and Jim's 2026-11-02 rule: never leave a message);
  - IVR menus ("press 1", "main menu") — menus are navigable
    (the local pipeline has press_digits); a menu is not a
    terminal announcement.

The busy pattern needs a second signal — a try-again tail —
because a live receptionist can say "he is on another call" and
then take a message (and if they offer one, the voicemail
exclusion above already keeps the call alive).
"""

import re

# Checked first: any hit means NOT a terminal announcement.
_VOICEMAIL_MARKERS = (
    "leave a message", "leave your message", "voicemail",
    "voice mail", "mailbox", "after the beep", "after the tone",
    "at the tone", "record your message", "record a message",
    "unavailable to take your call", "not available to take",
    "can't take your call", "cannot take your call",
    "can not take your call",
)
_MENU_MARKERS = (
    "press", "oprima", "menu", "options have changed",
    "listen to the following options", "for sales", "for support",
    "for billing", "for customer service",
)

# (label, regex) on normalized text. Every pattern here is a
# carrier/line-recording phrase, not conversational English.
_PATTERNS = (
    ("circuits_busy", r"\ball circuits are busy\b"),
    ("not_in_service", r"\bnot in service\b"),
    ("not_in_service", r"\bno longer in service\b"),
    ("not_in_service", r"\bnot a working number\b"),
    ("disconnected", r"\bhas been disconnected\b"),
    ("disconnected", r"\bnumber is disconnected\b"),
    ("disconnected", r"\bhas been changed\b"),
    ("cannot_complete", r"\bcannot be completed as dialed\b"),
    ("cannot_complete", r"\bcall cannot be completed\b"),
    ("cannot_complete", r"\bcould not be completed\b"),
    ("cannot_complete", r"\bdid not go through\b"),
    ("check_number", r"\bplease check the number\b"),
    ("check_number", r"\bcheck the number and (try|dial)\b"),
    ("intercept", r"\btry your call again\b"),
)
_BUSY_RE = re.compile(r"\bis on another call\b")
_TRY_AGAIN_RE = re.compile(
    r"\btry again\b|\bplease try\b|\bcall again\b")


def normalize_far_end(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed —
    STT output is punctuated inconsistently ("Jett's" arrived as
    "Jeff's"), so matching never leans on names or punctuation."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def machine_announcement_match(text: str) -> str | None:
    """Label of the terminal announcement in a far-end turn, or
    None. Pure function of the turn text; the caller decides
    whether the turn is in scope (outbound, first turn)."""
    t = normalize_far_end(text)
    if not t:
        return None
    if any(m in t for m in _VOICEMAIL_MARKERS):
        return None
    if any(m in t for m in _MENU_MARKERS):
        return None
    if _BUSY_RE.search(t):
        # Second signal required: carrier announcements pair the
        # busy claim with a try-again tail; a bare "is on another
        # call" can be a live receptionist mid-sentence, and the
        # call must survive her finishing it.
        if _TRY_AGAIN_RE.search(t):
            return "busy_line"
        return None
    for label, pat in _PATTERNS:
        if re.search(pat, t):
            return label
    return None
