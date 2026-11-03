"""Farewell guard matcher tests (2026-11-03).

The bridge's hang-up cannot rest on the model remembering end_call:
on the 2026-11-03 proof call the caller said "No. That's okay.
Thank you. Bye bye.", the model produced no reply and no end_call,
and the caller waited out 12 s of silence and hung up himself.
farewell_match decides, from code, that a turn is over. It must
fire on pure farewells and NEVER on reported speech, idioms, or
future arrangements.
"""
import deepgram_bridge as db

FIRES = [
    "Bye bye.",
    "No. That's okay. Thank you. Bye bye.",   # the 2026-11-03 proof call
    "Goodbye.",
    "goodbye",
    "Thanks, bye.",
    "Thank you. Goodbye.",
    "That's all, thanks.",
    "That is all.",
    "Nothing else, thank you.",
    "See you later.",
    "Take care.",
    "Okay, take care now.",
    "Alright. See you.",
    "Good night.",
    "I'm all set, thanks. Bye.",
]

NO_FIRE = [
    "Tell him I said bye.",
    "Tell Jim bye from me.",
    "By the way, the appointment moved.",
    "I need to buy more time.",
    "See you Friday at three.",
    "See you at the bowling appointment.",
    "Thank you.",                                # politeness alone
    "No.",                                       # refusal alone
    "Okay.",
    "Thanks for the update on Friday's call.",
    "The word goodbye means a lot here.",
    "He said that's all the money we have.",
    "Nothing else matters right now, we need the report.",
    "Take care of the paperwork for me.",
    "Yes, please pass the message along.",
    "",
    "Friday is maintained.",
]


def main():
    bad = []
    for text in FIRES:
        if not db.farewell_match(text):
            bad.append(("should fire", text))
    for text in NO_FIRE:
        if db.farewell_match(text):
            bad.append(("should NOT fire", text))
    for kind, text in bad:
        print("FAIL", kind, repr(text), "->",
              db.farewell_match(text))
    assert not bad, f"{len(bad)} matcher failures"
    print(f"farewell_match: {len(FIRES) + len(NO_FIRE)}"
          f"/{len(FIRES) + len(NO_FIRE)} passed")


if __name__ == "__main__":
    main()
