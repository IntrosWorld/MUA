import re

WAKE_WORDS = ["hey assistant", "ok assistant", "assistant"]

INTENT_RULES = [
    ("turn_on_light",    lambda t: "turn on" in t and "light" in t),
    ("turn_off_light",   lambda t: "turn off" in t and "light" in t),
    ("turn_on_fan",      lambda t: "turn on" in t and "fan" in t),
    ("turn_off_fan",     lambda t: "turn off" in t and "fan" in t),
    ("turn_on_ac",       lambda t: "turn on" in t and ("ac" in t or "air" in t)),
    ("turn_off_ac",      lambda t: "turn off" in t and ("ac" in t or "air" in t)),
    ("play_playlist",    lambda t: "play" in t and "playlist" in t),
    ("play_music",       lambda t: "play" in t and ("music" in t or "song" in t or "relaxing" in t)),
    ("get_schedule",     lambda t: "schedule" in t),
    ("set_alarm",        lambda t: "alarm" in t),
    ("set_reminder",     lambda t: "remind" in t or "reminder" in t),
]

DEVICE_MAP = {
    "turn_on_light": "light",
    "turn_off_light": "light",
    "turn_on_fan": "fan",
    "turn_off_fan": "fan",
    "turn_on_ac": "ac",
    "turn_off_ac": "ac",
    "play_music": "music",
    "play_playlist": "playlist",
}


def _has_wake_word(text: str) -> bool:
    return any(ww in text for ww in WAKE_WORDS)


def detect_intent(text: str) -> dict:
    """
    Returns:
    {
        "is_command": bool,
        "intent": str,
        "wake_word_detected": bool,
        "slots": dict
    }
    """
    normalized = text.lower().strip()
    # Remove punctuation
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    wake = _has_wake_word(normalized)

    # Strip wake word from text for intent matching
    clean = normalized
    for ww in sorted(WAKE_WORDS, key=len, reverse=True):
        clean = clean.replace(ww, " ").strip()

    for intent, rule in INTENT_RULES:
        if rule(clean) or rule(normalized):
            device = DEVICE_MAP.get(intent)
            slots = {"device": device} if device else {}
            return {
                "is_command": True,
                "intent": intent,
                "wake_word_detected": wake,
                "slots": slots,
            }

    if not wake and not any(kw in normalized for kw in ["turn", "play", "schedule", "alarm", "remind", "set"]):
        return {
            "is_command": False,
            "intent": "no_command",
            "wake_word_detected": wake,
            "slots": {},
        }

    return {
        "is_command": wake,
        "intent": "unknown_command",
        "wake_word_detected": wake,
        "slots": {},
    }
