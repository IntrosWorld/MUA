"""
Response Engine
---------------
Generates personalized responses using:
  1. LLM (Llama offline / Gemini online) with full user profile context
  2. Rule-based fallback if LLM is unavailable
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UNKNOWN_RESPONSE = "Unknown speaker detected. Please authenticate first."

# ── Rule-based fallback (fast, no model needed) ────────────────────────────

_RULE_PROFILES = {
    "mohit": {
        "room": "Mohit's bedroom",
        "playlist": "Mohit's rock playlist",
        "schedule": "Math class at 10 AM and robotics meeting at 6 PM",
    },
    "rahul": {
        "room": "Rahul's study room",
        "playlist": "Rahul's focus playlist",
        "schedule": "Lab at 2 PM and gym at 7 PM",
    },
    "priya": {
        "room": "Priya's room",
        "playlist": "Priya's relaxing playlist",
        "schedule": "Design standup at 11 AM and call home at 8 PM",
    },
}


def _rule_based(speaker: str, intent: str, slots: dict, transcript: str) -> str:
    key = speaker.lower()
    p   = _RULE_PROFILES.get(key, {})
    name = speaker

    default_room = name + "'s room"
    default_pl   = name + "'s playlist"
    if intent == "turn_on_light":   return f"Turning on the light in {p.get('room', default_room)}."
    if intent == "turn_off_light":  return f"Turning off the light in {p.get('room', default_room)}."
    if intent == "turn_on_fan":     return f"Turning on the fan in {p.get('room', default_room)}."
    if intent == "turn_off_fan":    return f"Turning off the fan in {p.get('room', default_room)}."
    if intent == "turn_on_ac":      return f"Turning on the AC in {p.get('room', default_room)}."
    if intent == "turn_off_ac":     return f"Turning off the AC in {p.get('room', default_room)}."
    if intent in ("play_music", "play_playlist"): return f"Playing {p.get('playlist', default_pl)}."
    if intent == "get_schedule":    return f"{name}, your schedule is: {p.get('schedule', 'nothing today')}."
    if intent == "set_alarm":
        import re
        t = re.search(r"(\d+\s*(am|pm)|seven|eight|nine|ten)", transcript.lower())
        return f"Alarm set for {t.group(0) if t else 'the requested time'}, {name}."
    if intent == "set_reminder":
        import re
        text = re.sub(r"(hey |ok )?assistant[,.]?", "", transcript.lower()).strip()
        text = re.sub(r"remind me to", "", text).strip(" .,!") or "your task"
        return f"Reminder saved for {name}: {text}."
    if intent == "unknown_command": return f"Sorry {name}, I didn't understand that command."
    if intent == "no_command":      return "No command detected."
    return f"Got it, {name}."


# ── Main entry point ────────────────────────────────────────────────────────

def generate_response(
    speaker: str,
    intent: str,
    slots: dict = None,
    transcript: str = "",
    use_llm: bool = True,
    llm_mode: str = None,      # None = use global setting, "offline" or "gemini"
) -> dict:
    """
    Returns:
        {
          "response": str,
          "source": "llm" | "rule",
          "llm_mode": str | None,
          "model": str | None,
        }
    """
    if not speaker or speaker == "Unknown":
        return {"response": UNKNOWN_RESPONSE, "source": "rule", "llm_mode": None, "model": None}

    # ── Try LLM path ──
    if use_llm:
        try:
            from .user_profiles import get_profile, get_context_summary
            from .llm_engine import generate as llm_generate

            profile = get_profile(speaker)
            if profile:
                context = get_context_summary(speaker)
                result  = llm_generate(
                    user_name=speaker,
                    user_context=context,
                    transcript=transcript,
                    intent=intent,
                    mode=llm_mode,
                )
                if result.get("error") is None:
                    return {
                        "response": result["response"],
                        "source":   "llm",
                        "llm_mode": result["mode"],
                        "model":    result["model"],
                    }
        except Exception as e:
            print(f"[Response] LLM path failed: {e}. Using rule-based fallback.")

    # ── Rule-based fallback ──
    resp = _rule_based(speaker, intent, slots or {}, transcript)
    return {"response": resp, "source": "rule", "llm_mode": None, "model": None}
