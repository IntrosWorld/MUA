"""
User Profile Manager
--------------------
Each user has a rich workspace stored as JSON in data/profiles/<name>.json
Sections: calendar, fitness, movies, music, notes, reminders, smart_home
"""

import os
import json
from datetime import datetime, date

BASE_DIR     = os.environ.get("MUA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

TODAY = date.today().isoformat()

# ── Default demo profiles ───────────────────────────────────────────────────
DEFAULTS = {
    "mohit": {
        "name": "Mohit",
        "display_name": "Mohit",
        "avatar_color": "#7c3aed",
        "avatar_emoji": "👨‍💻",
        "type": "tts",
        "room": "Mohit's bedroom",
        "workspace": {
            "calendar": [
                {"id": "c1", "date": TODAY, "time": "10:00", "event": "Math class",         "category": "academic", "done": False},
                {"id": "c2", "date": TODAY, "time": "18:00", "event": "Robotics club meeting","category": "club",     "done": False},
                {"id": "c3", "date": TODAY, "time": "21:00", "event": "Study session",       "category": "personal", "done": False},
            ],
            "fitness": {
                "goal": "Build muscle",
                "weight_kg": 68,
                "height_cm": 175,
                "steps_today": 8500,
                "calories_burned": 420,
                "workouts": [
                    {"date": TODAY, "type": "Chest & Triceps", "duration_min": 60, "calories": 350, "notes": "Increased bench by 5kg"},
                    {"date": TODAY, "type": "Running",          "duration_min": 30, "calories": 280, "notes": "5 km park run"},
                ],
                "water_ml": 1800,
            },
            "movies": {
                "watchlist": ["Dune Part 2", "Oppenheimer", "Mission Impossible 8", "Deadpool & Wolverine"],
                "watched": [
                    {"title": "Interstellar",    "rating": 10, "date": "2026-04-10", "review": "Mind-blowing visuals"},
                    {"title": "The Dark Knight", "rating": 9,  "date": "2026-03-22", "review": "Best superhero film ever"},
                    {"title": "Inception",       "rating": 9,  "date": "2026-02-14", "review": "Multiple rewatches needed"},
                ],
                "favorite_genres": ["Sci-Fi", "Thriller", "Action"],
            },
            "music": {
                "currently_playing": "Bohemian Rhapsody - Queen",
                "favorite_genres": ["Rock", "Metal", "Alternative"],
                "playlists": [
                    {"name": "Rock Classics",   "tracks": ["Bohemian Rhapsody", "Hotel California", "Stairway to Heaven", "Sweet Child O Mine"]},
                    {"name": "Workout Pump",    "tracks": ["Eye of the Tiger", "Lose Yourself", "Till I Collapse", "Stronger"]},
                    {"name": "Late Night Code", "tracks": ["Lo-fi Hip Hop", "Synthwave Chill", "Dark Ambient", "Blade Runner 2049 OST"]},
                ],
            },
            "notes": [
                {"id": "n1", "date": TODAY, "text": "Call mom this weekend",         "pinned": True},
                {"id": "n2", "date": TODAY, "text": "Buy robotics components online","pinned": False},
                {"id": "n3", "date": TODAY, "text": "Submit ML assignment by Friday", "pinned": True},
            ],
            "reminders": [
                {"id": "r1", "datetime": f"{TODAY} 09:00", "text": "Take vitamins",              "done": False},
                {"id": "r2", "datetime": f"{TODAY} 14:00", "text": "Assignment submission",      "done": False},
                {"id": "r3", "datetime": f"{TODAY} 20:00", "text": "Call Rahul about project",   "done": True},
            ],
            "smart_home": {
                "bedroom_light": False,
                "fan": False,
                "ac": False,
                "ac_temp": 24,
                "tv": False,
            },
        },
    },

    "rahul": {
        "name": "Rahul",
        "display_name": "Rahul",
        "avatar_color": "#db2777",
        "avatar_emoji": "🧑‍🔬",
        "type": "tts",
        "room": "Rahul's study room",
        "workspace": {
            "calendar": [
                {"id": "c1", "date": TODAY, "time": "09:00", "event": "Morning lab session",        "category": "academic", "done": False},
                {"id": "c2", "date": TODAY, "time": "14:00", "event": "Research paper review",      "category": "academic", "done": False},
                {"id": "c3", "date": TODAY, "time": "19:00", "event": "Gym",                         "category": "fitness",  "done": False},
            ],
            "fitness": {
                "goal": "Lose weight & stay active",
                "weight_kg": 75,
                "height_cm": 180,
                "steps_today": 12000,
                "calories_burned": 600,
                "workouts": [
                    {"date": TODAY, "type": "HIIT Cardio",  "duration_min": 45, "calories": 450, "notes": "New PR on row machine"},
                    {"date": TODAY, "type": "Back & Biceps","duration_min": 50, "calories": 300, "notes": ""},
                ],
                "water_ml": 2500,
            },
            "movies": {
                "watchlist": ["Arrival", "Ex Machina", "The Martian", "2001 A Space Odyssey"],
                "watched": [
                    {"title": "Interstellar",       "rating": 10, "date": "2026-04-18", "review": "Best hard sci-fi"},
                    {"title": "The Social Network", "rating": 8,  "date": "2026-03-30", "review": "Incredibly written"},
                ],
                "favorite_genres": ["Sci-Fi", "Documentary", "Drama"],
            },
            "music": {
                "currently_playing": None,
                "favorite_genres": ["Classical", "Lo-fi", "Jazz"],
                "playlists": [
                    {"name": "Focus Flow",      "tracks": ["Bach Cello Suite", "Chopin Nocturne", "Lo-fi Beats", "Coffee Shop Ambience"]},
                    {"name": "Lab Background",  "tracks": ["White Noise", "Brown Noise", "Rain Sounds", "Café Ambience"]},
                    {"name": "Evening Unwind",  "tracks": ["Miles Davis - Kind of Blue", "Coltrane - Blue Train", "Chet Baker Sings"]},
                ],
            },
            "notes": [
                {"id": "n1", "date": TODAY, "text": "Read Attention Is All You Need paper again", "pinned": True},
                {"id": "n2", "date": TODAY, "text": "Order lab supplies — micropipettes",          "pinned": False},
            ],
            "reminders": [
                {"id": "r1", "datetime": f"{TODAY} 07:00", "text": "Morning run",               "done": True},
                {"id": "r2", "datetime": f"{TODAY} 17:00", "text": "Submit weekly lab report",  "done": False},
                {"id": "r3", "datetime": f"{TODAY} 22:00", "text": "Set alarm for tomorrow",    "done": False},
            ],
            "smart_home": {
                "bedroom_light": True,
                "fan": False,
                "ac": True,
                "ac_temp": 22,
                "tv": False,
            },
        },
    },

    "priya": {
        "name": "Priya",
        "display_name": "Priya",
        "avatar_color": "#0891b2",
        "avatar_emoji": "👩‍🎨",
        "type": "tts",
        "room": "Priya's room",
        "workspace": {
            "calendar": [
                {"id": "c1", "date": TODAY, "time": "11:00", "event": "Design team standup",   "category": "work",     "done": False},
                {"id": "c2", "date": TODAY, "time": "14:30", "event": "Client presentation",   "category": "work",     "done": False},
                {"id": "c3", "date": TODAY, "time": "20:00", "event": "Call home",             "category": "personal", "done": False},
            ],
            "fitness": {
                "goal": "Stay flexible & mindful",
                "weight_kg": 55,
                "height_cm": 163,
                "steps_today": 7200,
                "calories_burned": 320,
                "workouts": [
                    {"date": TODAY, "type": "Yoga",     "duration_min": 45, "calories": 180, "notes": "Morning flow — felt great"},
                    {"date": TODAY, "type": "Pilates",  "duration_min": 40, "calories": 200, "notes": "Core strengthening"},
                ],
                "water_ml": 2000,
            },
            "movies": {
                "watchlist": ["Portrait of a Lady on Fire", "Everything Everywhere All at Once", "Past Lives"],
                "watched": [
                    {"title": "La La Land",     "rating": 9,  "date": "2026-04-20", "review": "Visually stunning"},
                    {"title": "Amélie",         "rating": 10, "date": "2026-03-10", "review": "A masterpiece"},
                    {"title": "Little Women",   "rating": 9,  "date": "2026-02-28", "review": "Beautiful adaptation"},
                ],
                "favorite_genres": ["Drama", "Romance", "Art House"],
            },
            "music": {
                "currently_playing": "Weightless - Marconi Union",
                "favorite_genres": ["Ambient", "Indie", "Classical"],
                "playlists": [
                    {"name": "Relaxing Vibes",  "tracks": ["Weightless", "Clair de Lune", "Gymnopédie No.1", "River Flows In You"]},
                    {"name": "Creative Work",   "tracks": ["Explosions in the Sky", "Sigur Rós", "Ólafur Arnalds", "Max Richter"]},
                    {"name": "Morning Energy",  "tracks": ["Good as Hell", "Levitating", "Watermelon Sugar", "Blinding Lights"]},
                ],
            },
            "notes": [
                {"id": "n1", "date": TODAY, "text": "Finish UI mockups for client by Thursday", "pinned": True},
                {"id": "n2", "date": TODAY, "text": "Book flight home for Diwali",              "pinned": True},
                {"id": "n3", "date": TODAY, "text": "Try new recipe — ramen from scratch",      "pinned": False},
            ],
            "reminders": [
                {"id": "r1", "datetime": f"{TODAY} 08:00", "text": "Morning yoga",                  "done": True},
                {"id": "r2", "datetime": f"{TODAY} 13:00", "text": "Lunch with design team",        "done": False},
                {"id": "r3", "datetime": f"{TODAY} 20:00", "text": "Call home",                     "done": False},
            ],
            "smart_home": {
                "bedroom_light": True,
                "fan": True,
                "ac": False,
                "ac_temp": 26,
                "tv": False,
            },
        },
    },
}


# ── CRUD ────────────────────────────────────────────────────────────────────

def _path(name: str) -> str:
    if not name or name in ('.', '..') or any(c in name for c in ('/', '\\', '\0')):
        raise ValueError('Invalid profile name')
    return os.path.join(PROFILES_DIR, f"{name.lower()}.json")


def get_profile(name: str) -> dict | None:
    p = _path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    # Auto-create from defaults
    key = name.lower()
    if key in DEFAULTS:
        save_profile(DEFAULTS[key])
        return DEFAULTS[key]
    return None


def save_profile(profile: dict):
    name = profile.get("name", profile.get("display_name", "unknown")).lower()
    with open(_path(name), "w") as f:
        json.dump(profile, f, indent=2)


def list_profiles() -> list:
    profiles = []
    # Ensure defaults exist
    for key in DEFAULTS:
        get_profile(key)
    for fname in os.listdir(PROFILES_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(PROFILES_DIR, fname)) as f:
                p = json.load(f)
            profiles.append({
                "name": p.get("name", fname[:-5]),
                "display_name": p.get("display_name", p.get("name", "")),
                "avatar_color": p.get("avatar_color", "#7c3aed"),
                "avatar_emoji": p.get("avatar_emoji", "👤"),
                "type": p.get("type", "real"),
                "room": p.get("room", ""),
            })
    return profiles


def update_workspace_section(name: str, section: str, data: dict | list):
    profile = get_profile(name)
    if not profile:
        raise ValueError(f"Profile '{name}' not found.")
    profile["workspace"][section] = data
    save_profile(profile)
    return profile


def update_smart_home(name: str, device: str, state):
    profile = get_profile(name)
    if not profile:
        raise ValueError(f"Profile '{name}' not found.")
    profile["workspace"]["smart_home"][device] = state
    save_profile(profile)
    return profile["workspace"]["smart_home"]


def add_calendar_event(name: str, event: dict):
    profile = get_profile(name)
    if not profile:
        raise ValueError(f"Profile '{name}' not found.")
    import uuid
    event["id"] = event.get("id", str(uuid.uuid4())[:8])
    profile["workspace"]["calendar"].append(event)
    save_profile(profile)
    return event


def add_note(name: str, text: str):
    profile = get_profile(name)
    if not profile:
        raise ValueError(f"Profile '{name}' not found.")
    import uuid
    note = {"id": str(uuid.uuid4())[:8], "date": TODAY, "text": text, "pinned": False}
    profile["workspace"]["notes"].insert(0, note)
    save_profile(profile)
    return note


def add_reminder(name: str, text: str, dt: str = None):
    profile = get_profile(name)
    if not profile:
        raise ValueError(f"Profile '{name}' not found.")
    import uuid
    reminder = {
        "id": str(uuid.uuid4())[:8],
        "datetime": dt or f"{TODAY} 09:00",
        "text": text,
        "done": False,
    }
    profile["workspace"]["reminders"].append(reminder)
    save_profile(profile)
    return reminder


def get_context_summary(name: str) -> str:
    """Return a concise text summary of the user's profile for LLM context."""
    profile = get_profile(name)
    if not profile:
        return f"No profile found for {name}."

    ws   = profile.get("workspace", {})
    cal  = ws.get("calendar", [])
    fit  = ws.get("fitness", {})
    mov  = ws.get("movies", {})
    mus  = ws.get("music", {})
    notes= ws.get("notes", [])
    rem  = ws.get("reminders", [])
    sh   = ws.get("smart_home", {})

    today_events = [f"{e['time']} - {e['event']}" for e in cal if not e.get("done")]
    pending_rem  = [r["text"] for r in rem if not r.get("done")]
    pinned_notes = [n["text"] for n in notes if n.get("pinned")]
    playlists    = [p["name"] for p in mus.get("playlists", [])]
    watchlist    = mov.get("watchlist", [])[:3]

    lines = [
        f"User: {profile['display_name']}",
        f"Room: {profile.get('room', 'N/A')}",
        f"Today's schedule: {'; '.join(today_events) or 'Nothing scheduled'}",
        f"Pending reminders: {'; '.join(pending_rem) or 'None'}",
        f"Pinned notes: {'; '.join(pinned_notes) or 'None'}",
        f"Fitness today: {fit.get('steps_today',0)} steps, {fit.get('calories_burned',0)} cal burned, goal={fit.get('goal','')}",
        f"Music playlists: {', '.join(playlists)}",
        f"Movie watchlist: {', '.join(watchlist)}",
        f"Smart home: light={'on' if sh.get('bedroom_light') else 'off'}, fan={'on' if sh.get('fan') else 'off'}, AC={'on' if sh.get('ac') else 'off'} at {sh.get('ac_temp','?')}C",
    ]
    return "\n".join(lines)


# Initialise all default profiles on import
for _k in DEFAULTS:
    get_profile(_k)
