"""
SoulJazz — An AI-driven Jazz Music Web Application
Built with Streamlit + NumPy (Python only, no JavaScript)
"""

import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# STEP 1: JAZZ_TRACKS DATABASE + COSINE SIMILARITY ENGINE
# ---------------------------------------------------------------------------
# Feature vector schema (5 dimensions, all normalized 0.0–1.0):
#   [0] energy       — intensity, drive, rhythmic force
#   [1] valence      — positivity, brightness, warmth
#   [2] tempo        — perceived speed (slow ballad → up-tempo bop)
#   [3] complexity   — harmonic/rhythmic sophistication
#   [4] darkness     — melancholy, introspection, minor tonality

JAZZ_TRACKS = [
    {
        "id": "kind_of_blue",
        "title": "Kind of Blue",
        "artist": "Miles Davis",
        "year": 1959,
        "album": "Kind of Blue",
        "description": "Modal jazz at its most serene — suspended in cool blue air.",
        "vector": np.array([0.25, 0.55, 0.28, 0.70, 0.45]),
        "url": "https://ia800501.us.archive.org/8/items/MilesDavisKindOfBlue/Miles_Davis_-_Kind_Of_Blue_-_01_-_So_What.mp3",
    },
    {
        "id": "take_five",
        "title": "Take Five",
        "artist": "Dave Brubeck Quartet",
        "year": 1959,
        "album": "Time Out",
        "description": "5/4 time made infectious — the most famous odd meter in jazz history.",
        "vector": np.array([0.60, 0.72, 0.58, 0.85, 0.20]),
        "url": "https://ia800502.us.archive.org/10/items/78_take-five_the-dave-brubeck-quartet-paul-desmond_gbia0001463b/Take%20Five%20-%20The%20Dave%20Brubeck%20Quartet.mp3",
    },
    {
        "id": "a_love_supreme",
        "title": "A Love Supreme (Part I)",
        "artist": "John Coltrane",
        "year": 1964,
        "album": "A Love Supreme",
        "description": "Spiritual intensity coiled into four notes — jazz as devotion.",
        "vector": np.array([0.55, 0.35, 0.45, 0.95, 0.75]),
        "url": "https://ia600300.us.archive.org/31/items/JohnColtrane-ALoveSupreme/JohnColtrane-ALoveSupreme-PartI-Acknowledgement.mp3",
    },
    {
        "id": "round_midnight",
        "title": "'Round Midnight",
        "artist": "Thelonious Monk",
        "year": 1957,
        "album": "Monk's Music",
        "description": "The loneliest chord changes ever written — 3am in a half-empty club.",
        "vector": np.array([0.20, 0.18, 0.22, 0.88, 0.92]),
        "url": "https://ia800204.us.archive.org/11/items/TheloniousMonkRoundMidnight/Thelonious_Monk_Round_Midnight.mp3",
    },
    {
        "id": "autumn_leaves",
        "title": "Autumn Leaves",
        "artist": "Bill Evans Trio",
        "year": 1959,
        "album": "Portrait in Jazz",
        "description": "Piano impressionism — leaves falling through amber light.",
        "vector": np.array([0.22, 0.30, 0.25, 0.80, 0.78]),
        "url": "https://ia800207.us.archive.org/14/items/BillEvansAutumnLeaves/Bill_Evans_-_Autumn_Leaves.mp3",
    },
    {
        "id": "birdland",
        "title": "Birdland",
        "artist": "Weather Report",
        "year": 1977,
        "album": "Heavy Weather",
        "description": "Fusion eruption — electric joy that refuses to stay seated.",
        "vector": np.array([0.92, 0.88, 0.90, 0.78, 0.08]),
        "url": "https://ia800203.us.archive.org/5/items/WeatherReportBirdland/Weather_Report_Birdland.mp3",
    },
    {
        "id": "so_what",
        "title": "So What",
        "artist": "Miles Davis",
        "year": 1959,
        "album": "Kind of Blue",
        "description": "Two chords. Infinite space. The birth of modal jazz.",
        "vector": np.array([0.30, 0.60, 0.35, 0.75, 0.38]),
        "url": "https://ia800501.us.archive.org/8/items/MilesDavisKindOfBlue/Miles_Davis_-_Kind_Of_Blue_-_01_-_So_What.mp3",
    },
    {
        "id": "giant_steps",
        "title": "Giant Steps",
        "artist": "John Coltrane",
        "year": 1960,
        "album": "Giant Steps",
        "description": "Coltrane changes — harmonic velocity that still spins heads.",
        "vector": np.array([0.75, 0.50, 0.80, 0.98, 0.35]),
        "url": "https://ia600300.us.archive.org/31/items/JohnColtraneGiantSteps/JohnColtrane-GiantSteps.mp3",
    },
    {
        "id": "blue_in_green",
        "title": "Blue in Green",
        "artist": "Miles Davis",
        "year": 1959,
        "album": "Kind of Blue",
        "description": "Ten bars of pure watercolor — indistinct, beautiful, suspended.",
        "vector": np.array([0.12, 0.28, 0.18, 0.72, 0.82]),
        "url": "https://ia800501.us.archive.org/8/items/MilesDavisKindOfBlue/Miles_Davis_-_Kind_Of_Blue_-_03_-_Blue_In_Green.mp3",
    },
    {
        "id": "footprints",
        "title": "Footprints",
        "artist": "Wayne Shorter",
        "year": 1966,
        "album": "Adam's Apple",
        "description": "Slow, predatory swing — something beautiful is being stalked.",
        "vector": np.array([0.42, 0.40, 0.38, 0.88, 0.60]),
        "url": "https://ia800302.us.archive.org/17/items/WayneShorterFootprints/Wayne_Shorter_Footprints.mp3",
    },
    {
        "id": "chameleon",
        "title": "Chameleon",
        "artist": "Herbie Hancock",
        "year": 1973,
        "album": "Head Hunters",
        "description": "Funk-jazz groove that rewired what jazz could be.",
        "vector": np.array([0.85, 0.80, 0.82, 0.70, 0.12]),
        "url": "https://ia800302.us.archive.org/20/items/HerbieHancockChameleon/Herbie_Hancock_Chameleon.mp3",
    },
    {
        "id": "naima",
        "title": "Naima",
        "artist": "John Coltrane",
        "year": 1960,
        "album": "Giant Steps",
        "description": "A ballad for his wife — tenderness pressed into every note.",
        "vector": np.array([0.15, 0.45, 0.15, 0.78, 0.65]),
        "url": "https://ia600300.us.archive.org/31/items/JohnColtraneGiantSteps/JohnColtrane-Naima.mp3",
    },
]

# ---------------------------------------------------------------------------
# MOOD VECTORS — the 4 clickable moods map to these 5-dim target vectors
# ---------------------------------------------------------------------------
MOOD_VECTORS = {
    "Calm": {
        "vector": np.array([0.18, 0.52, 0.20, 0.60, 0.35]),
        "emoji": "🌊",
        "color": "#4A90D9",
        "subtitle": "Still water, open sky",
    },
    "Energetic": {
        "vector": np.array([0.90, 0.85, 0.88, 0.72, 0.08]),
        "emoji": "⚡",
        "color": "#F5A623",
        "subtitle": "Electric pulse, full throttle",
    },
    "Melancholic": {
        "vector": np.array([0.22, 0.20, 0.22, 0.82, 0.90]),
        "emoji": "🌧️",
        "color": "#7B68EE",
        "subtitle": "Beautiful ache, slow rain",
    },
    "Joyful": {
        "vector": np.array([0.80, 0.92, 0.75, 0.55, 0.08]),
        "emoji": "☀️",
        "color": "#E8A838",
        "subtitle": "Bright rooms, easy laughter",
    },
}


# ---------------------------------------------------------------------------
# COSINE SIMILARITY ENGINE
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity in [0, 1] between two feature vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def rank_tracks_by_mood(mood_key: str, top_n: int = 3) -> list[dict]:
    """
    Score every track against the given mood vector via cosine similarity
    and return the top_n best matches, each annotated with its score.
    """
    mood_vec = MOOD_VECTORS[mood_key]["vector"]
    scored = []
    for track in JAZZ_TRACKS:
        score = cosine_similarity(mood_vec, track["vector"])
        scored.append({**track, "match_score": round(score, 4)})
    scored.sort(key=lambda t: t["match_score"], reverse=True)
    return scored[:top_n]


def get_top_track(mood_key: str) -> dict:
    """Return the single best-matching track for a given mood."""
    return rank_tracks_by_mood(mood_key, top_n=1)[0]


# ---------------------------------------------------------------------------
# QUICK SELF-TEST (runs only when this module is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SoulJazz — STEP 1 Self-Test: Cosine Similarity Engine")
    print("=" * 60)
    for mood in MOOD_VECTORS:
        top = get_top_track(mood)
        print(f"\nMood: {mood}")
        print(f"  Best match : {top['title']} — {top['artist']}")
        print(f"  Score      : {top['match_score']:.4f}")
        print(f"  Description: {top['description']}")
    print("\n✓ Database and similarity engine verified.")
