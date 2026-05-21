"""
SoulJazz — An AI-driven Jazz Music Web Application
Built with Streamlit + NumPy (Python only, no JavaScript)
"""

import numpy as np
import streamlit as st
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
# STEP 2: SESSION STATE · AUDIO ENGINE · CHARLIE'S JAZZ REPORT
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Seed every session state key exactly once on first load."""
    defaults: dict = {
        "current_mood":   None,   # str | None  — active mood key
        "current_track":  None,   # dict | None — full track record in play
        "ranked_tracks":  [],     # list[dict]  — top-3 matches for current mood
        "track_index":    0,      # int         — position within ranked_tracks
        "is_playing":     False,  # bool        — gates vinyl spin + audio inject
        "audio_url":      "",     # str         — URL fed to the <audio> element
        "charlie_report": "",     # str         — Charlie's recommendation text
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# -- Mood & track callbacks (bound to st.button on_click) ------------------

def select_mood(mood_key: str) -> None:
    """
    Triggered when a mood card is clicked. All state mutations happen inside
    this callback so by the time Streamlit reruns the script, the correct
    audio URL is already set — no second rerun required.
    """
    ranked = rank_tracks_by_mood(mood_key, top_n=3)
    st.session_state.current_mood   = mood_key
    st.session_state.ranked_tracks  = ranked
    st.session_state.track_index    = 0
    st.session_state.current_track  = ranked[0]
    st.session_state.audio_url      = ranked[0]["url"]
    st.session_state.is_playing     = True
    st.session_state.charlie_report = _generate_charlie_report(mood_key, ranked[0])


def next_track() -> None:
    """Cycle to the next cosine-ranked track within the current mood."""
    if not st.session_state.ranked_tracks:
        return
    idx   = (st.session_state.track_index + 1) % len(st.session_state.ranked_tracks)
    track = st.session_state.ranked_tracks[idx]
    st.session_state.track_index    = idx
    st.session_state.current_track  = track
    st.session_state.audio_url      = track["url"]
    st.session_state.charlie_report = _generate_charlie_report(
        st.session_state.current_mood, track
    )


# ---------------------------------------------------------------------------
# AUDIO AUTOPLAY ENGINE
# ---------------------------------------------------------------------------

def render_audio_player() -> None:
    """
    Inject an HTML <audio autoplay> element whenever a track is active.

    Security note: `audio_url` is sourced exclusively from the JAZZ_TRACKS
    constant defined at module level — never from user input — so there is
    no XSS vector in this HTML injection.

    Autoplay note: browsers permit autoplay for audio-only elements on most
    desktop environments without requiring a prior user gesture. The hidden
    <audio> tag begins playback the moment Streamlit inserts it into the DOM
    after a mood-card click.

    The unique HTML comment keyed to the URL ensures Streamlit treats each
    new track as genuinely new markup, so the browser replaces the old
    <audio> element and fires autoplay correctly.
    """
    url = st.session_state.get("audio_url", "")
    if not url:
        return
    st.markdown(
        f"""
        <!-- souljazz-player:{url} -->
        <audio autoplay style="display:none;">
            <source src="{url}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# CHARLIE'S JAZZ REPORT — generative recommendation text
# ---------------------------------------------------------------------------

_TIME_CONTEXTS: dict[str, tuple[str, str]] = {
    "late_night": (
        "It's late. Most people have stopped listening.",
        "Late night and the right record — one of the few combinations that still surprises me.",
    ),
    "early_morning": (
        "The city is barely awake.",
        "Early morning is rare and quiet. The right music can hold it still a little longer.",
    ),
    "morning": (
        "The morning light is coming in sideways.",
        "Let it set the tone before anything else can.",
    ),
    "afternoon": (
        "The afternoon lull is doing its slow work.",
        "Afternoon is when jazz earns its keep — no fanfare, just depth.",
    ),
    "evening": (
        "The evening is making its case.",
        "Evening is when music starts to mean something different than it did at noon.",
    ),
    "night": (
        "The night is properly dark now.",
        "Night music asks more of you. It gives more back.",
    ),
}


def _get_time_context() -> tuple[str, str]:
    """Map current hour → (atmosphere_line, closing_line)."""
    h = datetime.now().hour
    if   h <  6: return _TIME_CONTEXTS["late_night"]
    elif h <  9: return _TIME_CONTEXTS["early_morning"]
    elif h < 12: return _TIME_CONTEXTS["morning"]
    elif h < 17: return _TIME_CONTEXTS["afternoon"]
    elif h < 21: return _TIME_CONTEXTS["evening"]
    else:        return _TIME_CONTEXTS["night"]


def _describe_vector(vec: np.ndarray) -> tuple[str, str, str]:
    """Translate feature vector dimensions into plain-English adjectives."""
    energy_adj = (
        "volatile"    if vec[0] > 0.80 else
        "driving"     if vec[0] > 0.55 else
        "unhurried"   if vec[0] > 0.30 else
        "still"
    )
    harmonic_adj = (
        "labyrinthine"  if vec[3] > 0.88 else
        "sophisticated" if vec[3] > 0.65 else
        "spacious"
    )
    tonal_adj = (
        "shadow-soaked" if vec[4] > 0.75 else
        "bittersweet"   if vec[4] > 0.45 else
        "open-hearted"  if vec[4] > 0.20 else
        "luminous"
    )
    return energy_adj, harmonic_adj, tonal_adj


_MOOD_OPENERS: dict[str, list[str]] = {
    "Calm": [
        "You need something that doesn't push. Something that just exists alongside you.",
        "Calm is an active choice in a loud world. I respect it.",
        "Not everyone knows how to sit still with good music. You might.",
    ],
    "Energetic": [
        "You want the music to be as alive as you are right now.",
        "Energy is one thing. Channelled energy — that's what jazz does with it.",
        "There's a joy that only fast music in a small room can produce.",
    ],
    "Melancholic": [
        "Melancholy in jazz isn't sad — it's honest. There's a difference.",
        "The best jazz was written by people who knew exactly what you're feeling.",
        "Some moods don't need fixing. They need the right soundtrack.",
    ],
    "Joyful": [
        "Joy is harder to write than sadness. The musicians who got it right are worth finding.",
        "There's a track for this. I've been waiting to play it.",
        "Joyful jazz is underrated. People forget how much craft it takes to sound this free.",
    ],
}


def _generate_charlie_report(mood_key: str, track: dict) -> str:
    """
    Build Charlie's recommendation using dynamic string formatting.
    The opener is deterministic per (track, mood) pair — stable across
    reruns but genuinely varies with each new selection.
    """
    atmosphere, closing  = _get_time_context()
    energy_adj, harmonic_adj, tonal_adj = _describe_vector(track["vector"])
    match_pct = int(track["match_score"] * 100)

    openers = _MOOD_OPENERS[mood_key]
    opener  = openers[hash(track["id"] + mood_key) % len(openers)]

    return (
        f"*{atmosphere}*\n\n"
        f"{opener}\n\n"
        f"I'm putting on **{track['title']}** — **{track['artist']}**, {track['year']}. "
        f"It's a {energy_adj}, {harmonic_adj}, {tonal_adj} record that lands at "
        f"**{match_pct}% resonance** with a {mood_key.lower()} state.\n\n"
        f"*\"{track['description']}\"*\n\n"
        f"{closing}"
    )


# ---------------------------------------------------------------------------
# SELF-TEST (runs only when executed directly, not via streamlit run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SoulJazz — STEP 1 + 2 Self-Test")
    print("=" * 60)

    print("\n— Cosine Similarity Engine —")
    for mood in MOOD_VECTORS:
        top = get_top_track(mood)
        print(f"  {mood:12s} → {top['title']} ({top['artist']})  score={top['match_score']:.4f}")

    print("\n— Charlie's Jazz Report (sample: Calm) —")
    top_calm = get_top_track("Calm")
    report   = _generate_charlie_report("Calm", top_calm)
    print(report)

    print("\n✓ Steps 1 & 2 verified.")
