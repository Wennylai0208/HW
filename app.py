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
# STEP 3: CUSTOM CSS — SPINNING VINYL · MOOD CARDS · WARM LIGHTING
# ---------------------------------------------------------------------------

# Fixed display order for the 4 mood cards (maps to CSS nth-child indices)
MOOD_CARD_ORDER: list[str] = ["Calm", "Energetic", "Melancholic", "Joyful"]

# Per-mood accent colours (used in dynamic active-state injection)
_MOOD_COLORS: dict[str, str] = {
    "Calm":       "#4A90D9",
    "Energetic":  "#F5A623",
    "Melancholic":"#9B7FD4",
    "Joyful":     "#E8A838",
}


def get_css() -> str:
    """Return the full <style> block as a string."""
    return """
<style>
/* ================================================================
   SOULJAZZ — VISUAL IDENTITY
   Palette: vinyl black · warm amber · cream · deep burgundy
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── Base ────────────────────────────────────────────────────── */
.stApp {
    background-color: #0D0A08;
    background-image:
        radial-gradient(ellipse 80% 55% at 15% 45%,
            rgba(180,100,20,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 55% 40% at 88% 70%,
            rgba(120,60,10,0.07) 0%, transparent 60%);
    font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
    color: #E8D5B7;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton            { display: none !important; }

::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: #0D0A08; }
::-webkit-scrollbar-thumb { background: #3D2008; border-radius: 3px; }

/* ── Header ──────────────────────────────────────────────────── */
.sj-header {
    text-align: center;
    padding: 2.8rem 1rem 1.4rem;
    border-bottom: 1px solid rgba(200,136,42,0.14);
    margin-bottom: 2.2rem;
}

.sj-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: #C8882A;
    letter-spacing: 0.09em;
    text-shadow: 0 0 44px rgba(200,136,42,0.42), 0 2px 6px rgba(0,0,0,0.9);
    margin: 0;
    line-height: 1;
}

.sj-subtitle {
    font-size: 0.80rem;
    color: #7A6040;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    margin-top: 0.55rem;
}

/* ── Vinyl spinning animation ────────────────────────────────── */
@keyframes vinyl-spin {
    to { transform: rotate(360deg); }
}

@keyframes amber-pulse {
    0%,100% { box-shadow: 0 0 28px rgba(200,136,42,0.22),
                           0 0 60px rgba(200,136,42,0.07); }
    50%      { box-shadow: 0 0 50px rgba(200,136,42,0.48),
                           0 0 90px rgba(200,136,42,0.16); }
}

.sj-vinyl-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.6rem 0 2rem;
    position: relative;
}

/* Tonearm needle */
.sj-arm {
    position: absolute;
    top: 12px;
    right: calc(50% - 136px);
    width: 76px;
    height: 3px;
    background: linear-gradient(90deg, #6B4E12, #C8882A 60%, #6B4E12);
    border-radius: 2px;
    transform-origin: right center;
    transform: rotate(-22deg);
    opacity: 0.65;
    z-index: 10;
}

.sj-arm::after {
    content: '';
    position: absolute;
    right: -4px;
    top: -3px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #C8882A;
    box-shadow: 0 0 6px rgba(200,136,42,0.7);
}

/* Vinyl disc */
.sj-vinyl {
    width: 228px;
    height: 228px;
    border-radius: 50%;
    background:
        /* centre label */
        radial-gradient(circle at 50% 50%,
            #2E1A06 0%, #3D2610 21%, transparent 21.5%),
        /* gold label ring */
        radial-gradient(circle at 50% 50%,
            transparent 20.5%, #C8882A 21%, #8B5A14 23.5%, transparent 24%),
        /* vinyl grooves */
        repeating-radial-gradient(circle at center,
            #181818 0px, #181818 3px,
            #262626 3px, #262626 4.5px,
            #181818 4.5px, #181818 8px,
            #1E1E1E 8px,  #1E1E1E 9px);
    border: 2px solid #2a2a2a;
    position: relative;
}

/* Spinhole */
.sj-vinyl::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #0D0A08;
    z-index: 5;
    box-shadow: 0 0 0 1px #3a3a3a;
}

.sj-vinyl.spinning {
    animation: vinyl-spin 2.6s linear infinite,
               amber-pulse 2.6s ease-in-out infinite;
}

.sj-vinyl.idle {
    animation: amber-pulse 5s ease-in-out infinite;
}

/* ── Mood cards ──────────────────────────────────────────────── */
.sj-mood-label {
    text-align: center;
    font-size: 0.70rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #5A4428;
    margin-bottom: 0.9rem;
}

/* Base style for ALL column buttons (mood cards) */
div[data-testid="stColumn"] .stButton > button {
    width: 100% !important;
    min-height: 115px !important;
    background: linear-gradient(160deg, #191208, #22180A) !important;
    border: 1px solid rgba(200,136,42,0.18) !important;
    border-radius: 14px !important;
    color: #B89A60 !important;
    font-size: 1.6rem !important;
    padding: 1.1rem 0.4rem !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    line-height: 1.45 !important;
    position: relative !important;
    overflow: hidden !important;
    white-space: pre-line !important;
}

div[data-testid="stColumn"] .stButton > button:hover {
    border-color: rgba(200,136,42,0.52) !important;
    background: linear-gradient(160deg, #231A0D, #2E2210) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.55),
                0 0  22px rgba(200,136,42,0.11) !important;
    color: #DEBB7A !important;
}

div[data-testid="stColumn"] .stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Focus ring — accessible but styled */
div[data-testid="stColumn"] .stButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(200,136,42,0.45) !important;
}

/* ── Now Playing panel ───────────────────────────────────────── */
.sj-now-playing {
    background: linear-gradient(140deg, #100E08, #191408);
    border: 1px solid rgba(200,136,42,0.17);
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    margin: 1.2rem 0;
    position: relative;
    overflow: hidden;
}

/* Top accent bar */
.sj-now-playing::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, #C8882A 40%, #F5D898 60%, transparent 100%);
    opacity: 0.55;
}

.sj-np-badge {
    font-size: 0.62rem;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: #C8882A;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 0.7rem;
}

/* Blinking live dot */
.sj-np-badge::before {
    content: '';
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #C8882A;
    animation: blink-dot 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes blink-dot {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.15; }
}

.sj-track-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.0rem;
    font-weight: 700;
    color: #F0E0C0;
    line-height: 1.2;
    margin: 0 0 0.3rem;
}

.sj-track-artist {
    font-size: 1.0rem;
    color: #C8882A;
    letter-spacing: 0.04em;
}

.sj-track-meta {
    font-size: 0.76rem;
    color: #5A4428;
    margin-top: 0.35rem;
}

/* Cosine match score bar */
.sj-score-bar {
    margin-top: 1.3rem;
    height: 3px;
    background: rgba(200,136,42,0.12);
    border-radius: 2px;
    overflow: hidden;
}

.sj-score-fill {
    height: 100%;
    background: linear-gradient(90deg, #6B4010, #C8882A, #F5D898);
    border-radius: 2px;
    transition: width 0.9s cubic-bezier(0.4,0,0.2,1);
}

/* ── Charlie's Jazz Report ───────────────────────────────────── */
.sj-charlie {
    background: rgba(14,10,5,0.85);
    border-left: 3px solid rgba(200,136,42,0.70);
    border-radius: 0 14px 14px 0;
    padding: 1.4rem 1.9rem;
    margin: 0.8rem 0;
    font-size: 0.93rem;
    line-height: 1.80;
    color: #C0A070;
    position: relative;
}

.sj-charlie-tag {
    display: block;
    font-style: normal;
    font-size: 0.60rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #7A5C28;
    margin-bottom: 0.9rem;
}

/* ── Secondary playlist rail ─────────────────────────────────── */
.sj-playlist {
    margin-top: 1.4rem;
    border-top: 1px solid rgba(200,136,42,0.10);
    padding-top: 1.0rem;
}

.sj-playlist-tag {
    font-size: 0.60rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #3E2C18;
    margin-bottom: 0.75rem;
}

.sj-pl-item {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 0.55rem 0.75rem;
    border-radius: 8px;
    border: 1px solid transparent;
    margin-bottom: 0.35rem;
    transition: all 0.18s ease;
}

.sj-pl-item:hover {
    background: rgba(200,136,42,0.05);
    border-color: rgba(200,136,42,0.18);
}

.sj-pl-num   { font-size: 0.68rem; color: #3E2C18; width: 14px; flex-shrink: 0; text-align: center; }
.sj-pl-info  { flex: 1; }
.sj-pl-title { font-size: 0.86rem; color: #C8A068; font-weight: 600; }
.sj-pl-artist{ font-size: 0.73rem; color: #5A4428; margin-top: 1px; }
.sj-pl-score { font-size: 0.70rem; color: #7A5C28; flex-shrink: 0; }

/* ── Next-track button ───────────────────────────────────────── */
.sj-next-wrap .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(200,136,42,0.32) !important;
    color: #C8882A !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.6rem !important;
    border-radius: 30px !important;
    min-height: 0 !important;
    transition: all 0.20s ease !important;
}

.sj-next-wrap .stButton > button:hover {
    background: rgba(200,136,42,0.09) !important;
    border-color: rgba(200,136,42,0.68) !important;
    box-shadow: 0 0 18px rgba(200,136,42,0.14) !important;
    color: #E8A838 !important;
}

/* ── Empty state ─────────────────────────────────────────────── */
.sj-empty {
    text-align: center;
    padding: 3.5rem 2rem;
    color: #3E2C18;
    font-style: italic;
    font-size: 0.98rem;
    line-height: 1.85;
}

/* ── Misc ────────────────────────────────────────────────────── */
.sj-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent, rgba(200,136,42,0.22), transparent);
    margin: 1.4rem 0;
}

.stMarkdown p      { color: inherit; }
.stMarkdown em     { color: #9A7848; font-style: italic; }
.stMarkdown strong { color: #E8C888; }

/* Tighten vertical spacing inside blocks */
div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
</style>
"""


def render_css() -> None:
    """Inject the static CSS block once per Streamlit run."""
    st.markdown(get_css(), unsafe_allow_html=True)


def render_active_mood_style(active_mood: str | None) -> None:
    """
    Inject a tiny <style> block that highlights the currently active
    mood card using its column's nth-child index. This is re-injected
    on every run but is cheap (< 300 bytes).
    """
    if active_mood is None:
        return

    idx   = MOOD_CARD_ORDER.index(active_mood) + 1   # CSS nth-child is 1-based
    color = _MOOD_COLORS.get(active_mood, "#C8882A")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    st.markdown(
        f"""
        <style>
        div[data-testid="stColumn"]:nth-child({idx}) .stButton > button {{
            border-color: rgba({r},{g},{b},0.75) !important;
            background: linear-gradient(160deg, #231A0D, #321E0A) !important;
            box-shadow: 0 0 24px rgba({r},{g},{b},0.18),
                        inset 0 0 12px rgba({r},{g},{b},0.06) !important;
            color: #F0D090 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_vinyl(is_spinning: bool) -> None:
    """Render the CSS vinyl record with the tonearm needle."""
    spin_class = "spinning" if is_spinning else "idle"
    st.markdown(
        f"""
        <div class="sj-vinyl-wrap">
            <div class="sj-arm"></div>
            <div class="sj-vinyl {spin_class}"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_now_playing(track: dict) -> None:
    """Render the Now Playing info panel for a track."""
    match_pct  = int(track["match_score"] * 100)
    bar_width  = match_pct
    st.markdown(
        f"""
        <div class="sj-now-playing">
            <div class="sj-np-badge">Now Playing</div>
            <div class="sj-track-title">{track['title']}</div>
            <div class="sj-track-artist">{track['artist']}</div>
            <div class="sj-track-meta">{track['album']} &middot; {track['year']}</div>
            <div class="sj-score-bar">
                <div class="sj-score-fill" style="width:{bar_width}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_charlie_report(report_text: str) -> None:
    """Render Charlie's recommendation panel."""
    st.markdown(
        f"""
        <div class="sj-charlie">
            <span class="sj-charlie-tag">Charlie&#39;s Jazz Report</span>
            {report_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_playlist_rail(ranked_tracks: list[dict], active_index: int) -> None:
    """Render the secondary 'Also Curated' playlist rail."""
    if not ranked_tracks:
        return
    items_html = ""
    for i, track in enumerate(ranked_tracks):
        label    = "▶" if i == active_index else str(i + 1)
        pct      = int(track["match_score"] * 100)
        items_html += (
            f'<div class="sj-pl-item">'
            f'  <span class="sj-pl-num">{label}</span>'
            f'  <div class="sj-pl-info">'
            f'    <div class="sj-pl-title">{track["title"]}</div>'
            f'    <div class="sj-pl-artist">{track["artist"]}</div>'
            f'  </div>'
            f'  <span class="sj-pl-score">{pct}%</span>'
            f'</div>'
        )
    st.markdown(
        f"""
        <div class="sj-playlist">
            <div class="sj-playlist-tag">Also Curated for This Mood</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
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
