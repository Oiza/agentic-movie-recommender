"""
Hybrid movie recommendation engine.

The implementation keeps the configured LLM model unchanged, but avoids relying
on the model for the final JSON payload. Instead, it uses the model as a query
planner and combines that plan with deterministic SQL, pandas, and lightweight
vector-style retrieval over the local CSV.
"""

import argparse
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

import ollama
import pandas as pd


MODEL = "gemma4:31b-cloud"
REMOTE_QUERY_PLAN_ENABLED = os.getenv("ENABLE_REMOTE_QUERY_PLAN", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
MOVIES = pd.read_csv(DATA_PATH).copy()
MOVIES["tmdb_id"] = MOVIES["tmdb_id"].astype(int)
MOVIES["year"] = pd.to_numeric(MOVIES["year"], errors="coerce").fillna(0).astype(int)

TEXT_COLUMNS = [
    "title",
    "original_title",
    "genres",
    "overview",
    "tagline",
    "director",
    "top_cast",
    "keywords",
    "production_countries",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "movie",
    "movies",
    "my",
    "of",
    "on",
    "or",
    "something",
    "that",
    "the",
    "their",
    "them",
    "to",
    "want",
    "with",
}

FACET_SYNONYMS = {
    "comedy": ["comedy", "funny", "humor", "humorous", "laugh"],
    "action": ["action", "fight", "fighting", "explosive"],
    "adventure": ["adventure", "quest", "journey", "epic"],
    "science fiction": ["science fiction", "sci-fi", "sci fi", "space", "futuristic", "alien"],
    "crime": ["crime", "criminal", "heist", "gangster", "detective"],
    "drama": ["drama", "dramatic", "emotional", "character-driven"],
    "fantasy": ["fantasy", "magical", "mythic", "fairytale"],
    "history": ["history", "historical", "period piece", "true events"],
    "romance": ["romance", "romantic", "love", "relationship"],
    "horror": ["horror", "scary", "spooky", "creepy", "terrifying"],
    "music": ["music", "musical", "songs", "singing"],
    "mystery": ["mystery", "mysterious", "whodunit", "detective"],
    "thriller": ["thriller", "suspense", "suspenseful", "tense"],
    "animation": ["animation", "animated", "pixar", "disney"],
    "family": ["family", "kids", "kid-friendly", "children"],
    "documentary": ["documentary", "doc", "real-life", "nonfiction"],
    "tv movie": ["tv movie", "television movie", "made for tv"],
    "war": ["war", "battlefront", "soldiers", "military"],
    "western": ["western", "cowboy", "frontier", "gunslinger"],
    "superhero": [
        "superhero",
        "superheroes",
        "super hero",
        "comic",
        "comic book",
        "marvel",
        "dc",
        "avenger",
        "batman",
        "spider-man",
        "spiderman",
        "mutant",
        "super power",
        "masked hero",
    ],
    "feel-good": ["feel-good", "feel good", "uplifting", "heartwarming", "lighthearted", "warm", "wholesome"],
    "dark": ["dark", "bleak", "gritty", "disturbing"],
    "psychological": ["psychological", "mind-bending", "mind bending", "twisty"],
    "fast-paced": ["fast-paced", "fast paced", "high energy", "high-energy", "quick", "intense"],
}

FACET_PATTERNS = {
    "comedy": ["comedy", "funny", "humor", "humorous", "laugh"],
    "action": ["action", "fight", "fighting", "martial arts", "gunfight", "battle"],
    "adventure": ["adventure", "quest", "journey", "expedition", "treasure"],
    "science fiction": ["science fiction", "sci-fi", "space", "alien", "futuristic", "time travel", "robot"],
    "crime": ["crime", "criminal", "heist", "gangster", "detective", "underworld"],
    "drama": ["drama", "dramatic", "emotional", "family conflict", "character study"],
    "fantasy": ["fantasy", "magic", "myth", "legend", "fairy tale"],
    "history": ["history", "historical", "period drama", "true story", "biography"],
    "romance": ["romance", "romantic", "love", "relationship"],
    "horror": ["horror", "scary", "creepy", "terrifying", "monster"],
    "music": ["music", "musical", "song", "singing", "band"],
    "mystery": ["mystery", "murder mystery", "whodunit", "investigation", "detective"],
    "thriller": ["thriller", "suspense", "suspenseful", "tense"],
    "animation": ["animation", "animated", "anime", "cartoon", "pixar", "disney"],
    "family": ["family", "kids", "children"],
    "documentary": ["documentary", "real life", "true story", "archive footage", "nonfiction"],
    "tv movie": ["tv movie", "made for tv", "television event"],
    "war": ["war", "soldier", "military", "battle", "army"],
    "western": ["western", "cowboy", "frontier", "gunslinger", "outlaw"],
    "superhero": [
        "superhero",
        "superheroes",
        "super hero",
        "based on comic",
        "marvel cinematic universe",
        "dc universe",
        "dc extended universe",
        "super power",
        "supervillain",
        "masked superhero",
        "hero, superhero",
        "mutant",
        "avenger",
        "spider-man",
        "batman",
        "ant-man",
        "deadpool",
        "guardians of the galaxy",
    ],
    "feel-good": ["feel-good", "feel good", "uplifting", "heartwarming", "hope", "joy", "lighthearted", "warm", "wholesome"],
    "dark": ["dark", "bleak", "gritty", "disturbing", "violent", "tragic", "brutal"],
    "psychological": ["psychological", "manipulation", "obsession", "mind", "paranoia", "delusion"],
    "fast-paced": ["fast-paced", "fast paced", "high energy", "high-energy", "race against time", "chase", "nonstop", "intense"],
}

NEGATIVE_FACET_PATTERNS = {
    "feel-good": ["dark", "bleak", "grim", "disturbing", "brutal", "gore", "tragic", "violent", "horror", "sex", "murder", "serial killer", "crime"],
    "dark": ["lighthearted", "uplifting", "wholesome", "feel-good", "cheerful"],
    "family": ["gore", "graphic violence", "brutal", "disturbing"],
}

CORE_GENRE_FACETS = {
    "action",
    "adventure",
    "science fiction",
    "crime",
    "drama",
    "fantasy",
    "history",
    "romance",
    "horror",
    "music",
    "mystery",
    "thriller",
    "animation",
    "family",
    "comedy",
    "documentary",
    "tv movie",
    "war",
    "western",
}


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", _normalize_text(text))
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _split_csv_field(text: str) -> list[str]:
    return [part.strip() for part in _clean_text(text).split(",") if part.strip()]


for column in TEXT_COLUMNS:
    MOVIES[column] = MOVIES[column].fillna("").astype(str)

MOVIES["combined_text"] = MOVIES.apply(
    lambda row: " | ".join(
        [
            row["title"],
            row["genres"],
            row["overview"],
            row["tagline"],
            row["director"],
            row["top_cast"],
            row["keywords"],
            row["production_countries"],
        ]
    ),
    axis=1,
)
MOVIES["normalized_title"] = MOVIES["title"].map(_normalize_text)
MOVIES["genre_list"] = MOVIES["genres"].map(_split_csv_field)
MOVIES["token_list"] = MOVIES["combined_text"].map(_tokenize)
MOVIES["token_set"] = MOVIES["token_list"].map(set)

TOP_MOVIES = MOVIES.copy()

DOC_FREQ = Counter()
for tokens in MOVIES["token_set"]:
    DOC_FREQ.update(tokens)

DOC_COUNT = max(len(MOVIES), 1)


def _build_vector(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    vector = {}
    for token, tf in counts.items():
        df = DOC_FREQ.get(token, 0)
        idf = math.log((DOC_COUNT + 1) / (df + 1)) + 1.0
        vector[token] = (1.0 + math.log(tf)) * idf
    return vector


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine_similarity(
    left: dict[str, float], right: dict[str, float], left_norm: float, right_norm: float
) -> float:
    if not left_norm or not right_norm:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    return dot / (left_norm * right_norm)


MOVIE_VECTORS = []
MOVIE_VECTOR_NORMS = []
for tokens in MOVIES["token_list"]:
    vector = _build_vector(tokens)
    MOVIE_VECTORS.append(vector)
    MOVIE_VECTOR_NORMS.append(_vector_norm(vector))

MOVIES["vector_index"] = range(len(MOVIES))

SQL_CONN = sqlite3.connect(":memory:", check_same_thread=False)
MOVIES.drop(columns=["genre_list", "token_list", "token_set"], errors="ignore").to_sql(
    "movies", SQL_CONN, index=False, if_exists="replace"
)


def _json_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _normalize_text(_clean_text(item))
        if text:
            result.append(text)
    return result


def _detect_facets(preferences: str) -> list[str]:
    normalized = _normalize_text(preferences)
    detected = []
    for facet, synonyms in FACET_SYNONYMS.items():
        if any(term in normalized for term in synonyms):
            detected.append(facet)
    return detected


def _facet_score(text: str, facet: str) -> float:
    normalized = _normalize_text(text)
    patterns = FACET_PATTERNS.get(facet, [])
    score = 0.0
    for pattern in patterns:
        if pattern in normalized:
            score += 1.0
    return score


def _negative_facet_score(text: str, facet: str) -> float:
    normalized = _normalize_text(text)
    patterns = NEGATIVE_FACET_PATTERNS.get(facet, [])
    score = 0.0
    for pattern in patterns:
        if pattern in normalized:
            score += 1.0
    return score


def _prompt_quality(preferences: str) -> dict[str, object]:
    normalized = _normalize_text(preferences)
    tokens = _tokenize(preferences)
    alpha_chars = sum(char.isalpha() for char in preferences)
    known_tokens = [token for token in tokens if DOC_FREQ.get(token, 0) >= 2]
    recognized_facets = _detect_facets(preferences)
    repeated_noise = bool(re.search(r"(.)\1{4,}", normalized))
    gibberish_ratio = 0.0 if not tokens else 1.0 - (len(known_tokens) / len(tokens))
    is_too_short = len(tokens) < 2 and not recognized_facets
    has_low_signal = not recognized_facets and len(known_tokens) < 2
    mostly_non_alpha = len(preferences) > 0 and alpha_chars / max(len(preferences), 1) < 0.45

    is_unclear = (
        not normalized
        or repeated_noise
        or mostly_non_alpha
        or is_too_short
        or (has_low_signal and gibberish_ratio >= 0.75)
    )
    return {
        "is_unclear": is_unclear,
        "tokens": tokens,
        "known_tokens": known_tokens,
        "recognized_facets": recognized_facets,
    }


def _clarification_description() -> str:
    return (
        "Your request was too unclear to match confidently. Please try a prompt like "
        "'funny feel-good family movie', 'dark psychological thriller', or "
        "'fast-paced sci-fi action movie'."
    )


@lru_cache(maxsize=128)
def _llm_query_plan_cached(preferences: str, history_key: tuple[str, ...]) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not REMOTE_QUERY_PLAN_ENABLED or not api_key:
        return {}

    prompt = f"""You are preparing a movie-search query plan for a local recommender.
Return ONLY valid JSON with this schema:
{{
  "include_terms": ["short phrases from the request"],
  "exclude_terms": ["things to avoid"],
  "genres": ["genre names"],
  "people": ["director or actor names"],
  "moods": ["tone or vibe terms"],
  "min_year": <integer or null>,
  "max_year": <integer or null>
}}

Examples:
Preference: funny, heartfelt family adventure
Output:
{{
  "include_terms": ["funny", "heartfelt", "family adventure"],
  "exclude_terms": [],
  "genres": ["comedy", "family", "adventure"],
  "people": [],
  "moods": ["heartwarming"],
  "min_year": null,
  "max_year": null
}}

Preference: dark psychological thriller
Output:
{{
  "include_terms": ["dark", "psychological thriller"],
  "exclude_terms": [],
  "genres": ["thriller"],
  "people": [],
  "moods": ["dark", "psychological"],
  "min_year": null,
  "max_year": null
}}

User preferences: {preferences}
Already watched: {", ".join(history) if history else "none"}
Rules:
- Keep lists short and focused.
- Do not include watched titles in include_terms.
- If unspecified, use null for min_year and max_year.
"""

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=2.5,
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        parsed = json.loads(response.message.content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _llm_query_plan(preferences: str, history: list[str]) -> dict:
    return _llm_query_plan_cached(preferences, tuple(history))


def _build_plan(preferences: str, history: list[str]) -> dict:
    raw_plan = _llm_query_plan(preferences, history)
    pref_tokens = _tokenize(preferences)
    pref_phrases = [term for term in re.split(r"[,;/]| and | but ", preferences) if term.strip()]
    include_terms = _json_list(raw_plan.get("include_terms"))
    exclude_terms = _json_list(raw_plan.get("exclude_terms"))
    genres = _json_list(raw_plan.get("genres"))
    people = _json_list(raw_plan.get("people"))
    moods = _json_list(raw_plan.get("moods"))

    for phrase in pref_phrases:
        normalized = _normalize_text(phrase)
        if normalized and normalized not in include_terms:
            include_terms.append(normalized)

    for token in pref_tokens:
        if token not in include_terms:
            include_terms.append(token)

    detected_facets = _detect_facets(preferences)
    for facet in detected_facets:
        if facet not in genres and facet not in {"superhero", "feel-good", "dark", "psychological", "fast-paced"}:
            genres.append(facet)
        for synonym in FACET_SYNONYMS.get(facet, [])[:4]:
            if synonym not in include_terms:
                include_terms.append(synonym)

    min_year = raw_plan.get("min_year")
    max_year = raw_plan.get("max_year")
    min_year = int(min_year) if isinstance(min_year, int) or str(min_year).isdigit() else None
    max_year = int(max_year) if isinstance(max_year, int) or str(max_year).isdigit() else None

    return {
        "include_terms": include_terms[:12],
        "exclude_terms": exclude_terms[:10],
        "genres": genres[:6],
        "people": people[:6],
        "moods": moods[:6],
        "required_facets": detected_facets[:6],
        "min_year": min_year,
        "max_year": max_year,
    }


def _history_id_set(history_ids: list[int]) -> set[int]:
    return {int(item) for item in history_ids if str(item).strip()}


def _history_title_set(history: list[str]) -> set[str]:
    return {_normalize_text(title) for title in history if _clean_text(title)}


def _history_matches_title(normalized_title: str, history_titles: set[str]) -> bool:
    for past_title in history_titles:
        if not past_title:
            continue
        if past_title == normalized_title:
            return True
        if past_title in normalized_title or normalized_title in past_title:
            return True
        if SequenceMatcher(None, past_title, normalized_title).ratio() >= 0.9:
            return True
    return False


def _history_rows(history: list[str], history_ids: list[int]) -> pd.DataFrame:
    rows = MOVIES.iloc[0:0].copy()
    id_set = _history_id_set(history_ids)
    title_set = _history_title_set(history)

    if id_set:
        rows = pd.concat([rows, MOVIES[MOVIES["tmdb_id"].isin(id_set)]], ignore_index=True)
    if title_set:
        title_matches = MOVIES[
            MOVIES["normalized_title"].map(lambda title: _history_matches_title(title, title_set))
        ]
        rows = pd.concat([rows, title_matches], ignore_index=True)

    if rows.empty:
        return rows
    return rows.drop_duplicates(subset=["tmdb_id"]).reset_index(drop=True)


def _sql_candidate_ids(plan: dict, preferences: str, excluded_ids: set[int]) -> set[int]:
    clauses = []
    params: list[object] = []
    searchable_terms = list(plan["genres"]) + list(plan["people"]) + list(plan["include_terms"]) + list(plan["moods"])

    for term in searchable_terms[:12]:
        like_value = f"%{term}%"
        clauses.append(
            "("
            "lower(title) LIKE ? OR lower(genres) LIKE ? OR lower(overview) LIKE ? "
            "OR lower(keywords) LIKE ? OR lower(director) LIKE ? OR lower(top_cast) LIKE ?"
            ")"
        )
        params.extend([like_value] * 6)

    if not clauses:
        for token in _tokenize(preferences)[:6]:
            like_value = f"%{token}%"
            clauses.append("(lower(combined_text) LIKE ?)")
            params.append(like_value)

    query = "SELECT tmdb_id FROM movies"
    where_parts = []
    if clauses:
        where_parts.append("(" + " OR ".join(clauses) + ")")
    if plan["min_year"] is not None:
        where_parts.append("year >= ?")
        params.append(plan["min_year"])
    if plan["max_year"] is not None:
        where_parts.append("year <= ?")
        params.append(plan["max_year"])
    for term in plan["exclude_terms"][:8]:
        like_value = f"%{term}%"
        where_parts.append(
            "lower(combined_text) NOT LIKE ?"
        )
        params.append(like_value)

    if excluded_ids:
        placeholders = ",".join("?" for _ in excluded_ids)
        where_parts.append(f"tmdb_id NOT IN ({placeholders})")
        params.extend(sorted(excluded_ids))

    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)
    query += " ORDER BY vote_count DESC, vote_average DESC LIMIT 250"

    try:
        rows = SQL_CONN.execute(query, params).fetchall()
        return {int(row[0]) for row in rows}
    except Exception:
        return set()


def _score_movies(
    preferences: str,
    plan: dict,
    history: list[str],
    history_ids: list[int],
) -> pd.DataFrame:
    excluded_ids = _history_id_set(history_ids)
    excluded_titles = _history_title_set(history)
    sql_ids = _sql_candidate_ids(plan, preferences, excluded_ids)

    query_terms = list(dict.fromkeys(plan["include_terms"] + plan["genres"] + plan["people"] + plan["moods"]))
    query_text = " ".join(query_terms) or preferences
    query_tokens = _tokenize(query_text)
    query_vector = _build_vector(query_tokens)
    query_norm = _vector_norm(query_vector)
    phrase_terms = [term for term in query_terms if " " in term]
    normalized_preferences = _normalize_text(preferences)

    scored = MOVIES.copy()
    scored = scored[~scored["tmdb_id"].isin(excluded_ids)].copy()
    if excluded_titles:
        scored = scored[
            ~scored["normalized_title"].map(
                lambda title: _history_matches_title(title, excluded_titles)
            )
        ].copy()
    watched_rows = _history_rows(history, history_ids)
    watched_genres = set()
    watched_directors = set()
    watched_tokens = set()
    watched_vectors = []
    if not watched_rows.empty:
        for watched in watched_rows.itertuples():
            watched_genres.update(part.lower() for part in watched.genre_list if part)
            if _clean_text(watched.director):
                watched_directors.add(_normalize_text(watched.director))
            watched_tokens.update(watched.token_set)
            watched_vectors.append(
                (
                    MOVIE_VECTORS[watched.vector_index],
                    MOVIE_VECTOR_NORMS[watched.vector_index],
                )
            )

    semantic_scores = []
    overlap_scores = []
    exact_scores = []
    fuzzy_scores = []
    exclusion_penalties = []
    sql_boosts = []
    facet_scores = []
    missing_facet_penalties = []
    conjunction_boosts = []
    opposite_tone_penalties = []
    history_similarity_penalties = []
    genre_purity_bonuses = []
    genre_dilution_penalties = []
    required_facets = plan.get("required_facets", [])

    for row in scored.itertuples():
        vector_idx = row.vector_index
        row_text = row.combined_text.lower()
        semantic = _cosine_similarity(
            query_vector,
            MOVIE_VECTORS[vector_idx],
            query_norm,
            MOVIE_VECTOR_NORMS[vector_idx],
        )
        token_overlap = len(set(query_tokens) & row.token_set)
        exact = 0.0
        if normalized_preferences and normalized_preferences in row_text:
            exact += 5.0
        for term in phrase_terms:
            if term in row_text:
                exact += 2.5
        for genre in plan["genres"]:
            if genre in row.genres.lower():
                exact += 2.0
        for person in plan["people"]:
            if person in row.director.lower() or person in row.top_cast.lower():
                exact += 2.0
        fuzzy = max(
            SequenceMatcher(None, normalized_preferences, row.normalized_title).ratio(),
            SequenceMatcher(None, normalized_preferences, row.genres.lower()).ratio(),
        )
        penalty = 0.0
        for term in plan["exclude_terms"]:
            if term and term in row_text:
                penalty += 2.0

        history_penalty = 0.0
        if watched_vectors:
            max_history_similarity = max(
                _cosine_similarity(
                    MOVIE_VECTORS[vector_idx],
                    watched_vector,
                    MOVIE_VECTOR_NORMS[vector_idx],
                    watched_norm,
                )
                for watched_vector, watched_norm in watched_vectors
            )
            genre_overlap = len(set(part.lower() for part in row.genre_list if part) & watched_genres)
            token_overlap_with_history = len(row.token_set & watched_tokens)
            history_penalty = max_history_similarity * 4.0 + min(genre_overlap, 3) * 0.5
            if token_overlap_with_history >= 8:
                history_penalty += 1.5
            if _normalize_text(_clean_text(row.director)) in watched_directors:
                history_penalty += 1.0

        matched_facets = 0
        facet_total = 0.0
        opposite_penalty = 0.0
        for facet in required_facets:
            facet_match_score = _facet_score(row_text, facet) + _facet_score(row.genres.lower(), facet) * 1.2
            if facet_match_score > 0:
                matched_facets += 1
                facet_total += min(facet_match_score, 3.0)
            opposite_penalty += _negative_facet_score(row_text, facet)
        missing_penalty = float(max(len(required_facets) - matched_facets, 0))
        conjunction_boost = 4.0 if required_facets and matched_facets == len(required_facets) else 0.0
        genre_purity_bonus = 0.0
        genre_dilution_penalty = 0.0
        requested_core_facets = [facet for facet in required_facets if facet in CORE_GENRE_FACETS]
        row_genres = [part.lower() for part in row.genre_list if part]
        if len(requested_core_facets) == 1:
            requested_genre = requested_core_facets[0]
            other_genres = [genre for genre in row_genres if requested_genre not in genre]
            if row_genres and requested_genre in row_genres[0]:
                genre_purity_bonus += 1.5
            if not other_genres:
                genre_purity_bonus += 2.0
            if any(genre in {"action", "adventure", "comedy"} for genre in other_genres) and requested_genre in {"horror", "thriller"}:
                genre_dilution_penalty += 2.5
            if any(genre in {"horror", "thriller"} for genre in other_genres) and requested_genre in {"family", "feel-good"}:
                genre_dilution_penalty += 2.5
            genre_dilution_penalty += max(len(other_genres) - 1, 0) * 0.35

        semantic_scores.append(semantic)
        overlap_scores.append(float(token_overlap))
        exact_scores.append(exact)
        fuzzy_scores.append(fuzzy)
        exclusion_penalties.append(penalty)
        sql_boosts.append(2.5 if int(row.tmdb_id) in sql_ids else 0.0)
        facet_scores.append(facet_total)
        missing_facet_penalties.append(missing_penalty)
        conjunction_boosts.append(conjunction_boost)
        opposite_tone_penalties.append(opposite_penalty)
        history_similarity_penalties.append(history_penalty)
        genre_purity_bonuses.append(genre_purity_bonus)
        genre_dilution_penalties.append(genre_dilution_penalty)

    scored["semantic_score"] = semantic_scores
    scored["overlap_score"] = overlap_scores
    scored["exact_score"] = exact_scores
    scored["fuzzy_score"] = fuzzy_scores
    scored["sql_boost"] = sql_boosts
    scored["penalty_score"] = exclusion_penalties
    scored["facet_score"] = facet_scores
    scored["missing_facet_penalty"] = missing_facet_penalties
    scored["conjunction_boost"] = conjunction_boosts
    scored["opposite_tone_penalty"] = opposite_tone_penalties
    scored["history_similarity_penalty"] = history_similarity_penalties
    scored["genre_purity_bonus"] = genre_purity_bonuses
    scored["genre_dilution_penalty"] = genre_dilution_penalties
    scored["quality_score"] = (
        pd.to_numeric(scored["vote_average"], errors="coerce").fillna(0) * 0.35
        + pd.to_numeric(scored["vote_count"], errors="coerce").fillna(0).map(lambda x: math.log1p(x)) * 0.25
        + pd.to_numeric(scored["popularity"], errors="coerce").fillna(0).map(lambda x: math.log1p(x)) * 0.10
    )
    scored["final_score"] = (
        scored["semantic_score"] * 8.0
        + scored["overlap_score"] * 1.5
        + scored["exact_score"] * 1.8
        + scored["fuzzy_score"] * 2.0
        + scored["facet_score"] * 3.5
        + scored["conjunction_boost"]
        + scored["genre_purity_bonus"]
        + scored["sql_boost"]
        + scored["quality_score"]
        - scored["penalty_score"] * 2.0
        - scored["missing_facet_penalty"] * 6.0
        - scored["opposite_tone_penalty"] * 3.0
        - scored["history_similarity_penalty"] * 1.8
        - scored["genre_dilution_penalty"] * 2.0
    )
    return scored.sort_values(
        by=["final_score", "vote_average", "vote_count", "popularity"],
        ascending=False,
    )


def _truncate_description(text: str) -> str:
    text = re.sub(r"\s+", " ", _clean_text(text)).strip().strip('"')
    if len(text) <= 500:
        return text
    trimmed = text[:497].rsplit(" ", 1)[0].strip()
    return (trimmed or text[:497]).rstrip(" ,;:.") + "..."


def _truncate_soft(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", _clean_text(text)).strip().strip('"')
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 3].rsplit(" ", 1)[0].strip()
    return (trimmed or text[: limit - 3]).rstrip(" ,;:.") + "..."


def _normalize_preference_phrase(preferences: str) -> str:
    text = _normalize_text(preferences)
    prefixes = [
        "i want ",
        "i want a ",
        "i want an ",
        "i want something ",
        "i like ",
        "i love ",
        "show me ",
        "give me ",
        "looking for ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip(" .,!?:;")
    return text or _clean_text(preferences).strip(" .,!?:;")


def _matched_reasons(row: pd.Series, plan: dict) -> list[str]:
    reasons = []
    row_text = row["combined_text"].lower()
    matched_facets = []
    for facet in plan.get("required_facets", []):
        if _facet_score(row_text, facet) + _facet_score(row["genres"].lower(), facet) > 0:
            matched_facets.append(facet)
    if matched_facets:
        reasons.append("it hits the " + ", ".join(matched_facets[:3]).replace("-", " ") + " notes")

    genres = [genre for genre in row["genre_list"][:3] if genre]
    if genres:
        reasons.append("its " + " / ".join(genres) + " mix")

    if _clean_text(row["director"]):
        reasons.append(f"{row['director']}'s direction")

    cast_members = _split_csv_field(row.get("top_cast", ""))[:2]
    if cast_members:
        reasons.append("a cast led by " + " and ".join(cast_members))
    return reasons[:4]


def _short_overview_hook(overview: str) -> str:
    overview = _clean_text(overview)
    if not overview:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", overview, maxsplit=1)[0]
    return _truncate_soft(first_sentence, 140)


def _fallback_description(row: pd.Series, preferences: str, plan: dict) -> str:
    title = row["title"]
    overview_hook = _short_overview_hook(row["overview"])
    reasons = _matched_reasons(row, plan)
    taste = _normalize_preference_phrase(preferences)
    if reasons:
        hook = f"{title} is a strong pick for {taste}: {reasons[0]}"
        if len(reasons) > 1:
            hook += ", plus " + reasons[1]
        hook += "."
    else:
        hook = f"{title} is a strong pick for {taste}."

    summary = f" {overview_hook}" if overview_hook else ""
    return _truncate_description(_truncate_soft(hook + summary, 340))


@lru_cache(maxsize=256)
def _generate_description_with_gemma(
    preferences: str,
    title: str,
    genres: str,
    overview: str,
    director: str,
    cast_text: str,
    reasons_text: str,
) -> str:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return ""

    prompt = f"""Write one persuasive movie recommendation blurb.

Example 1:
User preference: funny feel-good family movie
Movie title: Inside Out 2
Why it matches: comedy, feel-good, family-friendly, emotionally warm
Good blurb: Inside Out 2 is a great pick if you want something funny and uplifting: it is warm, inventive, and full of big feelings that land in a genuinely crowd-pleasing way. The comedy is sharp, the heart is real, and the adventure keeps it moving without losing its emotional payoff.

Example 2:
User preference: fast-paced sci-fi action movie
Movie title: Mad Max: Fury Road
Why it matches: relentless pace, action, science fiction world-building
Good blurb: Mad Max: Fury Road is pure momentum if you want high-energy sci-fi action. It throws you straight into a brutal, visually wild chase and never really lets up, so the movie feels urgent, cinematic, and incredibly easy to get swept up in.

User preference: {preferences}
Movie title: {title}
Genres: {genres}
Director: {director or "Unknown"}
Top cast: {cast_text or "Unknown"}
Why it matches: {reasons_text or "Strong overall match"}
Overview: {overview}

Rules:
- Use plain text only.
- Target 220 to 320 characters, and never exceed 500 characters.
- Make it sound enticing and tailored to the user's taste.
- Mention why this movie fits the user in one or two concrete ways.
- Do not include spoilers.
- Do not use markdown or quotation marks around the whole response.
- Do not retell the full plot.
- Prefer 2 sentences max.
"""

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=1.8,
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        return _truncate_description(response.message.content)
    except Exception:
        return ""


def _build_description(row: pd.Series, preferences: str, plan: dict) -> str:
    fallback = _fallback_description(row, preferences, plan)
    reasons = _matched_reasons(row, plan)
    generated = _generate_description_with_gemma(
        preferences,
        row["title"],
        row["genres"],
        _clean_text(row["overview"])[:350],
        _clean_text(row["director"]),
        ", ".join(_split_csv_field(row.get("top_cast", ""))[:3]),
        ", ".join(reasons),
    )
    return generated or fallback


def _fallback_unseen_movies(history: list[str], history_ids: list[int]) -> pd.DataFrame:
    excluded_ids = _history_id_set(history_ids)
    excluded_titles = _history_title_set(history)
    fallback = MOVIES[~MOVIES["tmdb_id"].isin(excluded_ids)].copy()
    if excluded_titles:
        fallback = fallback[
            ~fallback["normalized_title"].map(
                lambda title: _history_matches_title(title, excluded_titles)
            )
        ].copy()
    return fallback.sort_values(
        by=["vote_average", "vote_count", "popularity"], ascending=False
    )


@lru_cache(maxsize=256)
def _get_recommendation_cached(
    preferences: str, history_key: tuple[str, ...], history_ids_key: tuple[int, ...]
) -> dict:
    preferences = _clean_text(preferences)
    history = list(history_key)
    history_ids = list(history_ids_key)
    quality = _prompt_quality(preferences)
    fallback = _fallback_unseen_movies(history, history_ids)

    if fallback.empty:
        raise ValueError("No unseen movies remain in the candidate list.")

    if quality["is_unclear"]:
        row = fallback.iloc[0]
        return {
            "tmdb_id": int(row["tmdb_id"]),
            "description": _clarification_description(),
        }

    plan = _build_plan(preferences, history)
    ranked = _score_movies(preferences, plan, history, history_ids)

    if ranked.empty:
        row = fallback.iloc[0]
    else:
        row = ranked.iloc[0]

    tmdb_id = int(row["tmdb_id"])
    description = _build_description(row, preferences, plan)
    return {
        "tmdb_id": tmdb_id,
        "description": description,
    }


def get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] = []
) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    return _get_recommendation_cached(
        _clean_text(preferences),
        tuple(history or []),
        tuple(int(item) for item in (history_ids or [])),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    args = parser.parse_args()

    print("Movie recommender - type your preferences and press Enter.")
    print("For watch history, enter comma-separated movie titles (or leave blank).")

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = (
        [title.strip() for title in history_raw.split(",") if title.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")
