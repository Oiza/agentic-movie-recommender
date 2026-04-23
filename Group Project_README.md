# Group Project README

## Project goal

Our goal is to build a movie recommendation agent that takes a user's free-text preferences and watch history, then returns:

- one recommended movie from the provided candidate pool
- a short explanation of why that movie fits

The assignment rewards three things in particular:

- strong recommendation performance
- a clear evaluation strategy
- creativity in how the agent is designed

Our current approach is aimed at all three. We use the required `gemma4:31b-cloud` model, but we do not rely on the model alone to pick the movie. Instead, we combine LLM reasoning with deterministic filtering and scoring over the local movie dataset.

One important note is that this document reflects the current version of the project, including the larger `llm.py` redesign that has been done in the working tree and staged locally. In other words, it describes the recommender as it exists now, not just the original starter template.

## What changed from the starter version

The original starter code was very simple:

- it only looked at a tiny top-movie subset
- it mostly depended on one direct LLM call to choose the answer
- it had much less control over filtering, ranking, and history handling

The current version is much more structured. The main changes so far are:

- expanding from a tiny baseline pool to the full local movie dataset
- replacing direct movie selection by the LLM with code-driven retrieval and ranking
- adding prompt cleaning, tokenization, and feature extraction
- adding explicit facet detection for genres and moods
- adding SQL-based candidate retrieval over multiple metadata fields
- adding vector-style similarity scoring over the movie text
- adding stronger history filtering, including fuzzy title matching and similarity penalties
- adding prompt-quality checks for vague or low-signal requests
- adding a safer explanation pipeline with both LLM-generated and deterministic fallback blurbs
- adding caching to keep repeated queries faster and more stable

This shift is a big part of our group's approach, because it makes the recommender more reliable, easier to tune, and easier to evaluate.

## Our approach

### 1. Start with the local movie database

The file `tmdb_top1000_movies.csv` is loaded into pandas at startup. We clean and normalize the movie metadata, including fields such as:

- title
- genres
- overview
- tagline
- director
- cast
- keywords
- production country

We also build a combined text field so the recommender can compare the user's request against several movie attributes at once.

In the current version, this preprocessing also creates:

- normalized titles for easier history matching
- token lists and token sets for text comparison
- lightweight TF-IDF-style vectors for semantic-style similarity
- an in-memory SQLite table for fast candidate lookup

### 2. Turn the user request into a search plan

The recommender first analyzes the user's prompt and tries to identify the main things they want, such as:

- genres like comedy, thriller, action, or family
- tone or mood like feel-good, dark, psychological, or fast-paced
- possible people names
- anything the user may want to avoid

There are two layers here:

- A rule-based layer detects important facets directly from the text using curated synonym lists.
- An optional LLM query-planning layer can turn the prompt into a more structured plan with include terms, exclude terms, genres, moods, and year limits.

This hybrid setup makes the system more stable than a pure prompt-only recommender.

The changes also added a prompt-quality check. If a request is too unclear, too short, or mostly noise, the system avoids pretending it understands perfectly and returns a safer fallback recommendation with a clarification-style description.

### 3. Retrieve a candidate set

We use lightweight SQL filtering over an in-memory SQLite table built from the dataset. This helps narrow the movie pool to titles that match the request terms in fields like:

- title
- overview
- genres
- keywords
- director
- cast

At the same time, the recommender removes movies that appear in the user's watch history.

The current version also goes beyond exact history removal. It tries to match watched titles more carefully, including close title matches, so the agent is less likely to recommend something the user has effectively already seen.

### 4. Rank movies with a weighted scoring system

After retrieval, the system scores candidate movies using several signals instead of one single rule. The ranking combines:

- semantic similarity between the request and each movie's metadata
- token overlap and phrase matches
- genre and mood matches
- boosts for movies surfaced by SQL retrieval
- penalties for unwanted terms
- penalties for titles that are too similar to what the user already watched
- bonuses for strong genre fit
- quality signals such as rating, popularity, and vote count

This lets the recommender balance relevance and movie quality while still respecting the user's history.

Some of the more recent ranking changes include:

- stronger weighting for exact phrase and facet matches
- penalties when required tones are missing
- penalties when a movie carries the opposite tone of the request
- penalties for movies that are too similar to past watches
- genre purity bonuses when the request points strongly to one genre
- genre dilution penalties when extra genres make the fit weaker

### 5. Generate the explanation separately

Once the top movie is chosen, the system creates a short recommendation blurb. We intentionally separate movie selection from explanation writing:

- the ranking logic picks the movie
- the LLM writes a short human-friendly pitch
- if the LLM is unavailable, the code falls back to a deterministic description

This design reduces the risk of invalid outputs while still keeping the final response natural and personalized.

In the current version, the explanation step is also cached and protected with a deterministic fallback, so even if the LLM response is slow or unavailable, the system can still return a usable recommendation description.

## Why we chose this design

We moved away from a simple "ask the LLM for a movie" strategy because it can be inconsistent. A pure LLM approach may:

- suggest movies outside the allowed candidate pool
- ignore watch history
- return unstable outputs across runs

Our current design is more reliable because the code keeps final control over:

- which movies are eligible
- how the ranking works
- how history is enforced
- how the response format is validated

In short, we use the LLM where it adds value, but we keep the important decision logic in code.

## Evaluation strategy

Our README needs to explain not just what the agent does, but how we judge whether it is improving.

### Baseline idea

Our practical baseline is a simpler recommender that would mostly rely on broad keyword or genre matching. The current system is meant to outperform that baseline by adding:

- richer prompt understanding
- retrieval over multiple metadata fields
- history-aware filtering
- weighted ranking instead of single-rule matching
- better explanation generation

More specifically, the current version is clearly beyond the starter baseline because it now includes:

- full-dataset retrieval instead of a very small candidate shortlist
- structured ranking rather than direct LLM selection
- explicit handling of vague prompts
- fuzzy history matching and similarity-aware penalties
- more controlled explanation generation

### What we evaluate

We evaluate the recommender on four main dimensions:

1. Relevance: does the recommended movie actually fit the user's stated preferences?
2. Novelty: does it avoid recommending something already in the user's watch history?
3. Robustness: does it still behave reasonably when prompts are vague, short, or noisy?
4. Format correctness: does it always return a valid `tmdb_id` and short description in the expected format?

### How we evaluate

We use a mix of automated checks and manual prompt-based review.

Automated evaluation:

- run `python test.py` to verify the output schema
- confirm that the returned `tmdb_id` is inside the allowed candidate list
- confirm the recommendation is not already in the watch history
- confirm runtime stays within the required time limit
- confirm imports are properly listed in `requirements.txt`
- re-run tests after each change to make sure tuning does not break correctness

Manual evaluation:

- try prompts across different genres and tones
- compare runs with and without watch history
- test ambiguous prompts to see whether the fallback behavior is sensible
- inspect whether the explanation clearly connects the recommendation back to the user's request

Examples of prompt categories we would use:

- "funny feel-good movie"
- "dark psychological thriller"
- "fast-paced sci-fi action movie"
- "family movie for kids"
- vague or low-signal prompts to test graceful fallback

### What improvement looks like

We consider the system better when it:

- makes more obviously relevant recommendations
- avoids repetitive or history-adjacent picks
- handles different prompt styles more consistently
- keeps passing all required automated checks


## Creativity in our solution

To make the agent more original and not just a basic prompt wrapper, we added a few creative elements:

- a hybrid architecture instead of a pure LLM-only recommender
- explicit mood and facet detection such as `feel-good`, `dark`, `psychological`, and `fast-paced`
- retrieval across metadata fields rather than title matching only
- history similarity penalties, not just exact history removal
- separate explanation generation with a safe fallback path
- prompt-quality detection for unclear inputs
- an optional LLM query-planning step feeding a deterministic ranker



## Brief code guide

### `llm.py`

This is the main file.

Key responsibilities:

- loads and preprocesses the movie dataset
- builds the prompt/query plan
- retrieves candidate movies with SQL
- scores and ranks movies
- removes watched movies
- handles unclear prompts with a safe fallback path
- generates the final explanation
- caches repeated work to improve speed and consistency
- exposes `get_recommendation(...)`

Important note:

- the model is intentionally kept as `gemma4:31b-cloud`, following the project requirement

### `test.py`

This is the validation script used during development. It checks:

- output structure
- valid `tmdb_id`
- no repeated movie from history
- runtime limit
- package coverage in `requirements.txt`

### `tmdb_top1000_movies.csv`

This is the local movie catalog that the recommender searches and ranks.

### `requirements.txt`

Lists the Python packages needed to run the recommender.

### `README.md`

The original assignment/setup file. It explains the basic project requirements and how to run the code.

## How to run the project

Run the recommender:

```bash
OLLAMA_API_KEY=your_key_here python llm.py
```

Run the test suite:

```bash
OLLAMA_API_KEY=your_key_here python test.py
```

## Current status

So far, the project has moved beyond a simple prompt-only recommender into a more structured hybrid system. The current version:

- uses the required model
- relies on local retrieval and scoring for the actual recommendation
- uses the LLM mainly for planning and explanation support
- includes history-aware filtering
- includes prompt-quality checks and safer fallback behavior
- includes local caching and multi-signal ranking improvements
- includes a clear testable evaluation process

This gives us a solid foundation for further tuning if we want to improve recommendation quality before final submission.
