from __future__ import annotations
import json
import math
import os
import pathlib
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Config & Setup

# Define file paths so we can easily locate our data and cache
APP_DIR = pathlib.Path(__file__).parent
RESUMES_PATH = APP_DIR / "resumes.json"
JOBS_PATH = APP_DIR / "job_opportunities.json"
EMBED_CACHE_PATH = APP_DIR / ".embed_cache.json"

# Models to choose from: default (newer) and optional legacy
DEFAULT_MODEL = "text-embedding-3-small"  
LEGACY_MODEL = "text-embedding-ada-002"   # optional legacy

# Load API key from environment variables for security
load_dotenv(".env")
metis_api_key = os.getenv("API_KEY")

# Utility Functions

def load_json(path: pathlib.Path) -> List[dict]:
    # Opens a JSON file and returns its contents as Python objects
    # Shows an error message in the UI if something goes wrong
    """Safe JSON loading with Streamlit error reporting."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return []

def save_json(path: pathlib.Path, data) -> None:
    # Saves Python data back to a JSON file with pretty formatting
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def quick_keywords(text: str, max_k: int = 15) -> List[str]:
    # Simple keyword extractor used when no explicit keywords are given
    """Naive keyword extractor as fallback."""
    import re
    from collections import Counter
    words = re.findall(r"[a-zA-Z][a-zA-Z\-\+\.#]{1,}", text.lower())
    stop = {
        "and","or","the","a","an","to","of","in","on","for","with","at","by","from","as",
        "is","are","was","were","be","been","has","have","had","that","this","it","its",
        "i","we","you","they","he","she","them","our","your","their","my","me","us"
    }
    filtered = [w.strip(".") for w in words if len(w) > 2 and w not in stop]
    common = [w for w, _ in Counter(filtered).most_common(max_k)]
    return common

def euclidean(a: List[float], b: List[float]) -> float:
    # Calculates Euclidean distance (straight-line distance) between two vectors
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def mean_vec(vectors: List[List[float]]) -> List[float]:
    # Averages a list of vectors to get a single "mean" vector
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    return [x / len(vectors) for x in out]

# Embed Cache

def load_embed_cache() -> Dict[str, Dict[str, List[float]]]:
    # Loads previously saved embeddings from the cache file to save API costs
    if EMBED_CACHE_PATH.exists():
        try:
            with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_embed_cache(cache: Dict[str, Dict[str, List[float]]]) -> None:
    # Saves embeddings back to the cache file for future reuse
    try:
        with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def get_embeddings_cached(keywords: List[str], model: str, client: OpenAI) -> Dict[str, List[float]]:
    # Retrieves embeddings for each keyword, using cache when available
    """Embed keywords; uses local JSON cache + API calls."""
    cache = load_embed_cache()
    cache.setdefault(model, {})
    to_query = [kw for kw in keywords if kw not in cache[model]]

    if to_query:
        # Break keywords into batches to reduce API calls
        BATCH = 200
        for i in range(0, len(to_query), BATCH):
            batch = to_query[i : i + BATCH]
            resp = client.embeddings.create(model=model, input=batch)
            for kw, emb in zip(batch, resp.data):
                cache[model][kw] = emb.embedding  # type: ignore

        save_embed_cache(cache)

    return {kw: cache[model][kw] for kw in keywords}

# Data Prep

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[List[dict], List[dict]]:
    # Loads resumes and jobs from JSON files and caches them for performance
    resumes = load_json(RESUMES_PATH)
    jobs = load_json(JOBS_PATH)
    return resumes, jobs

def keywords_for_item(item: dict) -> List[str]:
    # Returns a cleaned keyword list for a resume or job
    # Uses "keywords" field if available, otherwise extracts from text
    if "keywords" in item and isinstance(item["keywords"], list) and item["keywords"]:
        return [str(k).strip() for k in item["keywords"] if str(k).strip()]
    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
        str(item.get("description", "")),
    ])
    return quick_keywords(text)

def vector_mean_for_keywords(keywords: List[str], model: str, client: OpenAI) -> List[float]:
    # Converts all keywords into embeddings and averages them into a single vector
    if not keywords:
        return []
    embeds = get_embeddings_cached(keywords, model, client)
    vectors = [embeds[k] for k in keywords if k in embeds]
    return mean_vec(vectors)

# Streamlit UI

st.set_page_config(page_title="Resume-to-Job Matcher", page_icon="🧠", layout="centered")
st.title("🧠 NLP Resume-to-Job Matcher")
st.caption("Embeddings → Keyword Mean → Euclidean Distance Ranking")

with st.sidebar:
    # Sidebar allows user to input API key and select embedding model
    st.subheader("API Settings")
    manual_key = st.text_input("🔑 API Key (override)", value="", type="password")
    if manual_key:
        metis_api_key = manual_key

    if not metis_api_key:
        st.error("❌ No API key found. Please set API_KEY in .env or enter it here.")
        st.stop()
    else:
        st.success("✅ API key loaded.")

    # Create client once, reuse everywhere
    client = OpenAI(api_key=metis_api_key, base_url="https://api.metisai.ir/openai/v1")

    # Model selector lets user switch between models
    model = st.selectbox(
        "Embedding Model",
        [DEFAULT_MODEL, LEGACY_MODEL],
        index=0,
        help="Prefer 'text-embedding-3-small'."
    )

# Load resumes + jobs
resumes, jobs = load_data()

# Build job list for dropdown menu
job_titles = [f"{j.get('title','(untitled)')} — {j.get('company','')}".strip(" —") for j in jobs]
job_idx = st.selectbox("Select a job to match:", list(range(len(jobs))), format_func=lambda i: job_titles[i])
job = jobs[job_idx]
job_keywords = keywords_for_item(job)

# Show job details so user knows what they're matching against
with st.expander("📄 Job Details", expanded=True):
    st.markdown(f"**Title:** {job.get('title','')}")
    st.markdown(f"**Company:** {job.get('company','')}")
    if job.get("location"):
        st.markdown(f"**Location:** {job['location']}")
    if job.get("description"):
        st.markdown("**Description**")
        st.write(job["description"])
    st.markdown(f"**Keywords**: `{', '.join(job_keywords)}`")

run = st.button("🔍 Find Top 3 Matching Resumes", use_container_width=True)

if run:
    with st.spinner("Embedding & ranking..."):
        # Compute embedding for selected job
        job_vec = vector_mean_for_keywords(job_keywords, model, client)
        if not job_vec:
            st.error("Could not compute job embedding (no keywords?).")
            st.stop()

        rows = []
        # For each resume, compute embedding and calculate distance
        for r in resumes:
            r_keywords = keywords_for_item(r)
            r_vec = vector_mean_for_keywords(r_keywords, model, client)
            if not r_vec:
                continue
            dist = euclidean(job_vec, r_vec)
            rows.append((dist, r))

        # Sort resumes by smallest distance and keep top 3
        rows.sort(key=lambda t: t[0])
        top3 = rows[:3]

    st.subheader("🏆 Top Matches")
    if not top3:
        st.info("No matches found.")
    else:
        for rank, (score, r) in enumerate(top3, start=1):
            with st.container(border=True):
                st.markdown(f"### #{rank}: {r.get('name', r.get('title','Untitled'))}")
                st.markdown(f"**Distance:** `{score:.4f}`  — smaller is better")
                if r.get("summary"):
                    st.markdown("**Summary**")
                    st.write(r["summary"])
                if r.get("experience"):
                    st.markdown("**Experience**")
                    st.write(r["experience"])
                rk = keywords_for_item(r)
                st.markdown(f"**Keywords**: `{', '.join(rk)}`")

st.markdown("---")
st.caption("Made for practice: Edit JSONs to tweak keywords and re-run matching.")
