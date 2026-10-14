"""Existence checks against Wikipedia and Wikidata.

A fabricated entity is accepted only if an exact-phrase full-text search of
English Wikipedia returns zero hits AND Wikidata has no item whose label or
alias matches the string exactly (case-insensitive).
"""
import json
import time
from pathlib import Path

import requests

UA = {"User-Agent": "EdgeProbe/1.0 (non-commercial academic research on LLM hallucination)"}
WP = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "wiki_cache.json"


def _load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


_cache = _load()


def _get(url, params):
    for attempt in range(8):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            wait = float(r.headers.get("Retry-After", 0) or 0)
        except requests.RequestException:
            wait = 0
        time.sleep(max(wait, 20 * (attempt + 1)))
    raise RuntimeError(f"request failed: {url} {params}")


def wikipedia_hits(phrase: str) -> int:
    d = _get(WP, {"action": "query", "list": "search", "srsearch": f'"{phrase}"',
                  "srlimit": 1, "format": "json"})
    return d["query"]["searchinfo"]["totalhits"]


def wikidata_exact(phrase: str) -> bool:
    d = _get(WD, {"action": "wbsearchentities", "search": phrase, "language": "en",
                  "limit": 10, "format": "json"})
    p = phrase.casefold()
    for it in d.get("search", []):
        if it.get("label", "").casefold() == p:
            return True
        if it.get("match", {}).get("text", "").casefold() == p:
            return True
    return False


def exists(phrase: str) -> bool:
    """True if the phrase appears to name something real (or is documented anywhere)."""
    if phrase in _cache:
        return _cache[phrase]
    found = wikipedia_hits(phrase) > 0 or wikidata_exact(phrase)
    _cache[phrase] = found
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(_cache, indent=0, ensure_ascii=False))
    time.sleep(2.5)
    return found
