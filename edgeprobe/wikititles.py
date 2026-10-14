"""Offline existence check against the full list of English Wikipedia titles.

Uses the public dump enwiki-<date>-all-titles-in-ns0.gz (all article and redirect
titles in the main namespace). One streaming pass reports, case-insensitively,
  * which candidate tokens occur as a word in any title,
  * which candidate strings are exactly a title,
  * which (few) candidate phrases occur as a substring of any title.
"""
import gzip
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / ".cache/raw/wikititles/enwiki-20260901-all-titles-in-ns0.gz"
_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def _key(s):
    return s.casefold().replace(" ", "_")


def scan(tokens=(), exact=(), substr=()):
    tokens = {t.casefold() for t in tokens}
    exact = {_key(e) for e in exact}
    substr = [_key(s) for s in substr]
    found_t, found_e, found_s = set(), set(), set()
    with gzip.open(DUMP, "rt", encoding="utf-8", errors="replace") as f:
        next(f)  # header
        for line in f:
            low = line.rstrip("\n").casefold()
            if low in exact:
                found_e.add(low)
            for w in _SPLIT.split(low.replace("_", " ")):
                if w in tokens:
                    found_t.add(w)
            for s in substr:
                if s in low:
                    found_s.add(s)
    unkey = lambda s: s.replace("_", " ")  # noqa: E731
    return found_t, {unkey(e) for e in found_e}, {unkey(s) for s in found_s}
