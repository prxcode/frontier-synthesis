"""Originality pre-check: find word sequences the paper shares with the abstracts of cited works.

Fetches the abstract of every entry in paper/refs.bib (OpenAlex, then Semantic Scholar, then arXiv),
extracts the text of paper/main.pdf, and reports every run of >= N consecutive words (default 8)
that appears in both. Quoted titles of cited papers are expected hits and are listed separately.

This approximates, but does not replace, the iThenticate / CrossCheck screening IEEE uses.

Usage: python scripts/check_originality.py [--n 8]
"""
import argparse
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache/abstracts.json"
UA = {"User-Agent": "EdgeProbe/1.0 (academic originality check)"}


def bib_entries():
    text = (ROOT / "paper/refs.bib").read_text()
    out = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1), m.group(2)
        t = re.search(r"title\s*=\s*\{(.*)\},?\s*\n", body)
        if t:
            title = re.sub(r"[{}]", "", t.group(1)).replace("\\", "").strip()
            out.append((key, title))
    return out


def openalex_abstract(title):
    r = requests.get("https://api.openalex.org/works", headers=UA, timeout=30,
                     params={"search": re.sub(r"[^\w\s]", " ", title), "per_page": 1})
    res = r.json().get("results") or []
    inv = res[0].get("abstract_inverted_index") if res else None
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def fetch_abstract(title):
    try:
        a = openalex_abstract(title)
        if a:
            return a
    except Exception:
        pass
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                         params={"query": title, "fields": "title,abstract"}, headers=UA, timeout=30)
        if r.status_code == 200:
            d = r.json().get("data") or []
            if d and d[0].get("abstract"):
                return d[0]["abstract"]
        elif r.status_code == 429:
            time.sleep(10)
    except requests.RequestException:
        pass
    try:
        q = re.sub(r"[^\w\s]", " ", title)
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"search_query": f'ti:"{q}"', "max_results": 1}, headers=UA, timeout=30)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(r.text).find("a:entry", ns)
        if e is not None:
            return e.find("a:summary", ns).text
    except Exception:
        pass
    return ""


def words(s):
    s = unicodedata.normalize("NFKD", s.replace("-\n", "").replace("\n", " "))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", s)


def shared_runs(a, b, n):
    """All maximal runs of >= n consecutive words of a that also occur in b."""
    grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    runs, i = [], 0
    while i <= len(a) - n:
        if tuple(a[i:i + n]) in grams:
            j = i + n
            while j < len(a) and tuple(a[j - n + 1:j + 1]) in grams:
                j += 1
            runs.append(" ".join(a[i:j]))
            i = j
        else:
            i += 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    n = ap.parse_args().n
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    entries = bib_entries()
    for key, title in entries:
        if not cache.get(key):
            cache[key] = fetch_abstract(title)
            CACHE.write_text(json.dumps(cache, indent=1))
            time.sleep(0.5)
    paper = words(" ".join(p.get_text() for p in pymupdf.open(ROOT / "paper/main.pdf")))
    # drop the reference list: titles there match by construction
    txt = " ".join(paper)
    cut = txt.rfind(" references ")
    paper = txt[:cut].split() if cut > 0 else paper

    found, missing = 0, [k for k, _ in entries if not cache.get(k)]
    titles = {k: " ".join(words(t)) for k, t in entries}
    print(f"Paper body: {len(paper)} words; abstracts available for "
          f"{len(entries) - len(missing)}/{len(entries)} references (n = {n})\n")
    for key, _ in entries:
        if not cache.get(key):
            continue
        for run in shared_runs(paper, words(cache[key]), n):
            kind = "title" if run in titles[key] else "TEXT"
            found += kind == "TEXT"
            print(f"  [{kind}] {key}: \"{run}\"")
    print(f"\n{found} shared non-title passages of >= {n} words.")
    if missing:
        print("No abstract retrieved for:", ", ".join(missing))


if __name__ == "__main__":
    main()
