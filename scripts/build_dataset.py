"""Build the EdgeProbe evaluation set.

Three components, all written to data/edgeprobe.jsonl:
  1. MP   - 100 hand-written minimal pairs (data/minimal_pairs.jsonl): each
            unanswerable question has an answerable twin with the same form.
  2. POP  - real PopQA questions, stratified over five popularity bins.
  3. FAKE - the same seven PopQA templates instantiated with fabricated
            entities whose invented words occur in no English Wikipedia title
            (offline check against the full title dump, ~19M titles).

Usage: python scripts/build_dataset.py [--per-cell 8] [--fake-per-rel 20]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgeprobe.wikititles import scan  # noqa: E402

RELATIONS = {
    "director": "Who was the director of {}?",
    "author": "Who is the author of {}?",
    "composer": "Who was the composer of {}?",
    "country": "In what country is {}?",
    "place of birth": "In what city was {} born?",
    "capital": "What is the capital of {}?",
    "father": "Who is the father of {}?",
}
# Work-type subjects with a single token ("Hotel", "Wanted") are highly ambiguous.
MULTIWORD_REQUIRED = {"director", "author", "composer", "country"}

FIRST = ["Johan", "Marta", "Ilse", "Tomas", "Aurelio", "Kasimir", "Leontine", "Bertil",
         "Odile", "Radomir", "Signe", "Emeric", "Rosalind", "Dragan", "Mireille", "Anselm",
         "Vesna", "Ottavio", "Henrike", "Lucan", "Agathe", "Ferenc", "Isolde", "Matthias",
         "Brigitta", "Casimir", "Eleni", "Konrad", "Natalia", "Severin"]
SUR_A = ["Hal", "Mor", "Ost", "Ver", "Bran", "Kel", "Tor", "Ves", "Dal", "Ren", "Sav",
         "Lom", "Ard", "Cal", "Fen", "Gar", "Pel", "Quin", "Rav", "Sol", "Tam", "Wen",
         "Zar", "Ingel", "Jor", "Nad", "Brek", "Ulm", "Corv", "Drav"]
SUR_B = ["vikstren", "enhalt", "ovarek", "drecki", "marrow", "wynde", "ellanor", "ardini",
         "oszky", "strøm", "quistel", "beckmar", "fordley", "ancourt", "escuvel", "enkova",
         "idaris", "olvane", "ethmoor", "urvesk"]
PL_A = ["Kel", "Mor", "Ost", "Brav", "Tel", "Var", "Quel", "Dun", "Gral", "Hin", "Lor",
        "Mir", "Pent", "Rost", "Sel", "Thal", "Ven", "Wex", "Yar", "Zov", "Ebr", "Calv"]
PL_B = ["morrow", "havenby", "ovarra", "stadel", "ingrave", "eckmoor", "burnholt",
        "markov", "dalseth", "holtvar", "enzia", "orvika", "achsel", "feldrun", "wicklen",
        "gradny", "tonmere", "arvesk"]
NOUN = ["Lanterns", "Orchard", "Ferryman", "Archive", "Harbour", "Cartographer", "Silence",
        "Winter", "Glassmaker", "Bellringer", "Weaver", "Lighthouse", "Ashes", "Gardens",
        "Clockmaker", "Tides"]
ADJ = ["Quiet", "Last", "Hollow", "Crimson", "Distant", "Forgotten", "Seventh", "Northern",
       "Paper", "Iron", "Long", "Shallow"]
ROYAL = ["Eadric", "Ottokar", "Sviatomir", "Berengar", "Ingolf", "Casimir", "Aldric",
         "Waleran", "Thibault", "Ragnvald"]


def surname(r):
    return r.choice(SUR_A) + r.choice(SUR_B)


def place(r):
    return r.choice(PL_A) + r.choice(PL_B)


def person(r):
    return f"{r.choice(FIRST)} {surname(r)}"


def fake_subject(rel, r):
    if rel == "director":
        return r.choice([lambda: f"The {surname(r)} Affair",
                         lambda: f"Night Train to {place(r)}",
                         lambda: f"The {r.choice(ADJ)} Summer of {place(r)}"])()
    if rel == "author":
        return r.choice([lambda: f"The {r.choice(NOUN)} of {place(r)}",
                         lambda: f"Letters from {place(r)}",
                         lambda: f"{surname(r)}'s Daughter"])()
    if rel == "composer":
        return r.choice([lambda: f"The {place(r)} Suite",
                         lambda: f"Symphony of {place(r)}",
                         lambda: f"The {r.choice(NOUN)} of {place(r)}"])()
    if rel == "country":
        return r.choice([lambda: f"{place(r)}",
                         lambda: f"{place(r)} Bridge",
                         lambda: f"{place(r)} Abbey"])()
    if rel == "place of birth":
        return person(r)
    if rel == "capital":
        return f"{place(r)} {r.choice(['County', 'Province', 'District', 'Region'])}"
    if rel == "father":
        return r.choice([lambda: person(r),
                         lambda: f"{r.choice(ROYAL)} of {place(r)}"])()
    raise ValueError(rel)


TEMPLATE_WORDS = {w.casefold() for w in FIRST + NOUN + ADJ + ROYAL + [
    "The", "Affair", "Night", "Train", "to", "Summer", "of", "Letters", "from", "s",
    "Daughter", "Suite", "Symphony", "Bridge", "Abbey", "County", "Province", "District",
    "Region"]}


def invented_tokens(subj):
    toks = [t for t in re.split(r"[^\w]+", subj) if t]
    return {t.casefold() for t in toks if t.casefold() not in TEMPLATE_WORDS}


def load_pairs():
    rows = []
    for line in (ROOT / "data" / "minimal_pairs.jsonl").read_text().splitlines():
        p = json.loads(line)
        common = {"source": "MP", "category": p["category"], "pair_id": p["pair_id"],
                  "relation": "", "popularity": None}
        rows.append({**common, "qid": f"MP-{p['pair_id']}-F", "question": p["false_q"],
                     "answerable": False, "answers": [], "note": p["premise"],
                     "fake_term": p["fake_term"]})
        rows.append({**common, "qid": f"MP-{p['pair_id']}-T", "question": p["true_q"],
                     "answerable": True, "answers": p["answers"], "note": "",
                     "fake_term": ""})
    return rows


def sample_popqa(per_cell, seed):
    d = pd.read_csv(ROOT / ".cache/raw/popqa/test.tsv", sep="\t", keep_default_na=False)
    d = d[d.prop.isin(RELATIONS)].copy()
    multi = d.subj.str.split().str.len() >= 2
    d = d[~d.prop.isin(MULTIWORD_REQUIRED) | multi]
    d["logpop"] = np.log10(d.s_pop.astype(float).clip(lower=1))
    edges = np.quantile(d.logpop, [0, .2, .4, .6, .8, 1.0])
    d["pop_bin"] = np.clip(np.searchsorted(edges, d.logpop, side="right") - 1, 0, 4)
    rng = np.random.RandomState(seed)
    picked = []
    for b in range(5):
        cell = d[d.pop_bin == b]
        quota = per_cell * len(RELATIONS)
        take = []
        for rel in RELATIONS:
            sub = cell[cell.prop == rel]
            take.append(sub.sample(min(per_cell, len(sub)), random_state=rng))
        take = pd.concat(take)
        if len(take) < quota:  # top up thin cells from the remaining items of this bin
            rest = cell.drop(take.index)
            take = pd.concat([take, rest.sample(quota - len(take), random_state=rng)])
        picked.append(take)
    out = pd.concat(picked)
    rows = []
    for _, r in out.iterrows():
        rows.append({"source": "POP", "category": "REAL", "pair_id": "",
                     "relation": r.prop, "popularity": int(r.s_pop),
                     "pop_bin": int(r.pop_bin), "qid": f"POP-{r.id}",
                     "question": r.question, "answerable": True,
                     "answers": json.loads(r.possible_answers), "note": r.subj,
                     "fake_term": ""})
    return rows, edges


def make_fakes(per_rel, seed, pool_factor=4):
    r = random.Random(seed)
    pool, seen = {rel: [] for rel in RELATIONS}, set()
    for rel in RELATIONS:
        while len(pool[rel]) < per_rel * pool_factor:
            subj = fake_subject(rel, r)
            if subj not in seen:
                seen.add(subj)
                pool[rel].append(subj)
    toks = set().union(*(invented_tokens(x) for x in seen))
    print(f"scanning Wikipedia titles for {len(toks)} invented tokens / {len(seen)} subjects")
    hit_t, hit_e, _ = scan(tokens=toks, exact=seen)
    rows, rejected = [], 0
    for rel, tmpl in RELATIONS.items():
        n = 0
        for subj in pool[rel]:
            if n == per_rel:
                break
            if invented_tokens(subj) & hit_t or subj.casefold().replace(" ", "_") in hit_e:
                rejected += 1
                continue
            n += 1
            rows.append({"source": "FAKE", "category": "FAKE", "pair_id": "",
                         "relation": rel, "popularity": 0, "pop_bin": -1,
                         "qid": f"FAKE-{rel.replace(' ', '_')}-{n:02d}",
                         "question": tmpl.format(subj), "answerable": False,
                         "answers": [], "note": subj,
                         "fake_term": " ".join(sorted(invented_tokens(subj)))})
        print(f"{rel}: {n} fabricated subjects verified")
    print(f"rejected {rejected} candidates whose invented tokens occur in Wikipedia titles")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--fake-per-rel", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    pairs = load_pairs()
    terms = sorted({p["fake_term"] for p in pairs if p["fake_term"]})
    _, exact, sub = scan(exact=terms, substr=terms)
    bad = sorted(exact | sub)
    if bad:
        print("NOTE: hand-written fake terms occurring inside Wikipedia titles:", bad)
    pop, edges = sample_popqa(a.per_cell, a.seed)
    fakes = make_fakes(a.fake_per_rel, a.seed)

    rows = pairs + pop + fakes
    out = ROOT / "data" / "edgeprobe.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"pop_bin_edges_log10": [round(float(e), 3) for e in edges],
            "counts": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
            "flagged_hand_written_terms": bad, "seed": a.seed}
    (ROOT / "data" / "edgeprobe_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
