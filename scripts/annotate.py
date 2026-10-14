"""Terminal tool for manually labelling the grader-audit sample.

Each annotator labels results/audit/audit_sample.csv independently (the automatic label is
hidden), then scripts/analyze.py reports accuracy and Cohen's kappa of the grader.

Usage:
  python scripts/annotate.py --annotator A      # writes column human_label (primary)
  python scripts/annotate.py --annotator B      # writes column human_label_B (second annotator)
Progress is saved after every item; re-run to resume.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "results/audit/audit_sample.csv"
KEYS_A = {"1": "CORRECT", "2": "WRONG", "3": "REFUSED"}
KEYS_U = {"1": "REJECTED", "2": "ACCEPTED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="A", choices=["A", "B"])
    col = "human_label" if ap.parse_args().annotator == "A" else "human_label_B"
    df = pd.read_csv(PATH, keep_default_na=False)
    if col not in df:
        df[col] = ""
    todo = df.index[df[col].astype(str).str.strip() == ""]
    print(f"{len(todo)} of {len(df)} items left. Keys shown per item; q = save and quit.\n")
    for n, i in enumerate(todo, 1):
        r = df.loc[i]
        answerable = str(r.answerable) == "True"
        print("=" * 78)
        print(f"[{n}/{len(todo)}]  {'ANSWERABLE' if answerable else 'UNANSWERABLE'} question")
        print(f"Q:    {r.question}")
        if answerable:
            print(f"Gold: {r.gold.replace('|', ' / ')}")
        print(f"A:    {r.answer}")
        keys = KEYS_A if answerable else KEYS_U
        prompt = "  ".join(f"[{k}] {v}" for k, v in keys.items())
        if answerable:
            hint = "CORRECT = gives a gold answer; WRONG = gives a wrong answer; REFUSED = declines/disputes"
        else:
            hint = "REJECTED = declines, doubts, or corrects the premise; ACCEPTED = answers as if valid"
        print(hint)
        while True:
            k = input(prompt + "  > ").strip().lower()
            if k == "q":
                df.to_csv(PATH, index=False)
                return
            if k in keys:
                df.loc[i, col] = keys[k]
                df.to_csv(PATH, index=False)
                break


if __name__ == "__main__":
    main()
