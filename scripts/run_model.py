"""Run the EdgeProbe protocol for one model and append raw records to results/raw/<tag>.jsonl.

Per question we collect, under the NEUTRAL instruction:
  * a greedy answer with per-token log-probabilities,
  * K stochastic samples (temperature 1.0) for consistency-based uncertainty,
  * a verbalised confidence (0-100) elicited in a follow-up turn,
  * P(True): the model's probability that its own answer is correct,
  * P(IK):   the model's probability that it can answer the question at all;
and, under the ABSTAIN instruction (explicit permission to decline), a greedy answer.

The script is resumable: questions already present in the output file are skipped.

Usage: python scripts/run_model.py --model .cache/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --tag qwen2.5-1.5b
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgeprobe.client import LlamaServer, binary_prob  # noqa: E402

NEUTRAL = "Answer the following question in one short sentence."
ABSTAIN = ("Answer the following question in one short sentence. If the question is based "
           "on a false premise, asks about something that does not exist, or you do not "
           "know the answer, say so instead of guessing.")
CONF_Q = ("How confident are you that your answer is correct? Reply with only a number "
          "from 0 to 100.")
PTRUE_T = ("Question: {q}\nProposed answer: {a}\n\nIs the proposed answer true and factually "
           "correct? Reply with only True or False.")
PIK_T = ("Question: {q}\n\nCan you answer this question correctly with a true fact? "
         "Reply with only Yes or No.")


def user(text):
    return {"role": "user", "content": text}


def run_question(srv, q, k):
    rec = {"qid": q["qid"]}
    t0 = time.time()
    msgs = [user(f"{NEUTRAL}\n\nQuestion: {q['question']}")]

    ans, lp = srv.chat(msgs, max_tokens=64, temperature=0.0, top_logprobs=1)
    rec["answer"] = ans
    rec["token_logprobs"] = [round(t["logprob"], 5) for t in lp]

    rec["samples"] = [srv.chat(msgs, max_tokens=48, temperature=1.0, top_p=0.95,
                               top_k=50, seed=1000 + i)[0] for i in range(k)]

    conf_txt, _ = srv.chat(msgs + [{"role": "assistant", "content": ans}, user(CONF_Q)],
                           max_tokens=8, temperature=0.0)
    m = re.search(r"\d{1,3}(?:\.\d+)?", conf_txt)
    rec["verbal_conf_raw"] = conf_txt
    rec["verbal_conf"] = min(float(m.group()), 100.0) if m else None

    _, lp = srv.chat([user(PTRUE_T.format(q=q["question"], a=ans))], max_tokens=1,
                     temperature=0.0, top_logprobs=20)
    rec["p_true"] = binary_prob(lp[0]["top_logprobs"], {"true"}, {"false"}) if lp else None

    _, lp = srv.chat([user(PIK_T.format(q=q["question"]))], max_tokens=1, temperature=0.0,
                     top_logprobs=20)
    rec["p_ik"] = binary_prob(lp[0]["top_logprobs"], {"yes"}, {"no"}) if lp else None

    rec["answer_abstain"], _ = srv.chat([user(f"{ABSTAIN}\n\nQuestion: {q['question']}")],
                                        max_tokens=64, temperature=0.0)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N questions")
    a = ap.parse_args()

    qs = [json.loads(l) for l in (ROOT / "data/edgeprobe.jsonl").read_text().splitlines()]
    if a.limit:
        qs = qs[: a.limit]
    out = ROOT / "results/raw" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text().splitlines() if l.strip()}
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[{a.tag}] {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    with LlamaServer(ROOT / a.model if not Path(a.model).is_absolute() else a.model,
                     threads=a.threads) as srv, out.open("a") as f:
        t0 = time.time()
        for i, q in enumerate(todo, 1):
            rec = run_question(srv, q, a.k)
            rec["model"] = a.tag
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{a.tag}] {i}/{len(todo)}  {el / i:.1f}s/q  "
                      f"eta {(len(todo) - i) * el / i / 3600:.2f}h", flush=True)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
