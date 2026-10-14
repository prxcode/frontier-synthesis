# EdgeProbe data

| File | Content |
|---|---|
| `minimal_pairs.jsonl` | 100 hand-written pairs: an unanswerable question (`false_q`) and its answerable twin (`true_q`, gold `answers`). Categories: `NE` nonexistent entity (35), `FP` false premise about a real entity (35), `NC` nonexistent category/event (15), `TI` temporal impossibility (15). `premise` states why `false_q` has no answer; `fake_term` is the invented name (NE only). |
| `edgeprobe.jsonl` | The full evaluation set (620 questions) produced by `scripts/build_dataset.py`: the 200 minimal-pair questions, 280 PopQA questions, and 140 fabricated-entity questions. |
| `edgeprobe_meta.json` | Popularity-bin edges, counts and random seed. |

## Fields of `edgeprobe.jsonl`

`qid` unique id · `source` MP / POP / FAKE · `category` NE / FP / NC / TI / REAL / FAKE · `pair_id` (MP only) · `relation` PopQA relation · `popularity` monthly Wikipedia page views (0 for fabricated) · `pop_bin` popularity quintile 0 (rare) to 4 (popular), -1 for fabricated · `question` · `answerable` · `answers` gold aliases · `note` premise explanation or subject · `fake_term` invented word(s).

## How the parts were built

* **Minimal pairs.** Written by hand. Each invented name returned zero hits in an exact-phrase search of English Wikipedia and no exact label match on Wikidata (checked Oct. 2026); one name ("Marisole") had hits and was replaced. None of the names occurs inside any English Wikipedia title (dump of 2026-09-01).
* **PopQA sample.** From the PopQA test set (Mallen et al., ACL 2023), downloaded from Hugging Face `akariasai/PopQA`. The dataset card states no license; check the original repository (github.com/AlexTMallen/adaptive-retrieval) before redistributing the sampled questions. Seven relations (author, capital, composer, country, director, father, place of birth). Single-word subjects of work-type relations are excluded as ambiguous. 8 questions per relation × popularity quintile of log10 page views, seed 13.
* **Fabricated entities.** The same seven templates filled with syllable-composed names. A candidate is kept only if none of its invented words occurs as a word in any of the 19,264,252 article/redirect titles of the English Wikipedia dump `enwiki-20260901-all-titles-in-ns0.gz`.

## Known limitations

* PopQA gold aliases are incomplete for some subjects, which can turn a correct but differently phrased answer into a "wrong" one; the grader audit (`results/audit/`) measures this.
* Absence from Wikipedia titles is strong evidence that an invented name does not denote a notable real entity, but it is not proof that the string never appears anywhere.
* All questions are in English and about Western-centric general knowledge in the minimal-pair part.
