# Project context: EdgeProbe

Read this first. It explains what the project is, why it is designed this way, what is novel, what has been done, and what is still open. For setup and commands see `README.md`; for the literature see `docs/literature_matrix.md`.

## 1. The question

**Does a language model know what it doesn't know?** When a question lies beyond its knowledge, does a model recognise the edge and decline, or does it confidently make something up (hallucinate)?

A question can lie beyond a model's knowledge in two ways:

| Kind | Example | Correct behaviour |
|---|---|---|
| **No answer exists** (false premise, nonexistent entity) | "What is the name of the third moon of Mars?" | reject the premise / say it does not exist |
| **Answer exists but the model never learned it** (rare, long-tail fact) | "Who was the composer of *The Sanguine Fan*?" (Elgar) | answer if known, otherwise say "I don't know" |

## 2. Research questions and hypotheses (stated before seeing full results)

| | Question | Hypothesis |
|---|---|---|
| RQ1 | Behaviour: on unanswerable questions, do small LLMs refuse or fabricate? | H1: they fabricate on most of them; refusal barely separates unanswerable from answerable questions (boundary index J ≈ 0) |
| RQ2 | Boundary: as entities get rarer and then fictitious, do refusal and stated confidence follow falling accuracy? | H2: accuracy falls, refusal and verbal confidence stay flat |
| RQ3 | Latent self-knowledge: do probability-based signals detect hallucinations better than what the model says? | H3: consistency, token likelihood, P(True), P(IK) beat chance and verbal confidence |
| RQ4 | Intervention: does explicitly allowing "I don't know" help, and at what cost? | H4: fewer hallucinations on unanswerable questions, more over-refusal on answerable ones |

## 3. Design and why

* **Minimal pairs (100).** Each unanswerable question has an answerable twin that changes one entity, attribute or year. *Why:* in prior benchmarks the "fake" and "real" questions also differ in topic and style, so a reviewer can say the model reacted to style. With twins, any difference in behaviour is due to answerability alone. Four categories: NE nonexistent entity (35), FP false premise (35), NC nonexistent category/event (15), TI temporal impossibility (15).
* **Knowledge gradient (280 real + 140 fake).** PopQA questions in five popularity quintiles (Wikipedia page views), plus the *same templates* filled with fabricated entities. *Why:* it turns "the edge of knowledge" into a measurable axis: popular, then rare, then nonexistent. Fabricated names are kept only if none of their invented words appears in any of the 19.26M English Wikipedia titles.
* **Four outcomes.** Correct / correctly refused / hallucinated / over-refused, as in the original team plan. The **boundary index J = P(refuse | unanswerable) − P(refuse | answerable)** summarises boundary awareness: 1 is perfect, 0 means refusals are uninformative.
* **Seven uncertainty signals.** Mean and min token log-prob, sample disagreement and lexical semantic entropy (K = 5 samples), verbalised confidence (0-100), P(True), P(IK). *Why:* they separate what the model **says** from what its probabilities **reveal**.
* **Two prompts.** Neutral vs abstain-allowed ("…say so instead of guessing"). *Why:* tests the cheapest possible fix and its cost (over-refusal).
* **Five small open models on a CPU laptop.** Qwen2.5 0.5B/1.5B/3B and Llama-3.2 1B/3B, 4-bit, llama.cpp. *Why:* two families and a scaling axis. The setting is realistic for offline and low-cost deployment, and anyone can reproduce it without a GPU or API keys.

## 4. Novelty (how to defend it to reviewers)

1. **Controlled comparison.** First study of knowledge-boundary awareness with *minimal pairs*: unanswerable vs answerable twins of identical form.
2. **One continuum for both kinds of not-knowing.** Fabricated entities sit at the end of the same popularity axis as real rare entities, with the same templates, so the "unknown" and "nonexistent" boundaries are compared directly.
3. **Says vs reveals.** Behavioural refusal and verbal confidence are compared with probability-based signals on the same responses, showing whether models "know more than they say" at small scale.
4. **Small-model, CPU-only, training-free.** Prior work mostly uses 7B+ models, GPUs, APIs or fine-tuning. Everything here runs on an 8 GB laptop and is fully released.

What is *not* claimed: a new detection method, fine-tuning, or results on large or proprietary models.

## 5. Status (update as you go)

- [x] Dataset built and verified (`data/edgeprobe.jsonl`, 620 questions)
- [x] Pipeline, grader (16 regression tests), analysis with bootstrap CIs
- [x] Paper skeleton: intro, related work (55 verified references), benchmark, setup, threats to validity
- [ ] Model runs (about 15 h): `results/run_all.log`
- [ ] Results, discussion, abstract and conclusion written from the final numbers
- [ ] **Manual grader audit by two authors** (`scripts/annotate.py`), required before submission
- [ ] Author names and affiliations; final read-through; check MoSICom 2026 CFP (page limit, template)

## 6. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-13 | Use llama.cpp built natively (no BLAS) | 1.7× faster prompt processing than the prebuilt binary on this CPU |
| 2026-10-13 | Sequential (not batched) sampling | Batched `n=5` gave no speed-up on the 2-core CPU |
| 2026-10-13 | Verify fake names offline with the Wikipedia title dump | Wikimedia API rate-limited this IP (HTTP 429) |
| 2026-10-13 | Exclude single-word PopQA titles for work relations | Ambiguous subjects ("Hotel", "Wanted") make grading unfair |
| 2026-10-13 | Grader: any "did/does/do not + verb" counts as premise rejection | Consistent handling of false denials found in the 0.5B outputs |
| 2026-10-13 | Replaced fake name "Marisole" with "Quessaria" | "Marisole" returned Wikipedia hits |

## 7. Writing and integrity rules

* Every number in the paper comes from `paper/numbers.tex` / `paper/tables/`, which `scripts/analyze.py` generates. Never type a result by hand.
* Every reference in `paper/refs.bib` was checked against Crossref, ACL Anthology, PMLR or NeurIPS proceedings. Do not add a reference you have not opened.
* Text is original. Run `scripts/check_originality.py` after edits; it reports any long word sequences shared with the abstracts of the cited papers.
* AI use is disclosed in the Acknowledgment, as IEEE policy requires. Authors must review and take responsibility for all content.

## 8. Glossary

**Hallucination**: a fluent answer that is false or unsupported. **Over-refusal**: declining a question the model could answer. **Calibration / ECE**: whether stated confidence matches observed accuracy. **AUROC**: probability that a signal ranks a random hallucination as more uncertain than a random correct answer (0.5 = chance). **P(True)**: model's probability that its own answer is true. **P(IK)**: model's probability that it can answer the question ("I know"). **Boundary index J**: refusal rate on unanswerable minus refusal rate on answerable questions.
