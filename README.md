# EdgeProbe: Do small language models know what they don't know?

Code, data and LaTeX source for the paper **"Knowing the Edge: Do Small Language Models Recognize the Limits of Their Own Knowledge?"** (IEEE conference format, prepared for MoSICom).

The study asks whether an LLM can recognise the edge of its own knowledge, or whether it confidently makes things up. It combines:

1. **100 minimal pairs**: an unanswerable question (nonexistent entity, false premise, nonexistent category, temporal impossibility) next to an almost identical answerable twin;
2. a **knowledge gradient**: 280 real PopQA questions in five popularity quintiles, plus 140 questions about fabricated entities verified absent from all English Wikipedia titles;
3. **five open-weight models** (Qwen2.5 0.5B/1.5B/3B, Llama-3.2 1B/3B), **two prompts** (neutral, abstain-allowed), and **seven uncertainty signals** (token log-probs, sampling consistency, verbal confidence, P(True), P(IK)).

Every response is graded as *correct*, *correctly refused*, *hallucinated* or *over-refused*.

## Start here

* `CONTEXT.md`: the project in one page (question, hypotheses, design rationale, novelty, status, decisions)
* `docs/literature_matrix.md`: all 55 references with key findings and how we differ
* `docs/SUBMISSION_GUIDE.md`: step by step from results to a submitted (and camera-ready) paper, plus who does what

## Layout

```
data/            minimal_pairs.jsonl (hand-written), edgeprobe.jsonl (full set), datasheet
edgeprobe/       client.py (llama.cpp server), grading.py (grader + consistency),
                 wiki.py / wikititles.py (existence checks for fabricated names)
scripts/         build_dataset.py  run_model.py  run_all.sh  analyze.py
                 annotate.py (manual audit)  check_originality.py  render_pages.py
tests/           test_grading.py
docs/            literature_matrix.md, SUBMISSION_GUIDE.md
results/raw/     one JSONL per model: answers, samples, log-probs, confidences
results/graded/  all.csv: every response with its label and signals
results/summary/ metrics.json: every number in the paper, with 95% bootstrap CIs
results/audit/   audit_sample.csv: stratified sample for the manual grader check
paper/           main.tex, sections/, tables/ and figures/ (generated), refs.bib
```

## Reproduce

Tested on macOS 13 (Intel i5, 8 GB RAM, no GPU) with Python 3.14. Any machine that runs llama.cpp works; a GPU just makes it faster.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. llama.cpp (build b11577) - prebuilt binaries also work: set LLAMA_SERVER=/path/to/llama-server
git clone --depth 1 --branch b11577 https://github.com/ggml-org/llama.cpp .cache/src/llama.cpp-src
.venv/bin/pip install cmake ninja
cmake -S .cache/src/llama.cpp-src -B .cache/src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build .cache/src/build --target llama-server

# 2. models (Q4_K_M GGUF) into .cache/models/
#    Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct-GGUF : qwen2.5-*-instruct-q4_k_m.gguf
#    bartowski/Llama-3.2-{1B,3B}-Instruct-GGUF  : Llama-3.2-*-Instruct-Q4_K_M.gguf
#    (SHA-256 sums of the files used are in results/model_checksums.txt)

# 3. data (needs .cache/raw/popqa/test.tsv from HF akariasai/PopQA and the
#    enwiki-20260901-all-titles-in-ns0.gz dump in .cache/raw/wikititles/)
.venv/bin/python scripts/build_dataset.py

# 4. experiments (resumable; about 15 h on the laptop above)
caffeinate -i ./scripts/run_all.sh        # on Linux: ./scripts/run_all.sh

# 5. grading, statistics, tables and figures
.venv/bin/python scripts/analyze.py

# 6. paper (any LaTeX: Overleaf, pdflatex, or tectonic)
cd paper && tectonic -X compile main.tex
```

`analyze.py` writes `paper/numbers.tex`, `paper/tables/*.tex` and `paper/figures/*.pdf`, so the numbers in the PDF always match the data.

## Before submission (must be done by the authors)

1. **Manual grader audit.** After all runs finish, run `python scripts/analyze.py --audit` once to freeze the stratified sample. Then two authors label it independently:
   `python scripts/annotate.py --annotator A` and `python scripts/annotate.py --annotator B` (about 1 hour each).
   Re-run `analyze.py`: the paper's audit numbers (grader accuracy, Cohen's kappa, inter-annotator kappa) fill in automatically.
2. Replace the placeholder author block in `paper/main.tex` and fix any remaining red **[TODO]** marks.
3. Read the whole paper and confirm every claim against `results/summary/metrics.json`.
4. Check the current MoSICom call for papers for the page limit, template and anonymity rules. The paper uses the standard IEEE conference template (`IEEEtran`, two columns).
5. Keep the AI-use disclosure in the Acknowledgment. IEEE policy requires AI-generated content (text, code, figures) to be disclosed there, and AI systems cannot be listed as authors.

## How the original five-team plan maps onto this repository

| Team | Original task | What it corresponds to here |
|---|---|---|
| Fake questions (2) | write 100-150 questions with false assumptions | `data/minimal_pairs.jsonl` (100 pairs, 4 categories) and the 140 auto-generated, Wikipedia-verified fabricated entities. Review the pairs; extending them is easy. |
| Real questions (1) | pull answerable questions from a public dataset | PopQA sample in `build_dataset.py`, stratified by popularity, plus the 100 answerable twins |
| Main experiment (2) | run models, log, score | `run_model.py`, `run_all.sh`, `grading.py` |
| Analysis (1) | charts | `analyze.py` (all tables and figures) |
| Citations (1) | find and cite related work | `paper/refs.bib`; every DOI was checked against Crossref |
