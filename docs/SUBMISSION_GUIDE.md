# From this repository to a submitted MoSICom paper

## Step 1: Finish the experiments (running now, about 15 h)

Progress is in `results/run_all.log`. Keep the laptop plugged in and awake. If anything stops, re-run `caffeinate -i ./scripts/run_all.sh`; it resumes where it left off.

## Step 2: Generate all results

```bash
.venv/bin/python scripts/analyze.py --audit   # once, after ALL five models are done
```
This writes `paper/numbers.tex`, `paper/tables/*.tex`, `paper/figures/*.pdf`, `results/summary/metrics.json`, and freezes the audit sample `results/audit/audit_sample.csv`.

## Step 3: Manual grader audit (two authors, about 1 hour each, required)

```bash
.venv/bin/python scripts/annotate.py --annotator A    # first author
.venv/bin/python scripts/annotate.py --annotator B    # second author, independently
.venv/bin/python scripts/analyze.py                   # fills the audit numbers into the paper
```
Do not look at each other's labels or at the automatic label (the tool hides it). The paper reports the grader's accuracy and Cohen's kappa from these labels.

## Step 4: Build the PDF

**Option A, Overleaf (easiest for a team):**
1. Zip the `paper/` folder (it must contain `main.tex`, `numbers.tex`, `refs.bib`, `sections/`, `tables/`, `figures/`).
2. Overleaf → New Project → Upload Project → choose the zip.
3. Menu → Compiler: **pdfLaTeX** → Recompile. All eight authors can then edit together.

**Option B, local:** `cd paper && tectonic -X compile main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

## Step 5: Edit, as authors

1. Replace the placeholder author block in `main.tex` (names, department, institute, city, e-mails). Order the authors as the team agrees.
2. Search the PDF for red **[TODO]** marks and resolve every one.
3. **Rewrite the text in your own words where you can.** You must be able to explain every sentence at the presentation, and it makes the paper genuinely yours. Keep the numbers (they come from `numbers.tex`) and the citations.
4. After editing, check originality again: `.venv/bin/python scripts/check_originality.py`.
5. Read `CONTEXT.md` §4 (novelty) so that everyone can answer "what is new here?".

## Step 6: Check against the call for papers

* **Page limit.** IEEE conference papers are usually 6 pages including references (extra pages sometimes allowed for a fee); check the MoSICom 2026 CFP. To shorten, move Table IV or Fig. 5 to the released repository first.
* **Template.** The paper already uses the official IEEE conference template (`IEEEtran`, A4/Letter two-column).
* **Blind review?** If the CFP asks for anonymous submission, replace the author block with "Anonymous Authors" and remove the GitHub link.
* **Track.** Submit under *Computational Intelligence* (machine learning, deep learning) or the closest AI track.

## Step 7: Integrity checks before you press submit

* **Plagiarism.** IEEE screens every submission with iThenticate (CrossCheck). Your college library may give you access to run it yourself first. `check_originality.py` already compares the text against the abstracts of all 55 cited papers.
* **AI use.** IEEE policy requires disclosure of AI-generated text, code or figures in the Acknowledgment, and AI tools cannot be authors. The Acknowledgment already contains a disclosure. Adjust it to describe honestly how you used and checked the AI-assisted material; do not delete it. Disclosing protects you: AI-detector scores are unreliable, but undisclosed use discovered later can lead to retraction.
* **Every number** must match `results/summary/metrics.json`; **every reference** must be one you can open.

## Step 8: Submit

1. Create an account on the submission system named in the CFP (MoSICom has used EasyChair).
2. Enter the title, abstract, keywords and all authors with e-mails, choose the track, and upload the PDF.
3. Keep the confirmation e-mail.

## Step 9: If accepted (camera-ready)

1. Address the reviewers' comments; keep a short response list.
2. Create the final PDF with **IEEE PDF eXpress** using the conference ID given by the organisers. It checks fonts and format.
3. Sign the IEEE copyright form (eCF) linked by the organisers.
4. Register and pay the fee (your college covers it), then prepare slides. `paper/figures/` and Fig. 1 are ready to reuse.

## Who does what (8 students)

| Person(s) | Task |
|---|---|
| 2 | Manual grader audit (Step 3), independently |
| 2 | Review the 100 minimal pairs and the fake questions for factual errors; keep a list |
| 1 | Overleaf setup, author block, page-limit check, final formatting (Steps 4, 6) |
| 1 | Figures and tables: read every one against `metrics.json` |
| 1 | References: open every cited paper, confirm it says what we claim (`docs/literature_matrix.md`) |
| 1 | Submission, PDF eXpress, copyright, slides (Steps 8, 9) |
