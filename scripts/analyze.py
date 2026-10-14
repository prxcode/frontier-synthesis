"""Grade raw model outputs and compute every number, table and figure used in the paper.

Outputs
  results/graded/all.csv           one row per (model, question) with labels and features
  results/summary/metrics.json     all headline numbers (with 95% bootstrap CIs)
  paper/tables/*.tex               LaTeX tables included by paper/main.tex
  paper/figures/*.pdf              figures included by paper/main.tex
  results/audit/audit_sample.csv   stratified sample for manual validation of the grader

Usage: python scripts/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgeprobe.grading import consistency_features, grade  # noqa: E402

RNG = np.random.default_rng(7)
B = 2000  # bootstrap replicates

MODELS = [  # (tag, display name, family, params in B)
    ("qwen2.5-0.5b", "Qwen2.5-0.5B", "Qwen", 0.5),
    ("llama3.2-1b", "Llama-3.2-1B", "Llama", 1.2),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B", "Qwen", 1.5),
    ("qwen2.5-3b", "Qwen2.5-3B", "Qwen", 3.1),
    ("llama3.2-3b", "Llama-3.2-3B", "Llama", 3.2),
]
SIGNALS = [  # (column, display, sign) ; uncertainty score = sign * value
    ("mean_lp", "Mean token log-prob", -1),
    ("min_lp", "Min token log-prob", -1),
    ("disagree", "Sample disagreement", +1),
    ("lex_entropy", "Lexical semantic entropy", +1),
    ("verbal_conf", "Verbalized confidence", -1),
    ("p_true", "P(True)", -1),
    ("p_ik", "P(IK)", -1),
]
CAT_NAMES = {"NE": "Nonexistent entity", "FP": "False premise", "NC": "Nonexistent category/event",
             "TI": "Temporal impossibility", "FAKE": "Fabricated PopQA entity"}

OUT_T = ROOT / "paper/tables"
OUT_F = ROOT / "paper/figures"
OUT_S = ROOT / "results/summary"
for d in (OUT_T, OUT_F, OUT_S, ROOT / "results/graded", ROOT / "results/audit"):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- loading
def load():
    qs = {}
    for line in (ROOT / "data/edgeprobe.jsonl").read_text().splitlines():
        q = json.loads(line)
        qs[q["qid"]] = q
    rows = []
    for tag, name, fam, size in MODELS:
        f = ROOT / "results/raw" / f"{tag}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            q = qs.get(r["qid"])
            if q is None:
                continue
            lp = r["token_logprobs"] or [0.0]
            cf = consistency_features(r["answer"], r["samples"], q["question"])
            rows.append({
                "model": tag, "model_name": name, "family": fam, "size": size,
                "qid": q["qid"], "source": q["source"], "category": q["category"],
                "pair_id": q.get("pair_id", ""), "relation": q.get("relation", ""),
                "pop_bin": q.get("pop_bin", np.nan), "popularity": q.get("popularity"),
                "answerable": q["answerable"], "question": q["question"],
                "gold": "|".join(map(str, q["answers"][:6])),
                "answer": r["answer"], "answer_abstain": r["answer_abstain"],
                "label": grade(r["answer"], q["answerable"], q["answers"]),
                "label_abstain": grade(r["answer_abstain"], q["answerable"], q["answers"]),
                "mean_lp": float(np.mean(lp)), "min_lp": float(np.min(lp)),
                "disagree": 1 - cf["agree"], "lex_entropy": cf["lex_entropy"],
                "sample_refuse_frac": cf["sample_refuse_frac"],
                "verbal_conf": r["verbal_conf"], "p_true": r["p_true"], "p_ik": r["p_ik"],
                "seconds": r.get("seconds"),
            })
    df = pd.DataFrame(rows)
    for c in ("label", "label_abstain"):
        s = c.replace("label", "")
        df[f"refused{s}"] = df[c].isin(["REFUSED", "REJECTED"])
        df[f"halluc{s}"] = df[c].isin(["WRONG", "ACCEPTED"])
        df[f"correct{s}"] = df[c].isin(["CORRECT", "REJECTED"])
    return df


def present_models(df):
    have = set(df.model)
    return [m for m in MODELS if m[0] in have]


# ----------------------------------------------------------------------------- stats
def boot_ci(values, fn=np.mean):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(B, len(v)))
    bs = np.array([fn(v[i]) for i in idx])
    return (float(fn(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def auroc_ci(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y)) < 2:
        return (np.nan, np.nan, np.nan, int(ok.sum()))
    est = roc_auc_score(y, s)
    bs = []
    for _ in range(B):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return (float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(y))


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    ok = ~np.isnan(conf)
    conf, correct = conf[ok], correct[ok]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def boundary_j(sub, refused_col):
    """Youden J of 'refusal' as a detector of unanswerability: P(ref|U) - P(ref|A)."""
    a = sub[sub.answerable][refused_col].values.astype(float)
    u = sub[~sub.answerable][refused_col].values.astype(float)
    ia = RNG.integers(0, len(a), size=(B, len(a)))
    iu = RNG.integers(0, len(u), size=(B, len(u)))
    bs = u[iu].mean(1) - a[ia].mean(1)
    return (float(u.mean() - a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(x, ci=None, pct=True, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    k = 100 if pct else 1
    s = f"{k * x:.{d}f}"
    if ci is not None:
        s += f"\\,{{\\scriptsize[{k * ci[0]:.{d}f},\\,{k * ci[1]:.{d}f}]}}"
    return s


# ----------------------------------------------------------------------------- metrics
def outcome_metrics(df):
    out = {}
    for tag, name, *_ in present_models(df):
        sub = df[df.model == tag]
        A, U = sub[sub.answerable], sub[~sub.answerable]
        m = {"n_answerable": int(len(A)), "n_unanswerable": int(len(U))}
        for cond, sfx in (("neutral", ""), ("abstain", "_abstain")):
            lab = f"label{sfx}"
            m[cond] = {
                "acc_A": boot_ci(A[lab] == "CORRECT"),
                "hall_A": boot_ci(A[lab] == "WRONG"),
                "ref_A": boot_ci(A[lab] == "REFUSED"),
                "rej_U": boot_ci(U[lab] == "REJECTED"),
                "hall_U": boot_ci(U[lab] == "ACCEPTED"),
                "J": boundary_j(sub, f"refused{sfx}"),
                "hall_all": boot_ci(sub[f"halluc{sfx}"]),
            }
        # paired test: does the abstain instruction change unanswerable rejection?
        b = int(((U.label == "ACCEPTED") & (U.label_abstain == "REJECTED")).sum())
        c = int(((U.label == "REJECTED") & (U.label_abstain == "ACCEPTED")).sum())
        m["mcnemar_U"] = {"b": b, "c": c, "p": float(stats.binomtest(b, b + c).pvalue) if b + c else 1.0}
        b = int(((A.label != "REFUSED") & (A.label_abstain == "REFUSED")).sum())
        c = int(((A.label == "REFUSED") & (A.label_abstain != "REFUSED")).sum())
        m["mcnemar_A_refusal"] = {"b": b, "c": c, "p": float(stats.binomtest(b, b + c).pvalue) if b + c else 1.0}
        out[tag] = m
    return out


def category_metrics(df):
    out = {}
    U = df[~df.answerable]
    for tag, *_ in present_models(df):
        out[tag] = {}
        for cat in ["NE", "FP", "NC", "TI", "FAKE"]:
            s = U[(U.model == tag) & (U.category == cat)]
            out[tag][cat] = {"n": int(len(s)),
                             "rej_neutral": float((s.label == "REJECTED").mean()) if len(s) else np.nan,
                             "rej_abstain": float((s.label_abstain == "REJECTED").mean()) if len(s) else np.nan}
    return out


def pair_metrics(df):
    """Minimal-pair analysis: does the model behave differently on twin questions?"""
    out = {}
    mp = df[df.source == "MP"]
    for tag, *_ in present_models(df):
        s = mp[mp.model == tag]
        F = s[~s.answerable].set_index("pair_id")
        T = s[s.answerable].set_index("pair_id")
        ids = F.index.intersection(T.index)
        F, T = F.loc[ids], T.loc[ids]
        knows = T.label == "CORRECT"
        res = {"n_pairs": int(len(ids)),
               "both_ok": float(((T.label == "CORRECT") & (F.label == "REJECTED")).mean()),
               "knows_twin": float(knows.mean()),
               "accept_F_given_knows_T": float((F.label[knows] == "ACCEPTED").mean()) if knows.any() else np.nan}
        for sig in ("verbal_conf", "p_true", "p_ik", "disagree"):
            d = (F[sig].astype(float) - T[sig].astype(float)).dropna()
            res[f"delta_{sig}"] = float(d.mean()) if len(d) else np.nan
            res[f"wilcoxon_{sig}_p"] = float(stats.wilcoxon(d).pvalue) if len(d) > 5 and (d != 0).any() else np.nan
        out[tag] = res
    return out


def detection_metrics(df):
    """AUROC of each uncertainty signal for flagging hallucinated (vs correct) answers."""
    out = {}
    for tag, *_ in present_models(df):
        s = df[(df.model == tag) & ~df.refused]
        y = s.halluc.astype(int).values
        out[tag] = {"n": int(len(s)), "base_rate": float(y.mean())}
        for col, _, sign in SIGNALS:
            out[tag][col] = auroc_ci(y, sign * s[col].astype(float).values)
    # pooled across models
    s = df[~df.refused]
    y = s.halluc.astype(int).values
    out["pooled"] = {"n": int(len(s)), "base_rate": float(y.mean())}
    for col, _, sign in SIGNALS:
        out["pooled"][col] = auroc_ci(y, sign * s[col].astype(float).values)
    return out


def unanswerable_detection(df):
    """AUROC of each signal for telling unanswerable from answerable questions (all items)."""
    out = {}
    for tag, *_ in present_models(df):
        s = df[df.model == tag]
        y = (~s.answerable).astype(int).values
        out[tag] = {col: auroc_ci(y, sign * s[col].astype(float).values)[:3] for col, _, sign in SIGNALS}
    return out


def calibration_metrics(df):
    out = {}
    for tag, *_ in present_models(df):
        s = df[(df.model == tag) & ~df.refused]
        corr = s.correct.astype(float).values
        out[tag] = {
            "acc_attempted": float(corr.mean()),
            "verbal_conf_mean": float(np.nanmean(s.verbal_conf.astype(float) / 100)),
            "p_true_mean": float(np.nanmean(s.p_true.astype(float))),
            "ece_verbal": ece(s.verbal_conf.astype(float) / 100, corr),
            "ece_p_true": ece(s.p_true.astype(float), corr),
            "ece_p_ik": ece(s.p_ik.astype(float), corr),
            "frac_conf_ge_90": float(np.nanmean(s.verbal_conf.astype(float) >= 90)),
        }
    return out


def gradient_metrics(df):
    """Behaviour along the knowledge gradient: fabricated -> rare -> popular entities."""
    rows = []
    g = df[df.source.isin(["POP", "FAKE"])].copy()
    g["bin"] = g.pop_bin.fillna(-1).astype(int)
    for (tag, b), s in g.groupby(["model", "bin"]):
        rows.append({"model": tag, "bin": b, "n": len(s),
                     "acc": (s.label == "CORRECT").mean() if b >= 0 else np.nan,
                     "halluc": s.halluc.mean(), "refused": s.refused.mean(),
                     "refused_abstain": s.refused_abstain.mean(),
                     "verbal_conf": s.verbal_conf.astype(float).mean() / 100,
                     "p_ik": s.p_ik.astype(float).mean(), "p_true": s.p_true.astype(float).mean(),
                     "agree": 1 - s.disagree.mean()})
    gm = pd.DataFrame(rows)
    corr = {}
    if gm.empty:
        return gm, corr
    pop = df[df.source == "POP"]
    for tag, *_ in present_models(df):
        s = pop[pop.model == tag]
        if len(s) < 10:
            continue
        lp = np.log10(s.popularity.astype(float))
        corr[tag] = {}
        for col in ("correct", "refused", "verbal_conf", "p_ik", "disagree"):
            v = s[col].astype(float)
            ok = ~v.isna()
            rho, p = stats.spearmanr(lp[ok], v[ok])
            corr[tag][col] = {"rho": float(rho), "p": float(p)}
    return gm, corr


# ----------------------------------------------------------------------------- tables
def table_main(om, models):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Outcome rates (\%) with 95\% bootstrap confidence intervals. Answerable: $n{=}NA$ (100 minimal-pair twins + 280 PopQA); unanswerable: $n{=}NU$ (100 false-premise twins + 140 fabricated PopQA entities). $J$ = P(refuse$\mid$unanswerable) $-$ P(refuse$\mid$answerable); $J{=}0$ means refusals carry no information about answerability.}".replace(
             "NA", str(om[models[0][0]]["n_answerable"])).replace("NU", str(om[models[0][0]]["n_unanswerable"])),
         r"\label{tab:main}", r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize",
         r"\begin{tabular}{ll ccc cc c}", r"\toprule",
         r" & & \multicolumn{3}{c}{Answerable questions} & \multicolumn{2}{c}{Unanswerable questions} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & Prompt & Correct $\uparrow$ & Hallucinated $\downarrow$ & Over-refused $\downarrow$ & Correctly refused $\uparrow$ & Hallucinated $\downarrow$ & $J$ $\uparrow$ \\",
         r"\midrule"]
    for i, (tag, name, *_) in enumerate(models):
        for cond, lab in (("neutral", "Neutral"), ("abstain", "Abstain-allowed")):
            m = om[tag][cond]
            cells = [fmt(m[k][0], m[k][1:]) for k in ("acc_A", "hall_A", "ref_A", "rej_U", "hall_U", "J")]
            L.append(f"{name if cond == 'neutral' else ''} & {lab} & " + " & ".join(cells) + r" \\")
        if i < len(models) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT_T / "main_results.tex").write_text("\n".join(L) + "\n")


def table_categories(cm, models):
    cats = ["NE", "FP", "NC", "TI", "FAKE"]
    n = {c: cm[models[0][0]][c]["n"] for c in cats}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Correct-refusal rate (\%) on unanswerable questions by category, neutral / abstain-allowed prompt.}",
         r"\label{tab:categories}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{l" + "c" * len(cats) + "}", r"\toprule",
         "Model & " + " & ".join(f"{c} ({n[c]})" for c in cats) + r" \\", r"\midrule"]
    for tag, name, *_ in models:
        cells = []
        for c in cats:
            v = cm[tag][c]
            cells.append(f"{100 * v['rej_neutral']:.0f} / {100 * v['rej_abstain']:.0f}")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\multicolumn{%d}{p{0.95\columnwidth}}{\scriptsize NE: nonexistent entity; FP: false premise about a real entity; NC: nonexistent category or event; TI: temporal impossibility; FAKE: fabricated entity in a PopQA template.}\\" % (len(cats) + 1),
          r"\end{tabular}", r"\end{table}"]
    (OUT_T / "categories.tex").write_text("\n".join(L) + "\n")


def table_detection(dm, um, models):
    cols = [s for s in SIGNALS]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Hallucination detection. Left: AUROC for separating hallucinated from correct answers among non-refused responses (neutral prompt). Right: AUROC for separating unanswerable from answerable questions using the same signals. 0.5 = chance; best per row in bold.}",
         r"\label{tab:detection}", r"\setlength{\tabcolsep}{2.4pt}", r"\footnotesize",
         r"\begin{tabular}{l c " + "c" * len(cols) + " c " + "c" * len(cols) + "}", r"\toprule",
         r" & & \multicolumn{%d}{c}{Hallucinated vs.\ correct answer} & & \multicolumn{%d}{c}{Unanswerable vs.\ answerable question}\\" % (len(cols), len(cols)),
         r"\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}" % (2 + len(cols), 4 + len(cols), 3 + 2 * len(cols)),
         r"Model & $n$ & " + " & ".join(_short(c[1]) for c in cols) + " & & " + " & ".join(_short(c[1]) for c in cols) + r" \\",
         r"\midrule"]
    for tag, name, *_ in models + [("pooled", "All models", None, None)]:
        d = dm[tag]
        vals = [d[c[0]][0] for c in cols]
        best = np.nanmax(vals)
        cells = [(r"\textbf{%.2f}" if v == best else "%.2f") % v for v in vals]
        if tag == "pooled":
            ucells = [""] * len(cols)
            L.append(r"\midrule")
        else:
            uv = [um[tag][c[0]][0] for c in cols]
            ub = np.nanmax(uv)
            ucells = [(r"\textbf{%.2f}" if v == ub else "%.2f") % v for v in uv]
        L.append(f"{name} & {d['n']} & " + " & ".join(cells) + " & & " + " & ".join(ucells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT_T / "detection.tex").write_text("\n".join(L) + "\n")


def _short(name):
    return {"Mean token log-prob": "LP$_{\\mu}$", "Min token log-prob": "LP$_{\\min}$",
            "Sample disagreement": "Dis.", "Lexical semantic entropy": "SE$_{\\ell}$",
            "Verbalized confidence": "Verb.", "P(True)": "P(T)", "P(IK)": "P(IK)"}[name]


def table_calibration(cal, pm, models):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Confidence among attempted (non-refused) answers. Acc.: fraction of attempted answers that are correct; Verb.: mean verbalized confidence; ECE: expected calibration error (10 bins). Pairs: share of minimal pairs where the twin is answered correctly \emph{and} the false-premise question is refused.}",
         r"\label{tab:calibration}", r"\setlength{\tabcolsep}{2.5pt}", r"\footnotesize",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & Acc. & Verb. & ECE$_{\text{verb}}$ & ECE$_{\text{P(T)}}$ & Pairs \\", r"\midrule"]
    for tag, name, *_ in models:
        c = cal[tag]
        L.append(f"{name} & {100 * c['acc_attempted']:.1f} & {100 * c['verbal_conf_mean']:.1f} & "
                 f"{c['ece_verbal']:.3f} & {c['ece_p_true']:.3f} & {100 * pm[tag]['both_ok']:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "calibration.tex").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- figures
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OUTCOME = {"ok": "#2a78d6", "halluc": "#eb6834", "refused": "#c3c2b7"}


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _model_style(models):
    sty = {}
    for i, (tag, name, fam, _) in enumerate(models):
        sty[tag] = {"color": PALETTE[i % len(PALETTE)], "marker": "o" if fam == "Qwen" else "s",
                    "label": name}
    return sty


def fig_outcomes(df, models):
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.9), sharey=True)
    names = [m[1] for m in models]
    y = np.arange(len(models))[::-1]
    h = 0.36
    for ax, (cond, title) in zip(axes, (("", "Neutral prompt"), ("_abstain", "Abstain-allowed prompt"))):
        for j, (ans, off, tag) in enumerate(((True, h / 2 + 0.02, "A"), (False, -h / 2 - 0.02, "U"))):
            for k, (tg, *_) in enumerate(models):
                s = df[(df.model == tg) & (df.answerable == ans)]
                lab = s[f"label{cond}"]
                if ans:
                    parts = [(lab == "CORRECT").mean(), (lab == "WRONG").mean(), (lab == "REFUSED").mean()]
                    cols = [OUTCOME["ok"], OUTCOME["halluc"], OUTCOME["refused"]]
                else:
                    parts = [(lab == "REJECTED").mean(), (lab == "ACCEPTED").mean()]
                    cols = [OUTCOME["ok"], OUTCOME["halluc"]]
                left = 0
                for p, c in zip(parts, cols):
                    ax.barh(y[k] + off, p * 100, left=left, height=h, color=c,
                            edgecolor="white", linewidth=0.8)
                    if p >= 0.09:
                        ax.text(left + p * 50, y[k] + off, f"{100 * p:.0f}", ha="center",
                                va="center", fontsize=5.8,
                                color="white" if c != OUTCOME["refused"] else INK)
                    left += p * 100
                ax.text(-1.5, y[k] + off, tag, ha="right", va="center", fontsize=5.8, color=MUTED)
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Share of questions (%)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    from matplotlib.patches import Patch
    handles = [Patch(color=OUTCOME["ok"], label="Correct answer (A) / correct refusal (U)"),
               Patch(color=OUTCOME["halluc"], label="Hallucination"),
               Patch(color=OUTCOME["refused"], label="Over-refusal (A only)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.savefig(OUT_F / "outcomes.pdf")
    plt.close(fig)


def fig_gradient(gm, models):
    import matplotlib.pyplot as plt
    _style()
    sty = _model_style(models)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.85))
    xs = [-1, 0, 1, 2, 3, 4]
    xl = ["Fake", "Q1\n(rare)", "Q2", "Q3", "Q4", "Q5\n(popular)"]
    panels = [("acc", "(a) Accuracy", True), ("refused", "(b) Refusal rate (neutral)", False),
              ("verbal_conf", "(c) Verbalized confidence", False)]
    for ax, (col, title, skip_fake) in zip(axes, panels):
        for tag, *_ in models:
            s = gm[gm.model == tag].set_index("bin").reindex(xs)
            xx = [x for x in xs if not (skip_fake and x == -1)]
            ax.plot(xx, s.loc[xx, col] * 100, color=sty[tag]["color"], marker=sty[tag]["marker"],
                    markersize=3.2, linewidth=1.2, label=sty[tag]["label"],
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axvspan(-1.4, -0.6, color="#f0efec", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(xl)
        ax.set_xlim(-1.4, 4.4)
        ax.set_ylim(0, 102)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("%")
    axes[0].set_xlabel("Entity popularity quintile", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT_F / "gradient.pdf")
    plt.close(fig)


def fig_signals_gradient(gm, models):
    """Internal signals along the knowledge gradient (P(IK), agreement)."""
    import matplotlib.pyplot as plt
    _style()
    sty = _model_style(models)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7), sharey=True)
    xs = [-1, 0, 1, 2, 3, 4]
    xl = ["Fake", "Q1", "Q2", "Q3", "Q4", "Q5"]
    for ax, (col, title) in zip(axes, (("p_ik", "(a) P(IK)"), ("agree", "(b) Sample agreement"))):
        for tag, *_ in models:
            s = gm[gm.model == tag].set_index("bin").reindex(xs)
            ax.plot(xs, s[col] * 100, color=sty[tag]["color"], marker=sty[tag]["marker"],
                    markersize=3, linewidth=1.1, label=sty[tag]["label"],
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axvspan(-1.4, -0.6, color="#f0efec", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(xl)
        ax.set_xlim(-1.4, 4.4)
        ax.set_ylim(0, 102)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean value (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT_F / "signals_gradient.pdf")
    plt.close(fig)


def fig_auroc_heatmap(dm, models):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    ramp = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq", ramp)
    rows = models + [("pooled", "All models", None, None)]
    M = np.array([[dm[t][c][0] for c, _, _ in SIGNALS] for t, *_ in rows])
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=5.8, color="white" if v >= 0.72 else INK)
    ax.set_xticks(range(len(SIGNALS)))
    ax.set_xticklabels([_short(s[1]).replace("$", "").replace("_{\\mu}", "μ").replace("_{\\min}", "min")
                        .replace("_{\\ell}", "ℓ") for s in SIGNALS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(len(rows) - 1.5, color="white", linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_visible(False)
    cb.set_label("AUROC", fontsize=6)
    fig.savefig(OUT_F / "auroc_heatmap.pdf")
    plt.close(fig)


def fig_tradeoff(om, models):
    import matplotlib.pyplot as plt
    _style()
    sty = _model_style(models)
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    for tag, name, *_ in models:
        n, a = om[tag]["neutral"], om[tag]["abstain"]
        x0, y0 = 100 * n["ref_A"][0], 100 * n["hall_U"][0]
        x1, y1 = 100 * a["ref_A"][0], 100 * a["hall_U"][0]
        c = sty[tag]["color"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, mutation_scale=7,
                                    shrinkA=2, shrinkB=2))
        ax.plot([x0], [y0], marker=sty[tag]["marker"], color=c, markersize=4, markerfacecolor="white",
                markeredgewidth=1.0, linestyle="none")
        ax.plot([x1], [y1], marker=sty[tag]["marker"], color=c, markersize=4, linestyle="none",
                markeredgecolor="white", markeredgewidth=0.5, label=name)
    ax.set_xlabel("Over-refusal on answerable questions (%)")
    ax.set_ylabel("Hallucination on\nunanswerable questions (%)")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(left=-2)
    ax.set_ylim(top=102)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    ax.text(0.99, 0.98, "hollow = neutral, filled = abstain-allowed", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6, color=INK2)
    fig.savefig(OUT_F / "tradeoff.pdf")
    plt.close(fig)


def fig_reliability(df, models):
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7), sharey=True)
    s = df[~df.refused]
    bins = np.linspace(0, 1, 11)
    for ax, (col, title, scale) in zip(axes, (("verbal_conf", "(a) Verbalized", 100), ("p_true", "(b) P(True)", 1))):
        conf = s[col].astype(float) / scale
        ok = ~conf.isna()
        conf, corr = conf[ok].values, s.correct[ok].astype(float).values
        idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(corr[m].mean())
                ns.append(m.sum())
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.7, linestyle="--")
        ax.bar(bins[:-1] + 0.05, np.histogram(conf, bins)[0] / len(conf), width=0.09,
               color="#e1e0d9", zorder=0)
        ax.plot(xs, ys, color=PALETTE[0], marker="o", markersize=3, linewidth=1.2,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Stated confidence")
        ax.text(0.03, 0.93, f"ECE {ece(conf, corr):.2f}", transform=ax.transAxes, fontsize=6, color=INK2)
    axes[0].set_ylabel("Observed accuracy")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT_F / "reliability.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- audit sample
def audit_sample(df, n_per_cell=15):
    """Stratified sample (model x answerable x label) for manual verification of the grader."""
    parts = []
    for (m, a, l), s in df.groupby(["model", "answerable", "label"]):
        parts.append(s.sample(min(n_per_cell, len(s)), random_state=11))
    a = pd.concat(parts)
    a = a.sample(frac=1, random_state=11)[["model", "qid", "question", "answerable", "gold",
                                          "answer", "label"]]
    a["human_label"] = ""
    path = ROOT / "results/audit/audit_sample.csv"
    if not path.exists():  # never overwrite manual labels
        a.to_csv(path, index=False)
    return path


def audit_agreement():
    path = ROOT / "results/audit/audit_sample.csv"
    if not path.exists():
        return None
    a = pd.read_csv(path, keep_default_na=False)
    a = a[a.human_label.str.strip() != ""]
    if a.empty:
        return None
    from sklearn.metrics import cohen_kappa_score
    collapse = {"CORRECT": "ok", "REJECTED": "ok", "WRONG": "hall", "ACCEPTED": "hall",
                "REFUSED": "ref"}
    acc = float((a.label == a.human_label).mean())
    kappa = float(cohen_kappa_score(a.label, a.human_label))
    k3 = float(cohen_kappa_score(a.label.map(collapse), a.human_label.map(collapse)))
    res = {"n": int(len(a)), "accuracy": acc, "kappa": kappa, "kappa_3way": k3,
           "confusion": pd.crosstab(a.label, a.human_label).to_dict()}
    if "human_label_B" in a:
        b = a[a.human_label_B.astype(str).str.strip() != ""]
        if len(b):
            res["n_double"] = int(len(b))
            res["inter_annotator_kappa"] = float(cohen_kappa_score(b.human_label, b.human_label_B))
    return res



# ----------------------------------------------------------------------------- numbers.tex
def write_numbers(df, om, cm, pm, dm, um, cal, corr, gm, au, models):
    """Every number quoted in the prose is emitted as a macro: \\res{<model>}{<key>}."""
    defs = {}

    def put(tag, key, val, pct=True, d=1):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            txt = "--"
        else:
            txt = f"{100 * val:.{d}f}" if pct else f"{val:.{d}f}"
        defs[(tag, key)] = txt

    for tag, *_ in models:
        for cond, c in (("neutral", "N"), ("abstain", "A")):
            for k in ("acc_A", "hall_A", "ref_A", "rej_U", "hall_U", "hall_all"):
                put(tag, k.replace("_", "") + c, om[tag][cond][k][0])
            put(tag, "J" + c, om[tag][cond]["J"][0], pct=False, d=2)
        for k in ("mcnemar_U", "mcnemar_A_refusal"):
            p_ = om[tag][k]["p"]
            defs[(tag, k.replace("_", "") + "p")] = "<0.001" if p_ < 1e-3 else f"{p_:.3f}"
        for cat in ("NE", "FP", "NC", "TI", "FAKE"):
            put(tag, f"rej{cat}N", cm[tag][cat]["rej_neutral"], d=0)
            put(tag, f"rej{cat}A", cm[tag][cat]["rej_abstain"], d=0)
        for k in ("both_ok", "knows_twin", "accept_F_given_knows_T"):
            put(tag, k.replace("_", ""), pm[tag][k], d=0)
        for k in ("delta_verbal_conf", "delta_p_ik", "delta_p_true"):
            put(tag, k.replace("_", ""), pm[tag][k] / (100 if "verbal" in k else 1), d=1)
        for col, *_ in SIGNALS:
            put(tag, "auc" + col.replace("_", ""), dm[tag][col][0], pct=False, d=2)
            put(tag, "uauc" + col.replace("_", ""), um[tag][col][0], pct=False, d=2)
        for k, v in cal[tag].items():
            put(tag, k.replace("_", ""), v, pct=not k.startswith("ece"), d=3 if k.startswith("ece") else 1)
        for col, r in corr.get(tag, {}).items():
            put(tag, "rho" + col.replace("_", ""), r["rho"], pct=False, d=2)
            defs[(tag, "rhop" + col.replace("_", ""))] = "<0.001" if r["p"] < 1e-3 else f"{r['p']:.3f}"
    for col, *_ in SIGNALS:
        put("all", "auc" + col.replace("_", ""), dm["pooled"][col][0], pct=False, d=2)
        put("all", "auclo" + col.replace("_", ""), dm["pooled"][col][1], pct=False, d=2)
        put("all", "auchi" + col.replace("_", ""), dm["pooled"][col][2], pct=False, d=2)
    # ranges across models
    def rng(vals, pct=True, d=1):
        k = 100 if pct else 1
        return f"{k * min(vals):.{d}f}--{k * max(vals):.{d}f}"
    tags = [m[0] for m in models]
    defs[("all", "hallUNrange")] = rng([om[t]["neutral"]["hall_U"][0] for t in tags])
    defs[("all", "hallUArange")] = rng([om[t]["abstain"]["hall_U"][0] for t in tags])
    defs[("all", "refANrange")] = rng([om[t]["neutral"]["ref_A"][0] for t in tags])
    defs[("all", "refAArange")] = rng([om[t]["abstain"]["ref_A"][0] for t in tags])
    defs[("all", "accANrange")] = rng([om[t]["neutral"]["acc_A"][0] for t in tags])
    defs[("all", "JNrange")] = rng([om[t]["neutral"]["J"][0] for t in tags], pct=False, d=2)
    defs[("all", "JArange")] = rng([om[t]["abstain"]["J"][0] for t in tags], pct=False, d=2)
    defs[("all", "eceverbalrange")] = rng([cal[t]["ece_verbal"] for t in tags], pct=False, d=2)
    defs[("all", "confgerange")] = rng([cal[t]["frac_conf_ge_90"] for t in tags], d=0)
    defs[("all", "nmodels")] = str(len(models))
    defs[("all", "nq")] = str(df.qid.nunique())
    defs[("all", "nresp")] = f"{len(df):,}".replace(",", "{,}")
    defs[("all", "nA")] = str(om[tags[0]]["n_answerable"])
    defs[("all", "nU")] = str(om[tags[0]]["n_unanswerable"])
    if au:
        put("all", "auditacc", au["accuracy"])
        put("all", "auditkappa", au["kappa"], pct=False, d=2)
        put("all", "auditkappathree", au["kappa_3way"], pct=False, d=2)
        defs[("all", "auditn")] = str(au["n"])
        if "inter_annotator_kappa" in au:
            put("all", "iaakappa", au["inter_annotator_kappa"], pct=False, d=2)
            defs[("all", "iaan")] = str(au["n_double"])
    L = ["% Auto-generated by scripts/analyze.py -- do not edit by hand.",
         r"\makeatletter",
         r"\newcommand{\res}[2]{\@ifundefined{res@#1@#2}{\textbf{??}}{\csname res@#1@#2\endcsname}}"]
    for (t, k), v in sorted(defs.items()):
        L.append(r"\expandafter\def\csname res@%s@%s\endcsname{%s}" % (t, k, v.replace("<", "$<$")))
    L.append(r"\makeatother")
    (ROOT / "paper/numbers.tex").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    df = load()
    if df.empty:
        print("no results yet")
        return
    models = present_models(df)
    counts = df.groupby("model").size().to_dict()
    print("records per model:", counts)
    df.to_csv(ROOT / "results/graded/all.csv", index=False)

    om = outcome_metrics(df)
    cm = category_metrics(df)
    pm = pair_metrics(df)
    dm = detection_metrics(df)
    um = unanswerable_detection(df)
    cal = calibration_metrics(df)
    gm, corr = gradient_metrics(df)
    gm.to_csv(OUT_S / "gradient.csv", index=False)
    if "--audit" in sys.argv:  # freeze the audit sample only once all runs are complete
        audit_sample(df)
    au = audit_agreement()

    summary = {"records": counts, "outcomes": om, "categories": cm, "pairs": pm,
               "detection": dm, "unanswerable_detection": um, "calibration": cal,
               "popularity_spearman": corr, "audit": au,
               "mean_seconds_per_question": df.groupby("model").seconds.mean().to_dict()}
    (OUT_S / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))

    write_numbers(df, om, cm, pm, dm, um, cal, corr, gm, au, models)
    table_main(om, models)
    table_categories(cm, models)
    table_detection(dm, um, models)
    table_calibration(cal, pm, models)
    fig_outcomes(df, models)
    if not gm.empty:
        fig_gradient(gm, models)
        fig_signals_gradient(gm, models)
    fig_auroc_heatmap(dm, models)
    fig_tradeoff(om, models)
    fig_reliability(df, models)
    print("wrote tables to", OUT_T, "and figures to", OUT_F)


if __name__ == "__main__":
    main()
