# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 10 — VISUALIZATION (GodMode visualization standard)
# DejaVu Serif, no titles, no bold, full border, no grid, tab10, axis labels 13 pt, ticks 11 pt, PNG at dpi=600.
# Each figure is produced only when its analysis exists; missing inputs are logged and skipped.
# =====================================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "DejaVu Serif", "font.weight": "normal", "axes.labelweight": "normal",
                            "axes.titleweight": "normal"})
COLORS = plt.cm.tab10.colors
FIG_DIR = ANALYSIS_DIR / "figures"
FIGURES = []

def style(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontsize=13, fontweight="normal"); ax.set_ylabel(ylabel, fontsize=13, fontweight="normal")
    ax.tick_params(labelsize=11, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.grid(False)

def panel(ax, label: str):
    ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=12, fontweight="normal", va="bottom", ha="right")

def save(fig, name: str):
    path = FIG_DIR / name
    plt.tight_layout(pad=1.5)
    fig.savefig(path, dpi=600, bbox_inches="tight", format="png")
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 0, f"Export failed: {path}"
    FIGURES.append(path); logger.info(f"Figure saved: {path.name}")

def generate_figures(A: dict, df: pd.DataFrame):
    if "condition" not in df or "stage" not in df:              # no training runs evaluated yet
        df = pd.DataFrame(columns=["stage", "condition", "encoder", "seed", "test_acc", "cri", "calibration", "test_confusion"])
    df = df[df["kind"].fillna("train") == "train"] if "kind" in df else df
    # (1) H1 explicit share per corpus
    h1 = A.get("H1_explicit_share")
    if h1:
        names = ["setfit_test", "isear", "goemotions", "ed_test", "tweeteval_test", "meld"]
        names = [n for n in names if n in h1]
        fig, ax = plt.subplots(figsize=(6, 5))
        v = [100 * h1[n]["share"] for n in names]
        err = np.array([[100 * (h1[n]["share"] - h1[n]["wilson95"][0]) for n in names],
                        [100 * (h1[n]["wilson95"][1] - h1[n]["share"]) for n in names]])
        ax.bar(range(len(names)), v, yerr=err, color=[COLORS[i] for i in range(len(names))], alpha=0.85, capsize=3)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_test", "") for n in names], rotation=30, ha="right")
        ax.set_ylim(0, 100); style(ax, "Corpus", "Explicit items (%)")
        save(fig, "fig_explicit_share.png")
    # (2) H2 explicit vs implicit transfer accuracy
    h2 = A.get("H2_stratified_transfer")
    if h2:
        t = list(h2)
        fig, ax = plt.subplots(figsize=(6, 5)); x = np.arange(len(t)); w = 0.38
        ax.bar(x - w / 2, [100 * h2[k]["acc_explicit"] for k in t], w, color=COLORS[0], label="Explicit items")
        ax.bar(x + w / 2, [100 * h2[k]["acc_implicit"] for k in t], w, color=COLORS[1], label="Implicit items")
        ax.set_xticks(x); ax.set_xticklabels(t); ax.legend(fontsize=11); style(ax, "Target corpus", "Transfer accuracy (%)")
        save(fig, "fig_stratified_transfer.png")
    # (3) H3 CRI vs transfer accuracy, one panel per target
    if len(df) and "cri" in df and A.get("H3_cri_vs_transfer"):
        tg = [t for t in TRANSFER_TARGETS if t in A["H3_cri_vs_transfer"]]
        fig, axes = plt.subplots(1, len(tg), figsize=(5 * len(tg), 5), squeeze=False)
        encs = sorted(df.encoder.unique())
        for j, (ax, t) in enumerate(zip(axes[0], tg)):
            c = df[df.stage.isin(["main", "shortcut"]) & df.cri.notna() & df[f"transfer_{t}"].notna()]
            for i, e in enumerate(encs):
                s = c[c.encoder == e]
                if len(s):
                    ax.scatter(s.cri, [100 * d["restricted"]["acc"] for d in s[f"transfer_{t}"]], s=14, color=COLORS[i % 10],
                               alpha=0.75, label=e, marker="o")
            style(ax, "Cue-reliance index", f"{t} accuracy (%)"); ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            panel(ax, f"({chr(97 + j)})")
        axes[0][0].legend(fontsize=9)
        save(fig, "fig_cri_vs_transfer.png")
    # (4) seed-paired prompt effect (A vs none) per encoder with random-effects pooled estimate
    sp = pd.DataFrame(A.get("seed_paired", []))
    if len(sp):
        sp = sp[(sp.condition == "A") & sp.stage.isin(["main", "intra"]) & (sp.text_transform == "none")]
    if len(sp):
        corp = sorted(sp.corpus.unique())
        fig, axes = plt.subplots(1, len(corp), figsize=(5 * len(corp), 5), squeeze=False)
        for j, (ax, cp) in enumerate(zip(axes[0], corp)):
            s = sp[sp.corpus == cp].sort_values("encoder").reset_index(drop=True)
            for i, r in s.iterrows():
                ax.plot(r.ci95, [i, i], color=COLORS[0], lw=1.5); ax.plot(r["mean"], i, "o", color=COLORS[0], ms=5)
            key = next((k for k in A.get("random_effects", {}) if k.startswith(f"{s.stage.iloc[0]}|{cp}|A|")), None)
            labels = list(s.encoder)
            if key:
                re_ = A["random_effects"][key]; i = len(s)
                ax.plot(re_["ci95"], [i, i], color=COLORS[3], lw=1.5); ax.plot(re_["mu"], i, "o", color=COLORS[3], ms=6)
                labels.append("Random-effects pooled")
            ax.axvline(0, color="0.3", lw=0.8, ls="--")
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
            style(ax, "Accuracy difference, A minus none (pp)", ""); panel(ax, f"({chr(97 + j)})")
        save(fig, "fig_prompt_effects.png")
    # (5) multiseed summary: main-grid accuracy per encoder without prompt (mean +- SD)
    if len(df):
        m = df[(df.stage == "main") & (df.condition == "none")]
        if len(m):
            g = m.groupby("encoder").test_acc.agg(["mean", "std", "count"]).reset_index()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(range(len(g)), 100 * g["mean"], yerr=100 * g["std"], color=[COLORS[i] for i in range(len(g))], alpha=0.85, capsize=3, width=0.6)
            ax.set_xticks(range(len(g))); ax.set_xticklabels(g.encoder, rotation=30, ha="right")
            lo = 100 * (g["mean"] - g["std"]).min(); ax.set_ylim(max(0, lo - 2), min(100, 100 * (g["mean"] + g["std"]).max() + 1))
            style(ax, "Encoder", "Test accuracy (%)")
            save(fig, "multiseed_summary.png")
    # (6) convergence rate
    cv = A.get("convergence")
    if cv:
        fig, ax = plt.subplots(figsize=(6, 5)); x = np.arange(len(cv)); w = 0.38
        for j, (key, lab) in enumerate([("converged_none", "No prompt"), ("converged_A", "Template A")]):
            rates = [cv[e][key][0] / cv[e][key][1] for e in cv]
            ci = [wilson(cv[e][key][0], cv[e][key][1]) for e in cv]
            err = np.array([[100 * (r - c[0]) for r, c in zip(rates, ci)], [100 * (c[1] - r) for r, c in zip(rates, ci)]])
            ax.bar(x + (j - 0.5) * w, [100 * r for r in rates], w, yerr=err, color=COLORS[j], capsize=3, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(list(cv)); ax.set_ylim(0, 105); ax.legend(fontsize=11)
        style(ax, "Encoder", "Converged runs (%)")
        save(fig, "fig_convergence.png")

def calibration_and_confusion(df: pd.DataFrame):
    """Reliability diagram and row-normalized confusion matrix for RoBERTa (or the first encoder), template A, lowest seed."""
    if "condition" not in df or "stage" not in df:              # no training runs evaluated yet
        df = pd.DataFrame(columns=["stage", "condition", "encoder", "seed", "test_acc", "cri", "calibration", "test_confusion"])
    df = df[df["kind"].fillna("train") == "train"] if "kind" in df else df
    if not len(df) or "calibration" not in df:
        return
    m = df[(df.stage == "main") & (df.condition == "A") & df.calibration.notna()]
    if not len(m):
        return
    enc = "roberta" if "roberta" in set(m.encoder) else sorted(m.encoder)[0]
    r = m[m.encoder == enc].sort_values("seed").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for j, (tag, lab) in enumerate([("before", "Before temperature scaling"), ("after", "After temperature scaling")]):
        ax = axes[j]
        bins = [b for b in r.calibration[tag]["reliability_bins"] if b["count"]]
        ax.bar([(b["lo"] + b["hi"]) / 2 for b in bins], [b["acc"] for b in bins], width=0.9 / CALIBRATION_BINS, color=COLORS[0], alpha=0.8, label="Accuracy")
        ax.plot([(b["lo"] + b["hi"]) / 2 for b in bins], [b["conf"] for b in bins], "o", color=COLORS[3], ms=4, label="Mean confidence")
        ax.plot([0, 1], [0, 1], "--", color="0.4", lw=1)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.legend(fontsize=10, loc="upper left")
        style(ax, "Confidence", "Accuracy"); panel(ax, f"({chr(97 + j)})")
        ax.text(0.98, 0.04, f"ECE {100 * r.calibration[tag]['ece']:.2f}%", transform=ax.transAxes, ha="right", fontsize=11)
    save(fig, "fig_reliability.png")
    cm = np.asarray(r.test_confusion, float); cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(100 * cm, cmap="viridis", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white" if cm[i, j] < 0.6 else "black")
    names = SETFIT[:cm.shape[0]]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, extend="neither"); cbar.set_label("Row share (%)", fontsize=11)
    style(ax, "Predicted class", "True class")
    save(fig, "fig_confusion.png")

generate_figures(ANALYSIS, METRICS_DF)
calibration_and_confusion(METRICS_DF)
logger.info(f"{len(FIGURES)} figures generated")
