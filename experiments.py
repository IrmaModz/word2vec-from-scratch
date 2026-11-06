"""
CS585 PA02 - Analysis experiments (NOT submitted).
Reuses the Part A code to train embeddings under different settings,
then compares nearest neighbors, similarity scores and 3D PCA plots.

Usage (same folder as CS585_P02A_A20597307.py):
    python experiments.py CORPUS_FILE

Outputs go to the "experiments" folder:
    summary.md          tables to copy into the report
    D_<word>.png        D = 50 / 100 / 200 side by side
    GD_<word>.png       D = 50 with different gradient descent settings
"""
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import CS585_P02A_A20597307 as A

# ---------------- edit these for your corpus ----------------
TEST_WORDS = ["she", "mother", "sister", "morning", "two", "good"]

RELATED_PAIRS = [
    ("he", "she"), ("his", "her"), ("mr", "mrs"),
    ("mother", "father"), ("sister", "brother"), ("daughter", "son"),
    ("morning", "evening"), ("day", "night"),
    ("two", "three"), ("three", "four"),
    ("man", "woman"), ("lady", "gentleman"),
]

K = 8

BASELINE = {"D": 50, "LEARNING_RATE": 0.025, "EPOCHS": 5,
            "NEG_SAMPLES": 2, "WINDOW_SIZE": 2}

D_RUNS = [{"D": 50}, {"D": 100}, {"D": 200}]

GD_RUNS = [
    {"LEARNING_RATE": 0.005},
    {"LEARNING_RATE": 0.1},
    {"EPOCHS": 1},
    {"EPOCHS": 15},
    {"NEG_SAMPLES": 5},
    {"WINDOW_SIZE": 5},
]
# -------------------------------------------------------------

OUT_DIR = "experiments"


def label_of(changes):
    if not changes or changes == {"D": 50}:
        return "baseline"
    return ", ".join(f"{k.lower()}={v}" for k, v in changes.items())


def unit_rows(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return W / norms


def avg_loss(W_in, W_out, pairs, neg_probs, k, n_sample=20000):
    rng = np.random.default_rng(0)
    sample = pairs[rng.choice(len(pairs), size=min(n_sample, len(pairs)), replace=False)]
    negs = rng.choice(len(neg_probs), size=(len(sample), k), p=neg_probs)
    t = W_in[sample[:, 0]]
    c = W_out[sample[:, 1]]
    pos = A.sigmoid(np.sum(t * c, axis=1))
    neg = A.sigmoid(np.einsum("nd,nkd->nk", t, W_out[negs]))
    loss = -(np.log(pos + 1e-10) + np.sum(np.log(1 - neg + 1e-10), axis=1))
    return loss.mean()


def run(config, tokens, V, w2i, neg_probs, pair_cache):
    A.LEARNING_RATE = config["LEARNING_RATE"]
    A.EPOCHS = config["EPOCHS"]
    A.NEG_SAMPLES = config["NEG_SAMPLES"]
    A.WINDOW_SIZE = config["WINDOW_SIZE"]

    if A.WINDOW_SIZE not in pair_cache:
        pair_cache[A.WINDOW_SIZE] = A.build_training_pairs(tokens, w2i)
    pairs = pair_cache[A.WINDOW_SIZE]

    start = time.perf_counter()
    W_in, W_out = A.init_weights(len(V), config["D"])
    W_in, W_out = A.train(pairs, W_in, W_out, neg_probs)
    seconds = time.perf_counter() - start

    loss = avg_loss(W_in, W_out, np.array(pairs), neg_probs, config["NEG_SAMPLES"])
    return W_in, seconds, loss


def neighbors(W_unit, V, w2i, word, k):
    sims = W_unit @ W_unit[w2i[word]]
    sims[w2i[word]] = -np.inf
    idx = np.argsort(-sims)[:k]
    return [(V[i], sims[i]) for i in idx]


def pair_scores(W_unit, w2i, V):
    found = [(a, b) for a, b in RELATED_PAIRS if a in w2i and b in w2i]
    related = {f"{a}/{b}": float(W_unit[w2i[a]] @ W_unit[w2i[b]]) for a, b in found}
    rng = np.random.default_rng(1)
    r = rng.integers(0, len(V), size=(2000, 2))
    r = r[r[:, 0] != r[:, 1]]
    random_mean = float(np.mean(np.sum(W_unit[r[:, 0]] * W_unit[r[:, 1]], axis=1)))
    return related, random_mean


def draw(ax, W, V, w2i, word, k, title):
    pca = PCA(n_components=3).fit(W)
    nb = neighbors(unit_rows(W), V, w2i, word, k)
    idx = [w2i[word]] + [w2i[w] for w, _ in nb]
    pts = pca.transform(W[idx])
    labels = [word] + [w for w, _ in nb]
    for i, (lab, (x, y, z)) in enumerate(zip(labels, pts)):
        col = "red" if i == 0 else "steelblue"
        ax.plot([0, x], [0, y], [0, z], color=col, alpha=0.35)
        ax.scatter(x, y, z, color=col, s=60 if i == 0 else 25)
        ax.text(x, y, z, f" {lab}", color=col, fontsize=9 if i == 0 else 7)
    ax.scatter(0, 0, 0, color="black", s=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def comparison_figure(results, names, V, w2i, word, path, cols):
    rows = int(np.ceil(len(names) / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    for n, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, n + 1, projection="3d")
        draw(ax, results[name]["W"], V, w2i, word, K, name)
    fig.suptitle(f"'{word}' and its {K} nearest neighbors (PCA, 3D)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print("Usage: python experiments.py CORPUS_FILE")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    text = A.clean_file(sys.argv[1])
    tokens, V, w2i = A.create_vocabulary(text)
    neg_probs = A.build_neg_probs(tokens, w2i)
    print(f"Tokens: {len(tokens)}  Vocabulary: {len(V)}")

    words = [w for w in TEST_WORDS if w in w2i]
    missing = [w for w in TEST_WORDS if w not in w2i]
    if missing:
        print(f"Not in vocabulary (skipped): {missing}")

    results = {}
    pair_cache = {}
    d_names, gd_names = [], ["baseline"]

    for changes in D_RUNS + GD_RUNS:
        config = {**BASELINE, **changes}
        name = f"D={config['D']}" if changes in D_RUNS else label_of(changes)
        if changes == {"D": 50}:
            name = "baseline"
        print(f"Training {name} ...", flush=True)
        W, seconds, loss = run(config, tokens, V, w2i, neg_probs, pair_cache)
        results[name] = {"W": W, "seconds": seconds, "loss": loss, "config": config}
        print(f"   {seconds:.1f} s, loss {loss:.4f}")
        if changes in D_RUNS:
            d_names.append(name)
        else:
            gd_names.append(name)

    lines = ["# Experiment results", "",
             f"Tokens: {len(tokens)}, vocabulary: {len(V)}", "",
             "Baseline: " + ", ".join(f"{k}={v}" for k, v in BASELINE.items()), "",
             "## Runs", "",
             "| Run | D | lr | epochs | k | window | time (s) | loss | related sim | random sim |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    base_unit = unit_rows(results["baseline"]["W"])
    for name in ["baseline", "D=100", "D=200"] + gd_names[1:]:
        r = results[name]
        c = r["config"]
        Wu = unit_rows(r["W"])
        related, rand = pair_scores(Wu, w2i, V)
        r["related"], r["random"] = related, rand
        rel_mean = np.mean(list(related.values())) if related else float("nan")
        lines.append(f"| {name} | {c['D']} | {c['LEARNING_RATE']} | {c['EPOCHS']} | "
                     f"{c['NEG_SAMPLES']} | {c['WINDOW_SIZE']} | {r['seconds']:.1f} | "
                     f"{r['loss']:.4f} | {rel_mean:.3f} | {rand:.3f} |")

    pair_names = list(results["baseline"]["related"].keys())
    if pair_names:
        lines += ["", "## Cosine similarity of related pairs", "",
                  "| Pair | " + " | ".join(["baseline", "D=100", "D=200"] + gd_names[1:]) + " |",
                  "|---|" + "---|" * (len(gd_names) + 2)]
        for p in pair_names:
            vals = [results[n]["related"][p] for n in ["baseline", "D=100", "D=200"] + gd_names[1:]]
            lines.append(f"| {p} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    lines += ["", f"## Nearest neighbors (K={K})", ""]
    for word in words:
        lines += [f"### {word}", "", "| Run | Neighbors | Shared with baseline |", "|---|---|---|"]
        base_set = {w for w, _ in neighbors(base_unit, V, w2i, word, K)}
        for name in ["baseline", "D=100", "D=200"] + gd_names[1:]:
            nb = neighbors(unit_rows(results[name]["W"]), V, w2i, word, K)
            shared = len(base_set & {w for w, _ in nb})
            text_nb = ", ".join(f"{w} ({s:.2f})" for w, s in nb)
            lines.append(f"| {name} | {text_nb} | {shared}/{K} |")
        lines.append("")

        comparison_figure(results, ["baseline", "D=100", "D=200"], V, w2i, word,
                          os.path.join(OUT_DIR, f"D_{word}.png"), cols=3)
        comparison_figure(results, gd_names, V, w2i, word,
                          os.path.join(OUT_DIR, f"GD_{word}.png"), cols=4)

    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Done. Results in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
