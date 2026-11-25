"""The two figures of System Switch II, from the released records.

Figure 1 (Section 4): for each of the eleven actors of Paper I, on the 440 distinct command questions with
Qwen3.6 reasoning as thinker and 30% of the least confident decisions deferred: AUROC of the actor's confidence
against (a) the gain of confidence-gated deferral over random deferral and (b) the thinker's accuracy on the
deferred questions minus its accuracy on all questions.

Figure 2 (Section 7): (a) the student's confidence at the surprise checks (states drawn at random, gate closed)
before training (rounds gemma1, gemma2, deepseek1) and after (round dagger1), with the 0.3 threshold; (b) share of
explore decisions by game time in the evaluation games (seeds 48-65) of the untrained student, the three Gemma
students of one round and the two students of the second round (data/teacher/eval_decisions.jsonl.gz, one line per
decision, extracted from the games' decision logs).

Usage: figures.py <data directory of github.com/dexmac221/system-switch> <output directory>
"""
import collections, glob, gzip, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testset import keep_indices                                   # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3, validated on the light surface
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

# ---------------------------------------------------------------- Figure 1
test = json.load(open(f"{D}/test_set/test_900.json"))
keep = keep_indices(test, True)
T = {r["i"]: r for r in map(json.loads, open(f"{D}/thinkers/qwen36uff_think.jsonl")) if r["kind"] == "command"}
ACT = [("laya-base", "Laya base", 0), ("kev-0.8b", "Kev 0.8B", 0), ("laya-typed", "Laya typed", 0),
       ("kev-4b", "Kev 4B", 0), ("clef-flash-9b", "Clef-Flash", 0), ("lev-4b", "Lev", 0), ("kev-9b", "Kev 9B", 0),
       ("strands-2b", "Strands", 0), ("julia-1", "Julia-1", 0), ("laya-v3", "Laya-v3", 1), ("doomgemma", "doomGemma", 1)]


def auroc(rows):
    pos = [r["conf"] for r in rows if r["right"]]; neg = [r["conf"] for r in rows if not r["right"]]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


pts = []
for f, name, trained in ACT:
    A = {r["i"]: r for r in map(json.loads, open(f"{D}/actors/fixed_order/{f}.jsonl")) if r["kind"] == "command"}
    ids = sorted(i for i in A if i in keep and i in T)
    n = len(ids); k = round(0.3 * n)
    acc = sum(A[i]["right"] for i in ids) / n; thk = sum(T[i]["right"] for i in ids) / n
    d = set(sorted(ids, key=lambda i: (A[i]["conf"], i))[:k])
    comb = sum(T[i]["right"] if i in d else A[i]["right"] for i in ids) / n
    gain = 100 * (comb - (acc + k / n * (thk - acc)))
    diff = 100 * (sum(T[i]["right"] for i in d) / k - thk)
    pts.append((name, trained, auroc([A[i] for i in ids]), gain, diff, 100 * (comb - acc)))
    print(f"{name:11} n={n} AUROC {pts[-1][2]:.2f} gain over random {gain:+.1f} thinker difference {diff:+.1f} "
          f"net gain {pts[-1][5]:+.1f}")

LABELS = {"Laya base", "Laya-v3", "doomGemma", "Kev 9B", "Strands", "Julia-1"}
OFF = {"Laya base": (-6, 5, "right"), "Laya-v3": (6, -2, "left"), "doomGemma": (6, -2, "left"), "Kev 9B": (6, -3, "left"),
       "Strands": (6, 2, "left"), "Julia-1": (6, 2, "left")}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.55), constrained_layout=True)
for ax, col, ylab in ((axs[0], 3, "gain over random deferral (points)"),
                      (axs[1], 4, "thinker on deferred minus overall (points)")):
    ax.axhline(0, color=INK2, lw=0.6)
    for trained, color, marker, lab in ((0, BLUE, "o", "zero-shot actor"), (1, ORANGE, "s", "trained on the task")):
        P = [p for p in pts if p[1] == trained]
        ax.scatter([p[2] for p in P], [p[col] for p in P], s=34, marker=marker, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3, label=lab)
    for p in pts:
        if p[0] in LABELS:
            dx, dy, ha = OFF[p[0]]
            ax.annotate(p[0], (p[2], p[col]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=7, color=INK2)
    ax.set_xlabel("AUROC of the actor's confidence")
    ax.set_ylabel(ylab)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0.4, 0.9)
axs[0].set_title("(a) deferral gains where confidence separates", fontsize=8, loc="left", color=INK)
axs[1].set_title("(b) deferred questions are not average", fontsize=8, loc="left", color=INK)
axs[0].legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.3)
fig.savefig(f"{OUT}/fig_deferral.pdf")
print("written", f"{OUT}/fig_deferral.pdf")

# ---------------------------------------------------------------- Figure 2
R = os.path.join(D, "teacher", "rounds") if os.path.isdir(os.path.join(D, "teacher", "rounds")) else None
assert R, "data/teacher/rounds not found"


def surprise_conf(rounds):
    out = []
    for g in rounds:
        for f in sorted(glob.glob(f"{R}/{g}/etichette_s*.jsonl")):
            out += [float(r["conf"]) for r in map(json.loads, open(f)) if r.get("maestro") and r["tipo"] == "sorpresa"]
    return sorted(out)


before, after = surprise_conf(["gemma1", "gemma2", "deepseek1"]), surprise_conf(["dagger1"])
print(f"surprise checks: before training n={len(before)}, below 0.3 {sum(c < 0.3 for c in before) / len(before):.0%}; "
      f"after n={len(after)}, below 0.3 {sum(c < 0.3 for c in after) / len(after):.0%}")
tipo = lambda c: "open_door" if c.startswith("open") else re.sub(r"_\d+$", "", c)
WIN = [(0, 30), (30, 60), (60, 120), (120, 181)]
series = None
if os.path.exists(f"{D}/teacher/eval_decisions.jsonl.gz"):
    G = collections.defaultdict(list)
    for r in map(json.loads, gzip.open(f"{D}/teacher/eval_decisions.jsonl.gz", "rt")):
        G[r["variant"]].append({"game_seconds": r["t"], "choice": r["choice"]})

    def share(vs):
        out = []
        for a, b in WIN:
            d = [r for v in vs for r in G[v] if a <= r["game_seconds"] < b]
            out.append(100 * sum(tipo(r["choice"]) == "explore" for r in d) / len(d))
        return out
    series = [("untrained student", BLUE, "o", share(["base"])),
              ("Gemma students, one round", AQUA, "^", share(["gemma5", "gemma5b", "gemma5c"])),
              ("second-round students", ORANGE, "s", share(["dagger_a", "dagger_b"]))]
    for name, _, _, s in series:
        print(f"explore share by window {name}: " + ", ".join(f"{x:.0f}%" for x in s))

fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.55), constrained_layout=True)
ax = axs[0]
for data, color, lab, ls in ((before, BLUE, f"before training ({len(before)} checks)", "-"),
                             (after, ORANGE, f"after one round ({len(after)} checks)", "-")):
    y = [(i + 1) / len(data) for i in range(len(data))]
    ax.step(data, y, where="post", color=color, lw=1.4, label=lab, ls=ls)
ax.axvline(0.3, color=INK2, lw=0.8, ls=(0, (3, 2)))
ax.annotate("threshold", (0.3, 0.72), xytext=(-4, 0), textcoords="offset points", fontsize=7, color=INK2,
            va="center", ha="right")
ax.set_xlabel("student's confidence (top probability)")
ax.set_ylabel("share of surprise checks at or below")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.4)
ax.set_title("(a) confidence before and after training", fontsize=8, loc="left", color=INK)
ax = axs[1]
if series:
    x = [15, 45, 90, 150]
    dy = {"untrained student": -4, "Gemma students, one round": 4}
    for name, color, marker, s in series:
        ax.plot(x, s, color=color, lw=1.4, marker=marker, ms=5, mec="white", mew=0.8, label=name)
        ax.annotate(f"{s[-1]:.0f}%", (x[-1], s[-1]), xytext=(6, dy.get(name, 0)), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.set_xticks(x, ["0–30", "30–60", "60–120", "120–180"])
    ax.set_xlim(5, 172)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.4)
ax.set_xlabel("game time (s)")
ax.set_ylabel("explore decisions (%)")
ax.set_ylim(0, 60)
ax.set_title("(b) exploring during the game", fontsize=8, loc="left", color=INK)
for a in axs:
    a.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig_teacher.pdf")
print("written", f"{OUT}/fig_teacher.pdf")
