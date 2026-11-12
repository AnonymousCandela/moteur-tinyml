"""Figures de l'article (format colonne IEEE, 3,4 po de large)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.engine import EdgeEngine
from sim.scenario import run_scenario

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)

# Figure 2 : détection par C, par attaque et position de l'attaquant (moyenne ± écart type, 5 graines)
S = json.load(open("results/summary.json"))["par_attaque"]
att = ["C1_injection", "C2_modification", "C3_temp_figee", "C4_rejeu", "C5_bruit"]
lab = ["C1\ninjection", "C2\ncourant", "C3\nT figée", "C4\nrejeu", "C5\nbruit"]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
x = np.arange(len(att)); w = 0.26
for j, (var, col, name) in enumerate([("interception", "#9ecae1", "C, interception"), ("compromis", "#2171b5", "C, nœud compromis")]):
    m = [S[f"{a}|{var}"]["detection_C"]["moyenne"] * 100 for a in att]
    e = [S[f"{a}|{var}"]["detection_C"]["ecart_type"] * 100 for a in att]
    ax.bar(x + (j - 1) * w, m, w, yerr=e, color=col, label=name, capsize=1.5, error_kw=dict(lw=0.6))
b = [np.mean([S[f"{a}|{v}"]["B_trompe"]["moyenne"] for v in ("interception", "compromis")]) * 100 for a in att]
ax.bar(x + w, b, w, color="#e6550d", label="B trompé")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("% fenêtres attaquées"); ax.set_ylim(0, 108)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=6.2, columnspacing=0.8, handlelength=1.2)
fig.tight_layout(); fig.savefig("figures/fig2_detection.png", dpi=300); plt.close(fig)

# Figure 3 : scénario de masquage P1 surchauffe + C3 température figée (nœud compromis)
s = run_scenario(seed=7, fault=1, attack=3, attack_onset_s=50, variant="compromis")
eng = EdgeEngine("results/s0"); eng.commission(s["commissioning"])
D = [r for k in range(len(s["received"])) if (r := eng.push(s["received"][k], s["accepted"][k], s["watermark"][k]))]
t = np.arange(len(s["received"])) / 10
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 2.4), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
a1.plot(t, s["true"][:, 3], color="#999999", lw=0.9, ls="--", label="température réelle")
a1.plot(t, s["received"][:, 3], color="#08519c", lw=1.0, label="température reçue")
a1.axvline(50, color="#c62828", lw=0.8, ls=":"); a1.text(51, a1.get_ylim()[0] + 3, "début de l'attaque", color="#c62828", fontsize=6.5)
a1.set_ylabel("°C"); a1.legend(loc="center left", frameon=False)
cols = {0: "#2e7d32", 1: "#f28e2b", 2: "#ef6c00", 3: "#6a1b9a", 4: "#1565c0", 5: "#111111"}
for row, key in ((2, "A"), (1, "B"), (0, "C")):
    tt = [d["t"] for d in D]; vv = [d[key] for d in D]
    a2.scatter(tt, [row] * len(tt), c=[cols[v] for v in vv], s=5, marker="s", linewidths=0)
a2.set_yticks([0, 1, 2]); a2.set_yticklabels(["C", "B", "A"]); a2.set_ylim(-0.6, 2.6); a2.set_xlabel("temps (s)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="s", ls="", color=cols[k], ms=4, label=l) for k, l in ((0, "normal"), (1, "P1 surchauffe"), (5, "données suspectes"))]
a2.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.45), frameon=False)
fig.tight_layout(); fig.savefig("figures/fig3_scenario.png", dpi=300); plt.close(fig)
print("figures écrites")
