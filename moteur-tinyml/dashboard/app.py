"""Tableau de bord interactif : choisir une panne et une attaque, voir réagir les configurations A, B et C.

Lancement (depuis la racine du projet) : streamlit run dashboard/app.py
Le tableau de bord simule le scénario localement (aucun courtier requis). Pour la chaîne MQTT complète,
voir iot/ et le README.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.engine import LABELS, EdgeEngine  # noqa: E402
from sim.motor import CHANNELS  # noqa: E402
from sim.scenario import run_scenario  # noqa: E402

RESULTS = os.environ.get("EDGE_RESULTS", "results/s0")
COLORS = {0: "#2e7d32", 1: "#d84315", 2: "#ef6c00", 3: "#6a1b9a", 4: "#1565c0", 5: "#c62828"}
SHORT = ["Normal", "P1 Surchauffe", "P2 Surcharge", "P3 Sous-vitesse", "P4 Vibrations", "Données suspectes"]
UNITS = ["V", "A", "tr/min", "°C", "g"]
NAMES = ["Tension", "Courant", "Vitesse", "Température", "Vibrations"]

st.set_page_config(page_title="TinyML résilient — moteur IoT", layout="wide")


@st.cache_resource
def engine():
    return EdgeEngine(RESULTS)


@st.cache_data
def simulate(**kw):
    return run_scenario(**kw)


# ----------------------------------------------------------------------------- barre latérale
st.sidebar.title("Scénario")
fault = st.sidebar.selectbox("Panne physique réelle", range(5), format_func=lambda k: SHORT[k])
fault_onset = st.sidebar.slider("Début de la panne (s)", 10, 100, 30, disabled=fault in (0, 1),
                                help="P1 (surchauffe) est présente dès le début de l'essai.")
attack = st.sidebar.selectbox("Attaque sur les données", range(6),
                              format_func=lambda k: ["Aucune", "C1 Injection", "C2 Courant modifié",
                                                     "C3 Température figée", "C4 Rejeu", "C5 Bruit"][k])
variant = st.sidebar.radio("Position de l'attaquant", ["compromis", "interception"], disabled=attack == 0,
                           format_func=lambda v: {"compromis": "Nœud capteurs compromis (messages signés)",
                                                  "interception": "Interception sur le canal (sans la clé)"}[v])
attack_onset = st.sidebar.slider("Début de l'attaque (s)", 10, 100, 50, disabled=attack == 0)
noisy = st.sidebar.checkbox("Rejeu bruité (évite la détection des doublons)", disabled=attack != 4)
seed = st.sidebar.number_input("Moteur (graine)", 1, 9999, 7)
live = st.sidebar.checkbox("Lecture en temps réel (×10)", value=False)
st.sidebar.caption(f"Modèle Edge : `{RESULTS}`")

st.title("Détection d'anomalies TinyML résiliente aux attaques sur les données")
st.caption("Moteur à courant continu simulé · nœud capteurs signé (HMAC + compteur) · Edge : A = seuils, "
           "B = TinyML INT8, C = TinyML + intégrité + cohérence physique")

s = simulate(seed=int(seed), fault=fault, fault_onset_s=float(fault_onset), attack=attack, variant=variant,
             attack_onset_s=float(attack_onset), noisy_replay=noisy)
eng = engine()
eng.commission(s["commissioning"])
dec = []
for k in range(len(s["received"])):
    r = eng.push(s["received"][k], s["accepted"][k], s["watermark"][k])
    if r:
        r["verite"] = int(s["y_phys"][k])
        dec.append(r)
D = pd.DataFrame(dec)
t = np.arange(len(s["received"])) / 10


def figure(upto: float):
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[1, 1, 1, 1, 1, 1.4],
                        subplot_titles=[f"{n} ({u})" for n, u in zip(NAMES, UNITS)] + ["Diagnostic"])
    m = t <= upto
    for c in range(5):
        fig.add_trace(go.Scatter(x=t[m], y=s["true"][m, c], name="réel", line=dict(color="#9e9e9e", dash="dot", width=1),
                                 showlegend=c == 0), row=c + 1, col=1)
        fig.add_trace(go.Scatter(x=t[m], y=s["received"][m, c], name="reçu par l'Edge",
                                 line=dict(color="#1f4e79", width=1.4), showlegend=c == 0), row=c + 1, col=1)
    d = D[D.t <= upto]
    for row_y, cfg in ((3, "Vérité"), (2, "A"), (1, "B"), (0, "C")):
        vals = d["verite"] if cfg == "Vérité" else d[cfg]
        fig.add_trace(go.Scatter(x=d.t, y=[row_y] * len(d), mode="markers", showlegend=False,
                                 marker=dict(color=[COLORS[v] for v in vals], size=7, symbol="square"),
                                 text=[SHORT[v] for v in vals], hovertemplate=f"{cfg} : %{{text}}<extra></extra>"),
                      row=6, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["C", "B", "A", "Vérité"], row=6, col=1, range=[-0.6, 3.6])
    for x0, lab in ((s["attack_onset_s"], "attaque"),):
        if x0 is not None and x0 <= upto:
            for rr in range(1, 7):
                fig.add_vline(x=x0, line=dict(color="#c62828", dash="dash", width=1), row=rr, col=1)
    fig.update_xaxes(range=[0, t[-1]], title_text="temps (s)", row=6, col=1)
    fig.update_layout(height=900, margin=dict(l=40, r=20, t=40, b=40), legend=dict(orientation="h", y=1.03))
    return fig


def last_status(upto: float):
    d = D[D.t <= upto]
    if d.empty:
        return
    row = d.iloc[-1]
    cols = st.columns(4)
    for col, (name, key) in zip(cols, (("Vérité", "verite"), ("A — seuils", "A"), ("B — TinyML", "B"),
                                       ("C — TinyML protégé", "C"))):
        v = int(row[key])
        col.markdown(f"<div style='padding:10px;border-radius:8px;background:{COLORS[v]}22;border:2px solid {COLORS[v]}'>"
                     f"<b>{name}</b><br><span style='font-size:1.3em'>{SHORT[v]}</span></div>", unsafe_allow_html=True)


status = st.empty()
plot = st.empty()
if live:
    for upto in np.arange(2, t[-1] + 0.5, 0.5):
        with status.container():
            last_status(upto)
        plot.plotly_chart(figure(upto), width="stretch", key=f"p{upto}")
        time.sleep(0.05)
else:
    with status.container():
        last_status(t[-1])
    plot.plotly_chart(figure(t[-1]), width="stretch")

# ----------------------------------------------------------------------------- synthèse
st.subheader("Synthèse du scénario")
after = D[D.t >= (s["attack_onset_s"] or 0)] if attack else D
truth = after["verite"].values
c1, c2, c3 = st.columns(3)
for col, cfg in zip((c1, c2, c3), ("A", "B", "C")):
    v = after[cfg].values
    if attack:
        ok = (v == 5) | (v == truth)
        col.metric(f"Configuration {cfg}", f"{100 * ok.mean():.0f} % de décisions sûres",
                   help="Après le début de l'attaque : décision sûre = « données suspectes » ou diagnostic correct.")
    else:
        col.metric(f"Configuration {cfg}", f"{100 * np.mean(v == truth):.0f} % de diagnostics corrects")
if attack:
    hit = D[(D.t >= s["attack_onset_s"]) & (D.C == 5)]
    st.write(f"Délai de détection par C : **{hit.t.iloc[0] - s['attack_onset_s']:.1f} s**" if len(hit)
             else "C n'a pas détecté cette attaque (voir les limites dans le README).")
    flags = pd.DataFrame(list(D[D.t >= s["attack_onset_s"]]["controles"]))
    st.write("Contrôles déclenchés après le début de l'attaque (fraction des fenêtres) :")
    st.bar_chart(flags.mean().rename("fraction"))
    st.write(f"Couche 1 (HMAC + compteur) : {100 * (1 - D[D.t >= s['attack_onset_s']]['couche1_ok'].mean()):.0f} % "
             f"des fenêtres rejetées. Détails de l'attaque : `{json.dumps(s['info'], ensure_ascii=False)}`")

with st.expander("Résultats agrégés (5 graines)"):
    p = "results/summary.json"
    if os.path.exists(p):
        st.json(json.load(open(p)))
    else:
        st.write("Lancer `bash experiments/run_all.sh` pour produire results/summary.json.")
