"""Validation de la couche de cohérence sur des mesures RÉELLES : jeu « Electric Motor Temperature »
(Université de Paderborn, moteur synchrone à aimants permanents, 185 h, Kaggle : wkirgsn/electric-motor-temperature).

Le moteur réel est un PMSM, pas un moteur à courant continu : on utilise donc ses équations de tension en
régime permanent dans le repère d-q, où la résistance statorique dépend de la température du bobinage :
    u_q ≈ R(T)·i_q + k1·n·i_d + k2·n          u_d ≈ R(T)·i_d − k3·n·i_q          R(T) = R0·[1 + α(T − 20)]
Les coefficients sont ajustés par moindres carrés sur les profils d'entraînement (données propres). On injecte
ensuite, sur les profils de test, une température figée (C3) et un rejeu (C4), et on mesure si les résidus et
le contrôle de bruit les séparent des données propres (AUC, taux de détection au seuil calibré sur le propre).

Usage : python -m experiments.paderborn --csv measures_v2.csv --out results/paderborn.json
Prudence : les tensions du jeu sont des consignes de l'onduleur ; le couplage R(T) est faible devant les termes
de vitesse à haut régime. Le résultat, quel qu'il soit, est à rapporter tel quel dans l'article.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALPHA = 0.0039
FS = 2          # le jeu est échantillonné à 2 Hz
W = 4           # fenêtre de 2 s


def design(df, T):
    r = 1 + ALPHA * (T - 20)
    n = df["motor_speed"].values
    Aq = np.column_stack([df["i_q"].values * r, n * df["i_d"].values, n])
    Ad = np.column_stack([df["i_d"].values * r, n * df["i_q"].values])
    return Aq, Ad


def fit(df):
    Aq, Ad = design(df, df["stator_winding"].values)
    cq, *_ = np.linalg.lstsq(Aq, df["u_q"].values, rcond=None)
    cd, *_ = np.linalg.lstsq(Ad, df["u_d"].values, rcond=None)
    return cq, cd


def window_scores(df, T_reported, cq, cd):
    Aq, Ad = design(df, T_reported)
    res = np.abs(df["u_q"].values - Aq @ cq) + np.abs(df["u_d"].values - Ad @ cd)
    m = len(res) // W
    r = res[: m * W].reshape(m, W).mean(1)
    dT = np.diff(T_reported[: m * W].reshape(m, W), axis=1)
    noise = dT.std(1)
    return r, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/paderborn.json")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    pids = sorted(df["profile_id"].unique())
    rng = np.random.default_rng(0)
    test_ids = set(rng.choice(pids, size=max(1, len(pids) // 4), replace=False).tolist())
    train = df[~df.profile_id.isin(test_ids)]
    cq, cd = fit(train)

    clean_r, clean_n, att = [], [], {"C3_temp_figee": ([], []), "C4_rejeu": ([], [])}
    for pid in test_ids:
        g = df[df.profile_id == pid].reset_index(drop=True)
        if len(g) < 40 * W:
            continue
        T = g["stator_winding"].values
        r, n = window_scores(g, T, cq, cd); clean_r.append(r); clean_n.append(n)
        on = len(g) // 2
        Tf = T.copy(); Tf[on:] = T[: on].mean()                       # C3 : température figée
        r, n = window_scores(g.iloc[on:].reset_index(drop=True), Tf[on:], cq, cd)
        att["C3_temp_figee"][0].append(r); att["C3_temp_figee"][1].append(n)
        Tr = T.copy(); L = min(on, 600)                                # C4 : rejeu des températures passées
        Tr[on:] = np.resize(T[:L], len(T) - on)
        r, n = window_scores(g.iloc[on:].reset_index(drop=True), Tr[on:], cq, cd)
        att["C4_rejeu"][0].append(r); att["C4_rejeu"][1].append(n)

    cr, cn = np.concatenate(clean_r), np.concatenate(clean_n)
    thr_r, thr_n = np.percentile(cr, 99.9) * 1.25, np.percentile(cn, 0.1) / 1.25
    out = dict(profils_test=sorted(int(p) for p in test_ids), fenetres_propres=int(len(cr)),
               fausse_alerte=float(np.mean((cr > thr_r) | (cn < thr_n))), attaques={})
    for k, (rs, ns) in att.items():
        r, n = np.concatenate(rs), np.concatenate(ns)
        y = np.r_[np.zeros(len(cr)), np.ones(len(r))]
        out["attaques"][k] = dict(
            fenetres=int(len(r)),
            auc_residu_electrique=float(roc_auc_score(y, np.r_[cr, r])),
            detection_residu=float(np.mean(r > thr_r)),
            detection_bruit_fige=float(np.mean(n < thr_n)),
            detection_totale=float(np.mean((r > thr_r) | (n < thr_n))))
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
