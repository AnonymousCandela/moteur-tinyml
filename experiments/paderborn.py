"""Validation de la couche de cohérence sur des mesures RÉELLES : jeu « Electric Motor Temperature »
(Université de Paderborn, moteur synchrone à aimants permanents, 185 h, 2 Hz ; Kaggle : wkirgsn/electric-motor-temperature).

Trois contrôles sont évalués sur la température du bobinage statorique (stator_winding) :

1. Résidu ÉLECTRIQUE (v1) — équations de tension d-q avec R(T) :
       u_q ≈ R(T)·i_q + k1·n·i_d + k2·n      u_d ≈ R(T)·i_d − k3·n·i_q
   Résultat v1 sur les vraies données : AUC ≈ 0,45, inutile (R pèse trop peu face aux termes de vitesse).

2. Observateur THERMIQUE (v2) — modèle du premier ordre appris par moindres carrés sur les profils d'entraînement :
       T[k+1] − T[k] = a·(i_d² + i_q²) + b·n + c·n² + d·(T[k] − T_coolant) + e·(T[k] − T_amb) + f
   L'Edge fait tourner ce modèle en parallèle (observateur de Luenberger de gain L) et compare la température
   REÇUE à la température PRÉDITE par la physique ; une température figée ou rejouée s'en écarte dès que les
   conditions de fonctionnement changent.

3. Valeur FIGÉE (v2) — température strictement constante (écart type des différences nul) sur 30 s.
   En v1, ce contrôle travaillait sur 2 s et son seuil tombait à 0 sur ces données lentes : il ne se déclenchait jamais.

Découpage par profil : 60 % entraînement du modèle, 15 % calibration des seuils (données propres), 25 % test.
Les attaques sont injectées sur la seconde moitié de chaque profil de test. On rapporte aussi la détection
restreinte aux fenêtres « à impact », où la valeur falsifiée s'écarte d'au moins 10 °C de la vraie : une attaque
qui ne modifie pas la valeur ne masque rien.

Usage : python -m experiments.paderborn --csv data/measures_v2.csv --out results/paderborn.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALPHA = 0.0039
W = 4                 # fenêtre de 2 s (2 Hz)
FREEZE = 60           # 30 s pour le contrôle de valeur figée
GAIN = 0.02           # gain de l'observateur thermique (constante de temps ≈ 25 s)
IMPACT_C = 10.0       # écart minimal (°C) pour qu'une falsification « masque » quelque chose


# ----------------------------------------------------------------------------- contrôle électrique (v1)
def elec_design(g, T):
    r = 1 + ALPHA * (T - 20)
    n = g["motor_speed"].values
    Aq = np.column_stack([g["i_q"].values * r, n * g["i_d"].values, n])
    Ad = np.column_stack([g["i_d"].values * r, n * g["i_q"].values])
    return Aq, Ad


def elec_fit(df):
    Aq, Ad = elec_design(df, df["stator_winding"].values)
    cq, *_ = np.linalg.lstsq(Aq, df["u_q"].values, rcond=None)
    cd, *_ = np.linalg.lstsq(Ad, df["u_d"].values, rcond=None)
    return cq, cd


def elec_residual(g, T, cq, cd):
    Aq, Ad = elec_design(g, T)
    return np.abs(g["u_q"].values - Aq @ cq) + np.abs(g["u_d"].values - Ad @ cd)


# ----------------------------------------------------------------------------- observateur thermique (v2)
def thermal_features(g, T):
    i2 = g["i_d"].values ** 2 + g["i_q"].values ** 2
    n = g["motor_speed"].values
    return np.column_stack([i2, n, n ** 2, T - g["coolant"].values, T - g["ambient"].values, np.ones(len(g))])


def thermal_fit(train_profiles):
    X, y = [], []
    for g in train_profiles:
        T = g["stator_winding"].values
        X.append(thermal_features(g, T)[:-1])
        y.append(np.diff(T))
    X, y = np.vstack(X), np.concatenate(y)
    sc = np.abs(X).max(0) + 1e-12            # mise à l'échelle pour un système bien conditionné
    th, *_ = np.linalg.lstsq(X / sc, y, rcond=None)
    return th / sc


def thermal_residual(g, T_rep, th):
    """Observateur : T̂[k+1] = T̂[k] + f(entrées, T̂[k]) + L·(T_reçue[k] − T̂[k]) ; résidu = T_reçue − T̂."""
    i2 = g["i_d"].values ** 2 + g["i_q"].values ** 2
    n = g["motor_speed"].values
    cool, amb = g["coolant"].values, g["ambient"].values
    a, b, c, d, e, f = th
    That = np.empty(len(T_rep))
    That[0] = T_rep[0]
    for k in range(len(T_rep) - 1):
        dT = a * i2[k] + b * n[k] + c * n[k] ** 2 + d * (That[k] - cool[k]) + e * (That[k] - amb[k]) + f
        That[k + 1] = That[k] + dT + GAIN * (T_rep[k] - That[k])
    return np.abs(T_rep - That)


def frozen(T_rep):
    """Vrai si la température est restée strictement constante sur les FREEZE derniers échantillons."""
    d = np.r_[np.nan, np.diff(T_rep)]
    s = pd.Series(np.abs(d)).rolling(FREEZE - 1, min_periods=FREEZE - 1).max().values
    return np.nan_to_num(s, nan=1.0) == 0.0


def windows(x, agg=np.mean):
    m = len(x) // W
    return agg(x[: m * W].reshape(m, W), axis=1)


# ----------------------------------------------------------------------------- évaluation
def scores(g, T_rep, cq, cd, th):
    return dict(elec=windows(elec_residual(g, T_rep, cq, cd)),
                therm=windows(thermal_residual(g, T_rep, th)),
                fig=windows(frozen(T_rep).astype(float), agg=np.max) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/paderborn.json")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    pids = np.array(sorted(df["profile_id"].unique()))
    rng = np.random.default_rng(0)
    perm = rng.permutation(pids)
    n_tr, n_va = int(0.60 * len(perm)), int(0.15 * len(perm))
    tr_ids, va_ids, te_ids = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    prof = {p: g.reset_index(drop=True) for p, g in df.groupby("profile_id")}

    cq, cd = elec_fit(df[df.profile_id.isin(tr_ids)])
    th = thermal_fit([prof[p] for p in tr_ids])

    # seuils calibrés sur les profils de validation (propres uniquement)
    va = [scores(prof[p], prof[p]["stator_winding"].values, cq, cd, th) for p in va_ids]
    thr_e = np.percentile(np.concatenate([s["elec"] for s in va]), 99.9) * 1.25
    thr_t = np.percentile(np.concatenate([s["therm"] for s in va]), 99.9) * 1.25

    clean = {"elec": [], "therm": [], "fig": []}
    att = {k: {"elec": [], "therm": [], "fig": [], "impact": []} for k in ("C3_temp_figee", "C4_rejeu")}
    for p in te_ids:
        g = prof[p]
        if len(g) < 200 * W:
            continue
        T = g["stator_winding"].values
        on = (len(g) // 2) // W * W
        s = scores(g.iloc[:on].reset_index(drop=True), T[:on], cq, cd, th)   # moitié propre
        for k in clean:
            clean[k].append(s[k])
        fakes = {"C3_temp_figee": np.r_[T[:on], np.full(len(T) - on, T[:on].mean())],
                 "C4_rejeu": np.r_[T[:on], np.resize(T[: min(on, 1200)], len(T) - on)]}
        for name, Tf in fakes.items():
            s = scores(g, Tf, cq, cd, th)                     # l'observateur tourne sur tout le profil
            m0 = on // W
            for k in ("elec", "therm", "fig"):
                att[name][k].append(s[k][m0:])
            att[name]["impact"].append(windows(np.abs(Tf - T))[m0:] >= IMPACT_C)

    C = {k: np.concatenate(v) for k, v in clean.items()}
    flag_clean = (C["elec"] > thr_e) | (C["therm"] > thr_t) | C["fig"]
    out = dict(
        version=2,
        profils=dict(entrainement=len(tr_ids), calibration=len(va_ids), test=sorted(int(p) for p in te_ids)),
        modele_thermique=dict(zip(["a_i2", "b_n", "c_n2", "d_coolant", "e_ambient", "f_biais"], map(float, th))),
        seuils=dict(electrique=float(thr_e), thermique_C=float(thr_t)),
        fenetres_propres=int(len(C["elec"])),
        fausse_alerte=dict(electrique=float(np.mean(C["elec"] > thr_e)), thermique=float(np.mean(C["therm"] > thr_t)),
                           figee=float(np.mean(C["fig"])), totale=float(np.mean(flag_clean))),
        attaques={},
    )
    for name, v in att.items():
        A = {k: np.concatenate(x) for k, x in v.items()}
        y = np.r_[np.zeros(len(C["elec"])), np.ones(len(A["elec"]))]
        tot = (A["elec"] > thr_e) | (A["therm"] > thr_t) | A["fig"]
        imp = A["impact"]
        out["attaques"][name] = dict(
            fenetres=int(len(tot)), fenetres_a_impact=int(imp.sum()),
            auc_electrique=float(roc_auc_score(y, np.r_[C["elec"], A["elec"]])),
            auc_thermique=float(roc_auc_score(y, np.r_[C["therm"], A["therm"]])),
            detection_electrique=float(np.mean(A["elec"] > thr_e)),
            detection_thermique=float(np.mean(A["therm"] > thr_t)),
            detection_figee=float(np.mean(A["fig"])),
            detection_totale=float(np.mean(tot)),
            detection_totale_si_impact=float(np.mean(tot[imp])) if imp.any() else None,
        )
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
