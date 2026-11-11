"""Expérience principale : comparaison des configurations A (seuils), B (TinyML) et C (TinyML protégé).

Usage :
  python -m experiments.run_abc --data data/dataset_s0.npz --seed 0 --out results/s0 [--export firmware/main]
Produit results/sN/metrics.json (+ figures pour la graine 0, + modèles).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from edge.model import Int8Model, Normalizer, Thresholds, export_c, macs, to_int8, train_mlp
from edge.protection import CHECKS, Consistency, identify, run_residuals
from edge.windows import load, make_windows, runs_of
from sim.attacks import ATTACKS
from sim.motor import CLASSES

SUSPECT = 5


def f1m(y, p):
    return float(f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", default=None, help="dossier firmware/main pour l'export C")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = load(a.data)
    meta = d["meta"]

    # ------------------------------------------------------------------ fenêtres
    tr = make_windows(d, runs_of(d, "train"))
    va = make_windows(d, runs_of(d, "val"))
    te_clean = make_windows(d, runs_of(d, "test", "propre"))
    te_att = make_windows(d, np.concatenate([runs_of(d, "test", "attaque"), runs_of(d, "test", "combine")]))
    te_ood = make_windows(d, runs_of(d, "test_ood"))

    # ------------------------------------------------------------------ A : seuils
    A = Thresholds().fit(tr["x"], tr["y_phys"])

    # ------------------------------------------------------------------ B : MLP -> INT8
    norm = Normalizer().fit(tr["x"])
    mlp = train_mlp(norm(tr["x"]), tr["y_phys"], norm(va["x"]), va["y_phys"], seed=a.seed)
    blob = to_int8(mlp, norm(tr["x"]))
    B = Int8Model(blob)

    def predB(w):
        return B.predict(norm(w["x"]))

    # ------------------------------------------------------------------ C : couches 1 et 2
    params = {int(m["run"]): identify(d["commissioning"][m["run"]]) for m in meta}
    R_va, dup_va = run_residuals(d, va, params)
    layer2 = Consistency().fit(R_va)

    def predC(w, pB):
        R, dup = run_residuals(d, w, params)
        fl = layer2.flags(R, dup)
        susp = (~w["l1_ok"]) | fl.any(1)
        return np.where(susp, SUSPECT, pB), fl, susp

    res = dict(seed=a.seed, data=a.data)

    # ------------------------------------------------------------------ 1) données propres
    y = te_clean["y_phys"]
    pA, pB = A.predict(te_clean["x"]), predB(te_clean)
    pBf = mlp.predict(norm(te_clean["x"]), verbose=0).argmax(1)
    pC, flC, suspC = predC(te_clean, pB)
    res["propre"] = dict(
        n_fenetres=int(len(y)),
        f1_macro=dict(A=f1m(y, pA), B_float=f1m(y, pBf), B_int8=f1m(y, pB), C=f1m(y, pC)),
        f1_par_classe=dict(
            A=f1_score(y, pA, labels=list(range(5)), average=None, zero_division=0).round(4).tolist(),
            B_int8=f1_score(y, pB, labels=list(range(5)), average=None, zero_division=0).round(4).tolist()),
        faux_positifs_normal=dict(A=float(np.mean(pA[y == 0] != 0)), B_int8=float(np.mean(pB[y == 0] != 0)),
                                  C=float(np.mean(pC[y == 0] != 0))),
        fausse_suspicion_C=float(suspC.mean()),
        fausse_suspicion_C_par_classe={CLASSES[c]: float(suspC[y == c].mean()) for c in range(5)},
        fausse_suspicion_par_controle={k: float(flC[:, j].mean()) for j, k in enumerate(CHECKS)},
        confusion_B_int8=confusion_matrix(y, pB, labels=list(range(5))).tolist(),
    )
    res["delta_f1_quantification"] = res["propre"]["f1_macro"]["B_float"] - res["propre"]["f1_macro"]["B_int8"]

    # ------------------------------------------------------------------ 2) données attaquées
    ya, yp = te_att["y_att"], te_att["y_phys"]
    pA2, pB2 = A.predict(te_att["x"]), predB(te_att)
    pC2, fl2, susp2 = predC(te_att, pB2)
    pre = ya == 0                               # fenêtres avant le début de l'attaque
    res["avant_attaque_fausse_suspicion_C"] = float(susp2[pre].mean())
    run_info = {m["run"]: m for m in meta}
    by = {}
    for att in range(1, 6):
        for var in ("interception", "compromis"):
            m = (ya == att) & np.array([run_info[r]["variante"] == var for r in te_att["run"]])
            if not m.any():
                continue
            runs = np.unique(te_att["run"][m])
            delays = []
            for r in runs:
                rm = (te_att["run"] == r) & (ya > 0)
                hit = np.flatnonzero(susp2[rm])
                if len(hit):
                    delays.append((te_att["end"][rm][hit[0]] - te_att["end"][rm][0]) / 10 + 0.0)
            by[f"{ATTACKS[att]}|{var}"] = dict(
                essais=int(len(runs)), fenetres=int(m.sum()),
                detection_C=float(susp2[m].mean()),
                essais_detectes_C=f"{len(delays)}/{len(runs)}",
                delai_moyen_s=float(np.mean(delays)) if delays else None,
                B_trompe=float(np.mean(pB2[m] != yp[m])), A_trompe=float(np.mean(pA2[m] != yp[m])),
                controles={k: float(fl2[m][:, j].mean()) for j, k in enumerate(CHECKS)},
                couche1=float(np.mean(~te_att["l1_ok"][m])))
    res["attaques"] = by
    att = ya > 0
    res["attaques_global"] = dict(detection_C=float(susp2[att].mean()),
                                  B_trompe=float(np.mean(pB2[att] != yp[att])),
                                  A_trompe=float(np.mean(pA2[att] != yp[att])))

    # masquage : panne réelle + attaque -> rappel de la panne par A/B ; C « sûr » si suspect ou panne correcte
    comb = att & (yp > 0)
    res["masquage"] = dict(
        fenetres=int(comb.sum()),
        rappel_panne_A=float(np.mean(pA2[comb] == yp[comb])),
        rappel_panne_B=float(np.mean(pB2[comb] == yp[comb])),
        rappel_panne_B_sans_attaque=float(np.mean(pB[(y > 0) & np.isin(y, np.unique(yp[comb]))]
                                                  == y[(y > 0) & np.isin(y, np.unique(yp[comb]))])),
        C_sur=float(np.mean((pC2[comb] == SUSPECT) | (pC2[comb] == yp[comb]))),
        C_dit_normal=float(np.mean(pC2[comb] == 0)),
        B_dit_normal=float(np.mean(pB2[comb] == 0)))

    # ------------------------------------------------------------------ 3) moteurs hors plage
    yo = te_ood["y_phys"]
    pAo, pBo = A.predict(te_ood["x"]), predB(te_ood)
    _, _, suspO = predC(te_ood, pBo)
    res["hors_plage"] = dict(f1_macro=dict(A=f1m(yo, pAo), B_int8=f1m(yo, pBo)),
                             fausse_suspicion_C=float(suspO.mean()))

    # ------------------------------------------------------------------ 4) embarqué
    res["embarque"] = dict(parametres=int(mlp.count_params()), tflite_octets=len(blob), macs=macs(),
                           latence_estimee_us={"1_cycle_par_mac": round(macs() / 240, 1),
                                               "4_cycles_par_mac": round(4 * macs() / 240, 1)})
    res["seuils"] = dict(A=A.t, couche2=layer2.as_dict())
    res["duree_s"] = round(time.time() - t0, 1)

    open(os.path.join(a.out, "model_int8.tflite"), "wb").write(blob)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    if a.export:
        # fenêtres dorées : une par classe sur le test propre
        idx = [int(np.flatnonzero(y == c)[len(np.flatnonzero(y == c)) // 2]) for c in range(5)]
        res["sha256"] = export_c(a.export, B, norm, te_clean["x"][idx])
        np.savez(os.path.join(a.out, "edge_state.npz"), mu=norm.mu, sd=norm.sd)
        json.dump(dict(A=A.t, couche2=layer2.as_dict()), open(os.path.join(a.out, "seuils.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("propre", "attaques_global", "masquage", "hors_plage", "embarque")},
                     indent=1, ensure_ascii=False)[:4000])
    print(f"terminé en {res['duree_s']} s")


if __name__ == "__main__":
    main()
