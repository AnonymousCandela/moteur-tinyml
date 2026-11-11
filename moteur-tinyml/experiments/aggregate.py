"""Agrège les metrics.json de plusieurs graines : moyenne ± écart type, tableaux prêts pour l'article.

Usage : python -m experiments.aggregate results/s* --out results/summary.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return dict(moyenne=float(v.mean()), ecart_type=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def fmt(d, pct=False, nd=3):
    k = 100 if pct else 1
    return f"{d['moyenne'] * k:.{1 if pct else nd}f} ± {d['ecart_type'] * k:.{1 if pct else nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/summary.json")
    a = ap.parse_args()
    R = [json.load(open(os.path.join(p, "metrics.json"))) for p in a.dirs if os.path.exists(os.path.join(p, "metrics.json"))]
    g = lambda f: [f(r) for r in R]  # noqa: E731
    S = {
        "graines": [r["seed"] for r in R],
        "f1_macro_propre": {c: ms(g(lambda r, c=c: r["propre"]["f1_macro"][c])) for c in ("A", "B_float", "B_int8", "C")},
        "faux_positifs_normal": {c: ms(g(lambda r, c=c: r["propre"]["faux_positifs_normal"][c])) for c in ("A", "B_int8", "C")},
        "delta_f1_quantification": ms(g(lambda r: r["delta_f1_quantification"])),
        "fausse_suspicion_C_propre": ms(g(lambda r: r["propre"]["fausse_suspicion_C"])),
        "fausse_suspicion_C_par_classe": {k: ms(g(lambda r, k=k: r["propre"]["fausse_suspicion_C_par_classe"][k]))
                                          for k in R[0]["propre"]["fausse_suspicion_C_par_classe"]},
        "detection_C_globale": ms(g(lambda r: r["attaques_global"]["detection_C"])),
        "B_trompe_sous_attaque": ms(g(lambda r: r["attaques_global"]["B_trompe"])),
        "A_trompe_sous_attaque": ms(g(lambda r: r["attaques_global"]["A_trompe"])),
        "masquage": {k: ms(g(lambda r, k=k: r["masquage"][k])) for k in
                     ("rappel_panne_A", "rappel_panne_B", "rappel_panne_B_sans_attaque", "C_sur", "C_dit_normal", "B_dit_normal")},
        "hors_plage": {"A": ms(g(lambda r: r["hors_plage"]["f1_macro"]["A"])),
                       "B_int8": ms(g(lambda r: r["hors_plage"]["f1_macro"]["B_int8"])),
                       "fausse_suspicion_C": ms(g(lambda r: r["hors_plage"]["fausse_suspicion_C"]))},
        "par_attaque": {},
        "embarque": R[0]["embarque"],
    }
    keys = sorted({k for r in R for k in r["attaques"]})
    for k in keys:
        S["par_attaque"][k] = {
            "detection_C": ms([r["attaques"][k]["detection_C"] for r in R if k in r["attaques"]]),
            "delai_s": ms([r["attaques"][k]["delai_moyen_s"] for r in R if k in r["attaques"]]),
            "B_trompe": ms([r["attaques"][k]["B_trompe"] for r in R if k in r["attaques"]]),
        }
    json.dump(S, open(a.out, "w"), indent=2, ensure_ascii=False)

    print(f"Graines : {S['graines']}\n")
    print("Données propres (F1 macro)        A = %s | B float = %s | B INT8 = %s | C = %s" % tuple(
        fmt(S["f1_macro_propre"][c]) for c in ("A", "B_float", "B_int8", "C")))
    print("Faux positifs (fenêtres normales) A = %s %% | B INT8 = %s %% | C = %s %%" % tuple(
        fmt(S["faux_positifs_normal"][c], pct=True) for c in ("A", "B_int8", "C")))
    print("Écart F1 float → INT8             %s" % fmt(S["delta_f1_quantification"], nd=4))
    print("Fausse suspicion C (propre)       %s %%" % fmt(S["fausse_suspicion_C_propre"], pct=True))
    print("Détection des attaques par C      %s %%" % fmt(S["detection_C_globale"], pct=True))
    print("B trompé sous attaque             %s %%   (A : %s %%)" % (fmt(S["B_trompe_sous_attaque"], pct=True),
                                                                  fmt(S["A_trompe_sous_attaque"], pct=True)))
    m = S["masquage"]
    print("Masquage — rappel panne B         %s %% (sans attaque : %s %%) | C sûr : %s %%" % (
        fmt(m["rappel_panne_B"], pct=True), fmt(m["rappel_panne_B_sans_attaque"], pct=True), fmt(m["C_sur"], pct=True)))
    print("Hors plage (F1 macro)             A = %s | B INT8 = %s | fausse suspicion C = %s %%" % (
        fmt(S["hors_plage"]["A"]), fmt(S["hors_plage"]["B_int8"]), fmt(S["hors_plage"]["fausse_suspicion_C"], pct=True)))
    print("\nPar attaque (détection C %, délai s, B trompé %) :")
    for k, v in S["par_attaque"].items():
        print(f"  {k:32s} {fmt(v['detection_C'], pct=True):>14s}  {fmt(v['delai_s'], nd=1):>12s}  {fmt(v['B_trompe'], pct=True):>14s}")
    print(f"\nEmbarqué : {S['embarque']}")


if __name__ == "__main__":
    main()
