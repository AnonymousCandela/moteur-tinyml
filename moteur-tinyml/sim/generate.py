"""Génère le jeu de données : essais propres, en panne, attaqués et combinés, plus des moteurs hors plage.

Usage : python -m sim.generate --out data/dataset_s0.npz --seed 0
Le fichier contient, pour chaque essai : la série reçue par l'Edge (après attaque), la vérité physique,
l'étiquette d'attaque par échantillon, l'acceptation par la couche 1 (HMAC + compteur), le tatouage connu,
et 30 s de mise en service saine du même moteur (pour l'identification des paramètres par l'Edge).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .attacks import manipulate, transmit
from .motor import FS, RunSpec, sample_params, sense, simulate

# composition des essais (nombre par scénario)
CLEAN = {0: 170, 1: 60, 2: 60, 3: 60, 4: 60}                       # panne -> nombre (sans attaque)
ATTACKED_NORMAL = {1: 30, 2: 20, 3: 20, 4: 30, 5: 20}              # attaque sur moteur sain
COMBINED = [(1, 3, 20), (1, 4, 15), (2, 2, 20), (2, 4, 15)]        # (panne, attaque, nombre) : masquage
OOD = 100                                                           # moteurs hors plage, sans attaque


def _spec(rng, seed, fault, outside=False):
    sev = {0: 0.0, 1: rng.uniform(1.6, 2.4), 2: rng.uniform(0.4, 1.2), 3: rng.uniform(0.2, 0.4),
           4: rng.uniform(0.25, 0.7)}[fault]
    onset = 0.0 if fault in (0, 1) else rng.uniform(20, 60)
    return RunSpec(seed=seed, params=sample_params(rng, outside=outside), fault=fault, fault_onset_s=onset,
                   severity=sev, load_base=rng.uniform(0.12, 0.25), t_amb=rng.uniform(15, 35))


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    plan = []  # (fault, attack, group)
    for f, k in CLEAN.items():
        plan += [(f, 0, "propre")] * k
    for a, k in ATTACKED_NORMAL.items():
        plan += [(0, a, "attaque")] * k
    for f, a, k in COMBINED:
        plan += [(f, a, "combine")] * k
    plan += [(int(rng.integers(0, 5)), 0, "hors_plage") for _ in range(OOD)]

    specs = [_spec(rng, seed * 100_000 + j, f, outside=(g == "hors_plage")) for j, (f, a, g) in enumerate(plan)]
    t0 = time.time()
    real = simulate(specs, with_fault=True)
    twin = simulate(specs, with_fault=False)
    print(f"simulation physique : {len(specs)} essais × 2 en {time.time() - t0:.1f} s")

    seeds = [s.seed for s in specs]
    X_real = sense(real["true"], seeds, salt=0).astype(np.float32)
    X_twin = sense(twin["true"], seeds, salt=1).astype(np.float32)
    n = X_real.shape[1]

    X_rx = np.empty_like(X_real)
    att_lab = np.zeros((len(specs), n), dtype=np.int8)
    l1_ok = np.zeros((len(specs), n), dtype=bool)
    meta = []
    t0 = time.time()
    for j, ((f, a, g), sp) in enumerate(zip(plan, specs)):
        r = np.random.default_rng(sp.seed + 1)
        key = r.bytes(32)
        variant = "aucune" if a == 0 else r.choice(["interception", "compromis"])
        onset = int(r.uniform(20, 60) * FS) if a else n
        noisy = bool(a == 4 and variant == "compromis" and r.random() < 0.5)
        Y, info = manipulate(X_real[j], a, onset, r, replay_src=X_twin[j], noisy_replay=noisy)
        replay_off = None
        if a == 4:
            L = int(info["longueur_s"] * FS)
            replay_off = np.zeros(n, dtype=int)
            replay_off[onset:] = np.arange(n - onset) % L
        X_rx[j] = Y
        att_lab[j, onset:] = a
        l1_ok[j] = transmit(X_real[j], Y, a, onset, variant, key, replay_off)
        meta.append(dict(run=j, groupe=g, panne=int(f), attaque=int(a), variante=str(variant),
                         debut_panne_s=sp.fault_onset_s, severite=sp.severity, debut_attaque_s=onset / FS,
                         params={k: round(float(v), 6) for k, v in sp.params.items()}, **info))
    print(f"canal + HMAC : {time.time() - t0:.1f} s")

    # découpage par essai : propres -> 60/20/20 stratifié ; attaqués/combinés -> test ; hors plage -> test_ood
    split = np.empty(len(plan), dtype=object)
    for f in CLEAN:
        idx = [j for j, (ff, a, g) in enumerate(plan) if g == "propre" and ff == f]
        idx = rng.permutation(idx)
        a_, b_ = int(0.6 * len(idx)), int(0.8 * len(idx))
        split[idx[:a_]], split[idx[a_:b_]], split[idx[b_:]] = "train", "val", "test"
    for j, (f, a, g) in enumerate(plan):
        if g in ("attaque", "combine"):
            split[j] = "test"
        elif g == "hors_plage":
            split[j] = "test_ood"
    for m, s in zip(meta, split):
        m["split"] = s

    return dict(X=X_rx, X_clean=X_real, y_phys=real["labels"], y_att=att_lab, l1_ok=l1_ok,
                watermark=real["watermark"].astype(np.float32), commissioning=X_twin[:, : 30 * FS],
                split=np.array(split, dtype="U10"), meta=np.array(json.dumps(meta, ensure_ascii=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset_s0.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d = build(a.seed)
    np.savez_compressed(a.out, **d)
    sp = d["split"]
    print(f"{a.out} : {len(sp)} essais | train={np.sum(sp == 'train')} val={np.sum(sp == 'val')} "
          f"test={np.sum(sp == 'test')} test_ood={np.sum(sp == 'test_ood')}")


if __name__ == "__main__":
    main()
