"""Fenêtrage commun à l'entraînement, à l'évaluation et au nœud Edge."""
from __future__ import annotations

import json

import numpy as np

W = 20        # échantillons par fenêtre (2 s à 10 Hz)
STRIDE = 5    # une décision toutes les 0,5 s


def load(path: str) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def window_index(n: int) -> np.ndarray:
    """Indices de fin de fenêtre (inclus)."""
    return np.arange(W - 1, n, STRIDE)


def make_windows(d: dict, runs: np.ndarray, source: str = "X") -> dict:
    """Fenêtres (M, W, 5) + étiquettes de la dernière position + infos de suivi."""
    X = d[source]
    ends = window_index(X.shape[1])
    xs, yp, ya, ok, run_id, end_id = [], [], [], [], [], []
    for r in runs:
        for e in ends:
            xs.append(X[r, e - W + 1 : e + 1])
            ok.append(d["l1_ok"][r, e - W + 1 : e + 1].all())
        yp.append(d["y_phys"][r, ends])
        ya.append(d["y_att"][r, ends])
        run_id.append(np.full(len(ends), r))
        end_id.append(ends)
    return dict(x=np.stack(xs).astype(np.float32), y_phys=np.concatenate(yp).astype(int),
                y_att=np.concatenate(ya).astype(int), l1_ok=np.array(ok), run=np.concatenate(run_id),
                end=np.concatenate(end_id))


def runs_of(d: dict, split: str, group: str | None = None) -> np.ndarray:
    return np.array([m["run"] for m in d["meta"] if m["split"] == split and (group is None or m["groupe"] == group)])
