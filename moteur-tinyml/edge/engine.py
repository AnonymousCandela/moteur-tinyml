"""Nœud Edge en flux continu : reçoit les messages un par un et produit, toutes les 0,5 s, les décisions des
configurations A, B et C. Utilisé par le tableau de bord et par la chaîne MQTT (iot/edge_node.py).
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from edge.model import Int8Model, Thresholds
from edge.protection import CHECKS, DUP_K, WM_HORIZON, Consistency, identify, residuals, wm_agreement
from edge.windows import STRIDE, W
from sim.motor import CLASSES

LABELS = CLASSES + ["donnees_suspectes"]
SUSPECT = 5


class EdgeEngine:
    def __init__(self, results_dir: str):
        self.model = Int8Model(open(os.path.join(results_dir, "model_int8.tflite"), "rb").read())
        st = np.load(os.path.join(results_dir, "edge_state.npz"))
        self.mu, self.sd = st["mu"], st["sd"]
        s = json.load(open(os.path.join(results_dir, "seuils.json")))
        self.A = Thresholds()
        self.A.t = {int(k): v for k, v in s["A"].items()}
        c = s["couche2"]
        self.L2 = Consistency()
        self.L2.elec, self.L2.dT, self.L2.vib, self.L2.wm = c["elec"], c["dT"], c["vib"], c["wm"]
        self.L2.noise_lo, self.L2.noise_hi = np.array(c["noise_lo"]), np.array(c["noise_hi"])
        self.params = None

    def commission(self, healthy: np.ndarray):
        """Mise en service : identification des paramètres du moteur sur 30 s saines."""
        self.params = identify(healthy)
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=max(W, WM_HORIZON))
        self.ok = deque(maxlen=W)
        self.wm = deque(maxlen=WM_HORIZON)
        self.seen, self.n = set(), 0
        self.dup_recent = deque(maxlen=W)

    def push(self, sample, accepted: bool, watermark: float):
        """Un message reçu. Renvoie un dict de décisions toutes les STRIDE échantillons (sinon None)."""
        x = np.asarray(sample, dtype=np.float32)
        self.buf.append(x)
        self.ok.append(bool(accepted))
        self.wm.append(float(watermark))
        q = np.round(np.array(self.buf)[-DUP_K:], 3)
        dup = False
        if len(self.buf) >= DUP_K:
            h = q.tobytes()
            dup = h in self.seen
            self.seen.add(h)
        self.dup_recent.append(dup)
        self.n += 1
        if self.n < W or (self.n - W) % STRIDE:
            return None
        win = np.array(self.buf)[-W:][None]
        hist = np.array(self.buf)
        agree = wm_agreement(hist[:, 0][None], np.array(self.wm)[None])
        R = residuals(win, self.params, agree)
        flags = self.L2.flags(R, np.array([any(self.dup_recent)]))[0]
        l1 = all(self.ok)
        xn = ((win - self.mu) / self.sd).reshape(1, -1).astype(np.float32)
        logits = self.model.raw(xn)[0]
        b = int(logits.argmax())
        a = int(self.A.predict(win)[0])
        suspect = (not l1) or bool(flags.any())
        return dict(t=self.n / 10, A=a, B=b, C=SUSPECT if suspect else b, couche1_ok=l1,
                    controles={k: bool(v) for k, v in zip(CHECKS, flags)}, logits=logits.tolist())
