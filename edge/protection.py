"""Couche 2 de l'Edge : cohérence physique des mesures (configuration C).

Principe : chaque contrôle doit rester dans sa plage pour un moteur RÉEL, sain OU en panne, et en sortir lorsque
les mesures sont falsifiées. Les seuils sont calibrés uniquement sur des fenêtres propres (normales et en panne).

Contrôles par fenêtre de 2 s :
  r_elec   |moyenne(V − R0·i·(1+α(T−20)) − Ke·ω)|   équation électrique ; couple la température à la résistance
  r_dT     max |ΔT| entre deux échantillons              inertie thermique : la température ne saute pas
  r_vib    moyenne(vib − (c0 + c1 (ω/ω_nom)²))           vibrations sous le plancher physique imposé par la vitesse
  r_noise  bruit par capteur : écart type des différences trop faible (valeur figée) ou écart absolu médian
           des différences trop fort (bruit injecté)
  r_wm     accord de signe entre ΔV mesuré et Δ tatouage sur les 5 dernières secondes (tatouage actif connu
           de l'Edge ; une séquence rejouée porte un ancien tatouage → accord ≈ 50 %)
  r_dup    fenêtre identique à une fenêtre déjà vue        rejeu exact
Les paramètres R0, Ke, c0, c1 sont identifiés par moindres carrés pendant la mise en service du moteur (30 s saines).
"""
from __future__ import annotations

import numpy as np

from sim.motor import ALPHA_CU, OMEGA_NOM, WM_CHIP

I_V, I_I, I_RPM, I_T, I_VIB = range(5)
CHECKS = ["r_elec", "r_dT", "r_vib", "r_noise_lo", "r_noise_hi", "r_wm", "r_dup"]


def identify(commissioning: np.ndarray) -> dict:
    """Identification des paramètres d'un moteur à partir de sa mise en service (n, 5)."""
    V, I, rpm, T, vib = commissioning.T
    w = rpm * 2 * np.pi / 60
    A = np.stack([I * (1 + ALPHA_CU * (T - 20)), w], axis=1)
    (R0, Ke), *_ = np.linalg.lstsq(A, V, rcond=None)
    Bv = np.stack([np.ones_like(w), (w / OMEGA_NOM) ** 2], axis=1)
    (c0, c1), *_ = np.linalg.lstsq(Bv, vib, rcond=None)
    return dict(R0=R0, Ke=Ke, c0=c0, c1=c1)


WM_HORIZON = 50   # échantillons (5 s) pour le contrôle du tatouage


def wm_agreement(v: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """v, wm : (M, H). Fraction des changements de tatouage dont le signe se retrouve dans ΔV."""
    dv, dw = np.diff(v, axis=1), np.diff(wm, axis=1)
    has = np.abs(dw) > 1e-6
    agree = (np.sign(dv) == np.sign(dw)) & has
    return np.where(has.sum(1) > 0, agree.sum(1) / np.maximum(has.sum(1), 1), 1.0)


def residuals(x: np.ndarray, p: dict, wm_agree: np.ndarray) -> np.ndarray:
    """x : (M, W, 5) fenêtres d'UN moteur ; wm_agree : (M,) accord de tatouage. Renvoie (M, 14) résidus."""
    V, I, rpm, T, vib = (x[..., k] for k in range(5))
    w = rpm * 2 * np.pi / 60
    r_elec = np.abs(np.mean(V - p["R0"] * I * (1 + ALPHA_CU * (T - 20)) - p["Ke"] * w, axis=1))
    r_dT = np.max(np.abs(np.diff(T, axis=1)), axis=1)
    r_vib = np.mean(vib - (p["c0"] + p["c1"] * (w / OMEGA_NOM) ** 2), axis=1)
    dx = np.diff(x, axis=1)
    sd = np.std(dx, axis=1)                                      # (M, 5) : 0 si la valeur est figée
    mad = np.median(np.abs(dx), axis=1)                          # (M, 5) : robuste aux sauts de charge
    return np.column_stack([r_elec, r_dT, r_vib, sd, mad, wm_agree])   # 3 + 5 + 5 + 1 = 14 colonnes


class Consistency:
    """Seuils de la couche 2, appris sur des fenêtres propres uniquement."""

    def fit(self, R: np.ndarray, q_hi: float = 99.9, q_lo: float = 0.1, margin: float = 1.25):
        self.elec = np.percentile(R[:, 0], q_hi) * margin
        self.dT = np.percentile(R[:, 1], q_hi) * margin
        self.vib = np.percentile(R[:, 2], q_lo) * margin            # négatif : plancher
        self.noise_lo = np.percentile(R[:, 3:8], q_lo, axis=0) / margin
        self.noise_hi = np.percentile(R[:, 8:13], q_hi, axis=0) * margin
        self.wm = min(np.percentile(R[:, 13], q_lo) - 0.05, 0.85)
        return self

    def flags(self, R: np.ndarray, dup: np.ndarray) -> np.ndarray:
        """(M, 7) booléens dans l'ordre de CHECKS."""
        return np.column_stack([
            R[:, 0] > self.elec,
            R[:, 1] > self.dT,
            R[:, 2] < self.vib,
            np.any(R[:, 3:8] < self.noise_lo, axis=1),
            np.any(R[:, 8:13] > self.noise_hi, axis=1),
            R[:, 13] < self.wm,
            dup,
        ])

    def as_dict(self):
        return dict(elec=float(self.elec), dT=float(self.dT), vib=float(self.vib), wm=float(self.wm),
                    noise_lo=self.noise_lo.tolist(), noise_hi=self.noise_hi.tolist())


DUP_K = 10   # longueur (échantillons) d'une sous-séquence considérée comme une empreinte


def duplicate_samples(stream: np.ndarray, k: int = DUP_K) -> np.ndarray:
    """Pour un flux reçu (n, 5) : vrai à l'instant t si les k derniers échantillons ont déjà été reçus à
    l'identique. Le bruit réel des capteurs ne se répète jamais ; une séquence rejouée, si."""
    seen, out = set(), np.zeros(len(stream), dtype=bool)
    q = np.round(stream, 3)
    for t in range(k - 1, len(stream)):
        h = q[t - k + 1 : t + 1].tobytes()
        out[t] = h in seen
        seen.add(h)
    return out


def run_residuals(d: dict, win: dict, params: dict, source: str = "X") -> tuple[np.ndarray, np.ndarray]:
    """Résidus et doublons pour toutes les fenêtres, moteur par moteur."""
    from .windows import W
    R = np.zeros((len(win["x"]), 14))
    dup = np.zeros(len(win["x"]), dtype=bool)
    for r in np.unique(win["run"]):
        m = win["run"] == r
        ends = win["end"][m]
        lo = [max(0, e - WM_HORIZON + 1) for e in ends]
        agree = np.array([wm_agreement(d[source][r, l : e + 1, 0][None], d["watermark"][r, l : e + 1][None])[0]
                          for l, e in zip(lo, ends)])
        R[m] = residuals(win["x"][m], params[r], agree)
        ds = duplicate_samples(d[source][r])
        dup[m] = np.array([ds[e - W + 1 : e + 1].any() for e in ends])
    return R, dup
