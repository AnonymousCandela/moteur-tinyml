"""Simulateur physique d'un moteur à courant continu (électrique, mécanique, thermique, vibrations).

Équations (intégrées au pas DT = 1 ms, échantillonnées à FS = 10 Hz) :
  électrique (quasi statique, L/R ≈ 2 ms ≪ 100 ms) : i = (V − Ke·ω) / R(T)
  résistance du cuivre                              : R(T) = R0 · [1 + α (T − 20)]
  mécanique                                         : J dω/dt = Kt·i − B·ω − τ_charge
  thermique                                         : Cth dT/dt = R(T)·i² − (T − T_amb) / Rth
  vibrations (RMS, g)                               : a = vib0 + vib1 (ω/ω_nom)² (+ défaut)

Pannes physiques (codes de classe) :
  0 normal · 1 P1 surchauffe (Rth × k) · 2 P2 surcharge (τ × k) · 3 P3 sous-tension (V × (1−k)) · 4 P4 vibrations
Toute la simulation est vectorisée sur les essais (une boucle sur le temps seulement).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS = 10                    # Hz, fréquence des capteurs
DT = 1e-3                  # s, pas d'intégration
SUB = int(round(1 / (FS * DT)))
CHANNELS = ["tension_V", "courant_A", "vitesse_rpm", "temperature_C", "vibration_g"]
CLASSES = ["normal", "P1_surchauffe", "P2_surcharge", "P3_sous_vitesse", "P4_vibrations"]
ALPHA_CU = 0.0039          # 1/°C, coefficient de température du cuivre

NOMINAL = dict(R0=1.2, Ke=0.05, J=2e-4, B=1e-4, Rth=1.5, Cth=150.0, V=24.0, vib0=0.05, vib1=0.60)
OMEGA_NOM = 380.0          # rad/s (≈ 3 630 tr/min)

# bruit et résolution des capteurs (écart type, pas de quantification)
SENSOR_NOISE = np.array([0.05, 0.03, 8.0, 0.15, 0.004])
SENSOR_STEP = np.array([0.01, 0.01, 1.0, 0.1, 0.001])

# tatouage actif : petite composante pseudo-aléatoire connue de l'Edge sur la tension d'alimentation
WM_AMPL = 0.25             # V
WM_CHIP = 2                # échantillons par « bit » (0,2 s)


@dataclass
class RunSpec:
    seed: int
    duration_s: float = 120.0
    params: dict = field(default_factory=dict)  # paramètres du moteur (dérivés de NOMINAL)
    fault: int = 0                              # 0..4
    fault_onset_s: float = 0.0
    severity: float = 0.0                       # intensité de la panne (sens selon la panne)
    load_base: float = 0.2                      # N·m
    t_amb: float = 25.0                         # °C


def sample_params(rng: np.random.Generator, spread: float = 0.10, outside: bool = False) -> dict:
    """Paramètres d'un moteur tirés autour du nominal (± spread). outside=True : hors de la plage (10–25 %)."""
    p = {}
    for k, v in NOMINAL.items():
        if k == "V":
            p[k] = v
            continue
        if outside:
            mag = rng.uniform(spread, 2.5 * spread) * rng.choice([-1, 1])
        else:
            mag = rng.uniform(-spread, spread)
        p[k] = v * (1 + mag)
    p["Kt"] = p["Ke"]
    return p


def watermark(seed: int, n: int) -> np.ndarray:
    """Séquence ±WM_AMPL connue de l'Edge (même graine), constante par paquet de WM_CHIP échantillons."""
    r = np.random.default_rng(10_000_019 + seed)
    chips = r.choice([-1.0, 1.0], size=(n + WM_CHIP - 1) // WM_CHIP)
    return WM_AMPL * np.repeat(chips, WM_CHIP)[:n]


def _load_schedule(rng, n, base):
    """Charge variable : paliers aléatoires (10–30 s) × oscillation lente."""
    mult = np.empty(n)
    i = 0
    while i < n:
        L = int(rng.uniform(10, 30) * FS)
        mult[i:i + L] = rng.uniform(0.8, 1.2)
        i += L
    t = np.arange(n) / FS
    osc = 1 + 0.10 * np.sin(2 * np.pi * t / rng.uniform(15, 40) + rng.uniform(0, 2 * np.pi))
    return base * mult * osc


def simulate(specs: list[RunSpec], with_fault: bool = True) -> dict:
    """Simule un lot d'essais. Renvoie les grandeurs VRAIES (sans bruit capteur) à FS Hz.

    with_fault=False donne le « jumeau sain » (même moteur, même charge, même tatouage) qui sert à la mise en
    service (identification des paramètres par l'Edge) et de source pour les attaques par rejeu.
    """
    N = len(specs)
    n = int(specs[0].duration_s * FS)
    P = {k: np.array([s.params[k] for s in specs]) for k in ["R0", "Ke", "Kt", "J", "B", "Rth", "Cth", "V", "vib0", "vib1"]}
    rngs = [np.random.default_rng(s.seed) for s in specs]
    load = np.stack([_load_schedule(r, n, s.load_base) for r, s in zip(rngs, specs)])
    wm = np.stack([watermark(s.seed, n) for s in specs])
    fault = np.array([s.fault if with_fault else 0 for s in specs])
    onset = np.array([int(s.fault_onset_s * FS) for s in specs])
    sev = np.array([s.severity for s in specs])
    t_amb = np.array([s.t_amb for s in specs])
    imb = np.array([r.uniform(-0.1, 0.1) for r in rngs])          # balourd résiduel propre à chaque moteur

    # facteur de refroidissement (P1 : panne persistante depuis le début de l'essai)
    rth_eff = P["Rth"] * np.where(fault == 1, sev, 1.0)

    # conditions initiales : régime établi approximatif, moteur partiellement chauffé
    R_init = P["R0"] * (1 + ALPHA_CU * (60 - 20))
    tau0 = load[:, 0]
    w = (P["Kt"] * P["V"] / R_init - tau0) / (P["Kt"] * P["Ke"] / R_init + P["B"])
    i_ss = (P["V"] - P["Ke"] * w) / R_init
    T = t_amb + rth_eff * R_init * i_ss ** 2 * np.array([r.uniform(0.7, 1.0) for r in rngs])

    out = {c: np.empty((N, n)) for c in ["V", "i", "w", "T", "vib"]}
    for k in range(n):
        active = (k >= onset)
        V = P["V"] * np.where((fault == 3) & active, 1 - sev, 1.0) + wm[:, k]
        tau = load[:, k] * np.where((fault == 2) & active, 1 + sev, 1.0)
        for _ in range(SUB):
            R = P["R0"] * (1 + ALPHA_CU * (T - 20))
            i = (V - P["Ke"] * w) / R
            w = np.maximum(w + DT * (P["Kt"] * i - P["B"] * w - tau) / P["J"], 0.0)
            T = T + DT * (R * i * i - (T - t_amb) / rth_eff) / P["Cth"]
        R = P["R0"] * (1 + ALPHA_CU * (T - 20))
        i = (V - P["Ke"] * w) / R
        vib = P["vib0"] + P["vib1"] * (w / OMEGA_NOM) ** 2 * (1 + imb)
        vib = vib + np.where((fault == 4) & active, sev * (w / OMEGA_NOM) * (1 + 0.3 * np.sin(k / 3.0)), 0.0)
        out["V"][:, k], out["i"][:, k], out["w"][:, k], out["T"][:, k], out["vib"][:, k] = V, i, w, T, vib

    labels = np.zeros((N, n), dtype=np.int8)
    for r in range(N):
        if with_fault and fault[r]:
            labels[r, onset[r]:] = fault[r]
    return dict(true=out, labels=labels, watermark=wm)


def sense(true: dict, seeds: list[int], salt: int = 0) -> np.ndarray:
    """Modèle de capteur : bruit gaussien, petite dérive lente, quantification. Renvoie (N, n, 5)."""
    rpm = true["w"] * 60 / (2 * np.pi)
    X = np.stack([true["V"], true["i"], rpm, true["T"], true["vib"]], axis=-1)
    N, n, _ = X.shape
    out = np.empty_like(X)
    for r in range(N):
        g = np.random.default_rng(seeds[r] * 7 + 13 + salt)
        noise = g.normal(0, 1, size=(n, 5)) * SENSOR_NOISE
        drift = np.cumsum(g.normal(0, 1, size=(n, 5)), axis=0) * SENSOR_NOISE * 0.01
        out[r] = X[r] + noise + drift
    out[..., 2] = np.maximum(out[..., 2], 0)
    out[..., 4] = np.maximum(out[..., 4], 0)
    return np.round(out / SENSOR_STEP) * SENSOR_STEP
