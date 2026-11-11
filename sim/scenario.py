"""Un scénario unique (moteur + panne + attaque), pour le tableau de bord et la chaîne MQTT."""
from __future__ import annotations

import numpy as np

from .attacks import manipulate, transmit
from .motor import FS, RunSpec, sample_params, sense, simulate


def run_scenario(seed=1, fault=0, severity=None, fault_onset_s=30.0, attack=0, variant="compromis",
                 attack_onset_s=60.0, noisy_replay=False, duration_s=120.0, load_base=0.18, t_amb=25.0):
    rng = np.random.default_rng(seed)
    default_sev = {0: 0.0, 1: 2.0, 2: 0.8, 3: 0.3, 4: 0.5}
    spec = RunSpec(seed=seed, duration_s=duration_s, params=sample_params(rng), fault=fault,
                   fault_onset_s=0.0 if fault == 1 else fault_onset_s,
                   severity=default_sev[fault] if severity is None else severity, load_base=load_base, t_amb=t_amb)
    real = simulate([spec], with_fault=True)
    twin = simulate([spec], with_fault=False)
    X = sense(real["true"], [seed], salt=0)[0].astype(np.float32)
    Xt = sense(twin["true"], [seed], salt=1)[0].astype(np.float32)
    n = len(X)
    onset = int(attack_onset_s * FS) if attack else n
    r = np.random.default_rng(seed + 1)
    key = r.bytes(32)
    Y, info = manipulate(X, attack, onset, r, replay_src=Xt, noisy_replay=noisy_replay)
    off = None
    if attack == 4:
        L = int(info["longueur_s"] * FS)
        off = np.zeros(n, dtype=int)
        off[onset:] = np.arange(n - onset) % L
    ok = transmit(X, Y, attack, onset, variant if attack else "aucune", key, off)
    return dict(received=Y, true=X, accepted=ok, watermark=real["watermark"][0], commissioning=Xt[: 30 * FS],
                y_phys=real["labels"][0], attack_onset_s=onset / FS if attack else None, info=info, key=key)
