"""Attaques sur les données des capteurs, appliquées au flux de messages entre le nœud capteurs et l'Edge.

Codes : 0 aucune · 1 C1 injection · 2 C2 modification du courant · 3 C3 température figée · 4 C4 rejeu · 5 C5 bruit.
Variantes :
  « interception » — l'attaquant est sur le canal, sans la clé : il ne peut pas re-signer. Les messages modifiés
                    gardent l'ancienne étiquette ; les messages rejoués gardent leur ancien compteur.
  « compromis »    — l'attaquant contrôle le nœud capteurs : chaque message falsifié est correctement signé
                    avec un compteur neuf. Seule la cohérence physique peut alors le trahir.
"""
from __future__ import annotations

import numpy as np

from .motor import SENSOR_NOISE, SENSOR_STEP
from .protocol import Verifier, sign

ATTACKS = ["aucune", "C1_injection", "C2_modification", "C3_temp_figee", "C4_rejeu", "C5_bruit"]
I_V, I_I, I_RPM, I_T, I_VIB = range(5)


def manipulate(X: np.ndarray, attack: int, onset: int, rng: np.random.Generator,
               replay_src: np.ndarray | None = None, noisy_replay: bool = False) -> tuple[np.ndarray, dict]:
    """Renvoie la série falsifiée (n, 5) et les détails de l'attaque. X n'est pas modifié."""
    Y = X.copy()
    info = {}
    s = slice(onset, None)
    if attack == 1:      # injection : biais sur un capteur
        target = rng.choice(["T_bas", "I_haut", "vib_haute"])
        info["cible"] = target
        if target == "T_bas":
            Y[s, I_T] -= rng.uniform(20, 40)
        elif target == "I_haut":
            Y[s, I_I] += rng.uniform(2, 4)
        else:
            Y[s, I_VIB] *= rng.uniform(2.5, 4)
    elif attack == 2:    # modification : courant sous-estimé (masque une surcharge)
        k = rng.uniform(0.6, 0.8)
        info["facteur"] = k
        Y[s, I_I] *= k
    elif attack == 3:    # température figée à une valeur normale
        v = rng.uniform(45, 60)
        info["valeur"] = v
        Y[s, I_T] = v
    elif attack == 4:    # rejeu d'une séquence normale enregistrée plus tôt
        L = int(rng.uniform(10, 20) * 10)
        seg = replay_src[:L]
        reps = int(np.ceil((len(Y) - onset) / L))
        rep = np.tile(seg, (reps, 1))[: len(Y) - onset].copy()
        if noisy_replay:
            rep = rep + rng.normal(0, 1, rep.shape) * SENSOR_NOISE
        Y[s] = rep
        info.update(longueur_s=L / 10, bruite=bool(noisy_replay))
    elif attack == 5:    # perturbation : bruit fort sur 1 à 3 capteurs
        chans = rng.choice(5, size=rng.integers(1, 4), replace=False)
        k = rng.uniform(3, 10)
        info.update(capteurs=[int(c) for c in chans], facteur=k)
        Y[s, chans] += rng.normal(0, 1, (len(Y) - onset, len(chans))) * SENSOR_NOISE[chans] * k
    Y[..., I_VIB] = np.maximum(Y[..., I_VIB], 0)
    Y = np.round(Y / SENSOR_STEP) * SENSOR_STEP
    return Y.astype(np.float32), info


def transmit(X_clean: np.ndarray, X_att: np.ndarray, attack: int, onset: int, variant: str, key: bytes,
             replay_offset: np.ndarray | None = None) -> np.ndarray:
    """Simule le canal et la couche 1 de l'Edge. Renvoie un booléen par échantillon : message accepté ?

    replay_offset : pour un rejeu par interception, indice de l'échantillon d'origine (son ancien compteur).
    """
    ver = Verifier(key)
    n = len(X_clean)
    ok = np.zeros(n, dtype=bool)
    for k in range(n):
        if attack == 0 or k < onset or variant == "compromis":
            body, tag = sign(key, k, X_att[k])                      # message légitime ou nœud compromis
        elif attack == 4:                                           # rejeu par interception : ancien message
            old = int(replay_offset[k])
            body, tag = sign(key, old, X_att[k])
        else:                                                       # modifié sur le canal : étiquette d'origine
            _, tag = sign(key, k, X_clean[k])
            body, _ = sign(key, k, X_att[k])
        ok[k] = ver.check(body, tag)
    return ok
