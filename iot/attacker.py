"""Attaquant intercalé sur le canal MQTT (sans la clé) : relaie moteur/capteurs vers moteur/edge,
et falsifie ou rejoue les messages à partir de --onset.

  python -m iot.attacker --attack 0            # relais transparent (pas d'attaque)
  python -m iot.attacker --attack 3 --onset 60 # température figée par interception
  python -m iot.attacker --attack 4 --onset 60 # rejeu des messages enregistrés au début

Comme l'attaquant n'a pas la clé, un message modifié garde son ancienne étiquette HMAC, et un message rejoué
garde son ancien compteur : la couche 1 de l'Edge doit les rejeter.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np
import paho.mqtt.client as mqtt

from sim.protocol import FMT, TAG_LEN

IN, OUT = "moteur/capteurs", "moteur/edge"
SIZE = struct.calcsize(FMT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--attack", type=int, default=0, help="0 aucune, 1 injection, 2 courant ×0,7, 3 T figée, 4 rejeu, 5 bruit")
    ap.add_argument("--onset", type=float, default=60.0)
    a = ap.parse_args()
    onset = int(a.onset * 10)
    rng = np.random.default_rng(0)
    recorded, k = [], [0]

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attaquant")

    def on_message(cl, ud, msg):
        if msg.payload == b"FIN":
            cl.publish(OUT, b"FIN", qos=1); return
        body, tag = msg.payload[:SIZE], msg.payload[SIZE:SIZE + TAG_LEN]
        i = k[0]; k[0] += 1
        if i < 150:
            recorded.append(msg.payload)                 # enregistre 15 s de messages pour un futur rejeu
        if a.attack == 0 or i < onset:
            cl.publish(OUT, msg.payload, qos=1); return
        if a.attack == 4:
            cl.publish(OUT, recorded[(i - onset) % len(recorded)], qos=1); return
        cnt, *v = struct.unpack(FMT, body)
        if a.attack == 1:
            v[3] -= 30.0
        elif a.attack == 2:
            v[1] *= 0.7
        elif a.attack == 3:
            v[3] = 50.0
        elif a.attack == 5:
            v = [x + rng.normal(0, 1) * s for x, s in zip(v, (0.3, 0.2, 50, 1.0, 0.03))]
        cl.publish(OUT, struct.pack(FMT, cnt, *v) + tag, qos=1)   # étiquette d'origine : il ne peut pas re-signer

    c.on_message = on_message
    c.connect(a.host, a.port)
    c.subscribe(IN, qos=1)
    print(f"Relais {IN} → {OUT}, attaque {a.attack} à partir de {a.onset} s")
    c.loop_forever()


if __name__ == "__main__":
    main()
