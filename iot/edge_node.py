"""Nœud Edge MQTT : vérifie chaque message (HMAC + compteur), alimente le moteur de décision et publie le
diagnostic des configurations A, B et C sur moteur/diagnostic.

  python -m iot.edge_node --results results/s0 --seed 7
La mise en service (30 s saines du même moteur) et la séquence de tatouage sont reconstituées à partir de la
graine du scénario, comme si elles avaient été enregistrées lors de l'installation.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter

import paho.mqtt.client as mqtt

from edge.engine import LABELS, EdgeEngine
from sim.protocol import FMT, TAG_LEN, Verifier, decode
from sim.scenario import run_scenario

IN, OUT = "moteur/edge", "moteur/diagnostic"
SIZE = struct.calcsize(FMT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--results", default="results/s0")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    ref = run_scenario(seed=a.seed)                     # mise en service + clé + tatouage du moteur installé
    eng = EdgeEngine(a.results)
    eng.commission(ref["commissioning"])
    ver = Verifier(ref["key"])
    wm = ref["watermark"]
    k = [0]
    stats = {cfg: Counter() for cfg in ("A", "B", "C")}

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="noeud-edge")

    def on_message(cl, ud, msg):
        if msg.payload == b"FIN":
            print("\nRésumé des décisions (nombre de fenêtres) :")
            for cfg, cnt in stats.items():
                print(f"  {cfg} : " + ", ".join(f"{LABELS[l]}={n}" for l, n in sorted(cnt.items())))
            cl.disconnect(); return
        body, tag = msg.payload[:SIZE], msg.payload[SIZE:SIZE + TAG_LEN]
        ok = ver.check(body, tag)
        _, vals = decode(body)
        i = k[0]; k[0] += 1
        r = eng.push(vals, ok, wm[i] if i < len(wm) else 0.0)
        if r:
            for cfg in ("A", "B", "C"):
                stats[cfg][r[cfg]] += 1
            out = dict(t=r["t"], A=LABELS[r["A"]], B=LABELS[r["B"]], C=LABELS[r["C"]], couche1_ok=r["couche1_ok"],
                       controles=[n for n, v in r["controles"].items() if v])
            cl.publish(OUT, json.dumps(out, ensure_ascii=False), qos=0)
            if r["C"] == 5 and int(r["t"] * 2) % 10 == 0:
                print(f"[t={r['t']:6.1f} s] ALERTE C : données suspectes {out['controles'] or '(couche 1)'}")

    c.on_message = on_message
    c.connect(a.host, a.port)
    c.subscribe(IN, qos=1)
    print(f"Edge à l'écoute de {IN}")
    c.loop_forever()


if __name__ == "__main__":
    main()
