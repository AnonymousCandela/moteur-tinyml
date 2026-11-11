"""Nœud capteurs : simule le moteur et publie chaque mesure signée (compteur + HMAC) sur MQTT.

  python -m iot.sensor_node --fault 2 --fault-onset 40 [--compromis-attack 2 --attack-onset 60] --speed 10
--compromis-attack simule un nœud compromis : la falsification est appliquée AVANT la signature.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import paho.mqtt.client as mqtt

from sim.attacks import manipulate
from sim.protocol import sign
from sim.scenario import run_scenario

TOPIC = "moteur/capteurs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fault", type=int, default=0); ap.add_argument("--fault-onset", type=float, default=40)
    ap.add_argument("--compromis-attack", type=int, default=0); ap.add_argument("--attack-onset", type=float, default=60)
    ap.add_argument("--speed", type=float, default=1.0, help="accélération (10 = 100 messages/s)")
    a = ap.parse_args()

    s = run_scenario(seed=a.seed, fault=a.fault, fault_onset_s=a.fault_onset)
    X = s["true"]
    if a.compromis_attack:
        onset = int(a.attack_onset * 10)
        X, _ = manipulate(X, a.compromis_attack, onset, np.random.default_rng(a.seed + 2),
                          replay_src=s["commissioning"])
    key = s["key"]
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="noeud-capteurs")
    c.connect(a.host, a.port); c.loop_start()
    period = 0.1 / a.speed
    t_next = time.time()
    for k in range(len(X)):
        body, tag = sign(key, k, X[k])
        c.publish(TOPIC, body + tag, qos=1)
        t_next += period
        time.sleep(max(0.0, t_next - time.time()))
    time.sleep(0.5)
    c.publish(TOPIC, b"FIN", qos=1)
    time.sleep(0.5)
    c.loop_stop(); c.disconnect()
    print(f"{len(X)} messages publiés sur {TOPIC}")


if __name__ == "__main__":
    main()
