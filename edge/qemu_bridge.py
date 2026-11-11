"""Logiciel dans la boucle : envoie des fenêtres au micrologiciel Edge exécuté sous QEMU et compare ses
décisions à celles du modèle INT8 exécuté en Python.

Usage :
  python -m edge.qemu_bridge --image firmware/build/qemu_flash.bin --data data/dataset_s0.npz \
         --model results/s0/model_int8.tflite --state results/s0/edge_state.npz --n 300
Prérequis : QEMU d'Espressif (python $IDF_PATH/tools/idf_tools.py install qemu-xtensa) et l'image fusionnée
(voir README : esptool merge_bin).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import socket
import subprocess
import time

import numpy as np

from edge.model import Int8Model
from edge.windows import load, make_windows, runs_of


def find_qemu() -> str:
    q = shutil.which("qemu-system-xtensa")
    if q:
        return q
    c = glob.glob(os.path.expanduser("~/.espressif/tools/qemu-xtensa/*/qemu/bin/qemu-system-xtensa"))
    if not c:
        raise SystemExit("QEMU introuvable : python $IDF_PATH/tools/idf_tools.py install qemu-xtensa")
    return c[0]


class EdgeQEMU:
    def __init__(self, image: str, port: int = 5555):
        self.p = subprocess.Popen([find_qemu(), "-nographic", "-machine", "esp32s3",
                                   "-drive", f"file={image},if=mtd,format=raw",
                                   "-serial", f"tcp:127.0.0.1:{port},server=on,wait=on", "-monitor", "none"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
                break
            except OSError:
                time.sleep(0.1)
        self.f = self.s.makefile("rwb", buffering=0)
        self.boot_log = []
        while True:
            line = self.f.readline().decode(errors="ignore").strip()
            self.boot_log.append(line)
            if line == "PRET" or line.startswith("TAMPER"):
                break

    def classify(self, raw_window: np.ndarray):
        msg = "W," + ",".join(f"{v:.4f}" for v in raw_window.ravel()) + "\n"
        self.f.write(msg.encode())
        while True:
            line = self.f.readline().decode(errors="ignore").strip()
            if line.startswith("R,"):
                p = line.split(",")
                return int(p[1]), np.array([int(v) for v in p[2:7]], dtype=np.int8)
            if line.startswith("E,"):
                raise RuntimeError(line)

    def close(self):
        self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--state", required=True, help="edge_state.npz (normalisation)")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    d = load(a.data)
    st = np.load(a.state)
    w = make_windows(d, runs_of(d, "test"))
    idx = np.random.default_rng(0).choice(len(w["x"]), size=min(a.n, len(w["x"])), replace=False)
    x = w["x"][idx]
    # les fenêtres envoyées sont arrondies à 4 décimales : Python utilise exactement les mêmes valeurs
    x = np.round(x.astype(np.float64), 4).astype(np.float32)
    xn = ((x - st["mu"]) / st["sd"]).reshape(len(x), -1).astype(np.float32)
    im = Int8Model(open(a.model, "rb").read())
    py_raw = im.raw(xn)

    edge = EdgeQEMU(a.image)
    print("\n".join(l for l in edge.boot_log if "edge:" in l or l.startswith("TAMPER")))
    t0 = time.time()
    same_cls, maxdiff = 0, 0
    for k in range(len(x)):
        cls, raw = edge.classify(x[k])
        same_cls += int(cls == int(py_raw[k].argmax()))
        maxdiff = max(maxdiff, int(np.max(np.abs(raw.astype(int) - py_raw[k].astype(int)))))
    edge.close()
    print(f"{len(x)} fenêtres : décisions identiques C/Python = {same_cls}/{len(x)} | "
          f"écart max des sorties INT8 = {maxdiff} pas | {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
