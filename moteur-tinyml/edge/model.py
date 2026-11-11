"""Configuration A (seuils) et B (MLP TinyML) : entraînement, quantification INT8, inférence, export C."""
from __future__ import annotations

import hashlib
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

N_CLASSES = 5


# ----------------------------------------------------------------------------- configuration A
class Thresholds:
    """Détection classique : un seuil par capteur sur la moyenne de la fenêtre, appris sur l'entraînement."""
    RULES = [(1, 3, ">"), (2, 1, ">"), (3, 2, "<"), (4, 4, ">")]    # (classe, canal, sens)

    def fit(self, x, y):
        m = x.mean(1)
        self.t = {}
        for c, ch, op in self.RULES:
            cand = np.quantile(m[:, ch], np.linspace(0.01, 0.99, 197))
            best, bt = -1, None
            for t in cand:
                pred = m[:, ch] > t if op == ">" else m[:, ch] < t
                tp = np.sum(pred & (y == c)); fp = np.sum(pred & (y != c)); fn = np.sum(~pred & (y == c))
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best:
                    best, bt = f1, t
            self.t[c] = float(bt)
        return self

    def predict(self, x):
        m = x.mean(1)
        out = np.zeros(len(x), dtype=int)
        for c, ch, op in reversed(self.RULES):        # priorité : P1 > P2 > P3 > P4
            hit = m[:, ch] > self.t[c] if op == ">" else m[:, ch] < self.t[c]
            out[hit] = c
        return out


# ----------------------------------------------------------------------------- configuration B
def build_mlp(n_in=100, h1=32, h2=16):
    inp = tf.keras.Input(shape=(n_in,), name="fenetre")
    h = tf.keras.layers.Dense(h1, activation="relu")(inp)
    h = tf.keras.layers.Dense(h2, activation="relu")(h)
    # sorties brutes (logits) : la décision est l'argmax, un softmax est inutile en embarqué et son
    # approximation INT8 dans TFLM briserait la parité exacte avec Python
    out = tf.keras.layers.Dense(N_CLASSES, activation=None, name="logits")(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return m


class Normalizer:
    def fit(self, x):
        self.mu = x.reshape(-1, x.shape[-1]).mean(0)
        self.sd = x.reshape(-1, x.shape[-1]).std(0) + 1e-6
        return self

    def __call__(self, x):
        return ((x - self.mu) / self.sd).reshape(len(x), -1).astype(np.float32)


def train_mlp(xn_tr, y_tr, xn_va, y_va, seed=0, epochs=60, balance=False):
    """balance=False retenu : la pondération des classes faisait passer les faux positifs de 2 % à 12 %."""
    tf.keras.utils.set_random_seed(seed)
    m = build_mlp(xn_tr.shape[1])
    counts = np.bincount(y_tr, minlength=N_CLASSES)
    cw = {c: len(y_tr) / (N_CLASSES * max(counts[c], 1)) for c in range(N_CLASSES)} if balance else None
    m.fit(xn_tr, y_tr, validation_data=(xn_va, y_va), epochs=epochs, batch_size=256, verbose=0, class_weight=cw,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return m


def to_int8(model, rep):
    def gen():
        for k in np.random.default_rng(0).choice(len(rep), size=min(2000, len(rep)), replace=False):
            yield [rep[k : k + 1]]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def quantize_input(xn, scale, zp):
    """Arrondi demi loin de zéro, identique à lroundf() côté C."""
    v = xn / scale
    return np.clip(np.sign(v) * np.floor(np.abs(v) + 0.5) + zp, -128, 127).astype(np.int8)


class Int8Model:
    def __init__(self, blob):
        self.blob = blob
        # noyaux de référence : mêmes arrondis que TFLite Micro (XNNPACK arrondit différemment)
        self.it = tf.lite.Interpreter(model_content=blob,
                                      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        self.in_s, self.in_zp = self.i["quantization"]
        self.out_s, self.out_zp = self.o["quantization"]

    def raw(self, xn):
        q = quantize_input(xn, self.in_s, self.in_zp)
        out = np.empty((len(q), N_CLASSES), dtype=np.int8)
        for k in range(len(q)):
            self.it.set_tensor(self.i["index"], q[k : k + 1]); self.it.invoke()
            out[k] = self.it.get_tensor(self.o["index"])[0]
        return out

    def predict(self, xn):
        return self.raw(xn).argmax(1)


def macs(h=(100, 32, 16, 5)):
    return sum(a * b for a, b in zip(h[:-1], h[1:]))


# ----------------------------------------------------------------------------- export C
def export_c(fw_dir, im: Int8Model, norm: Normalizer, golden_raw: np.ndarray):
    """model_data.cc + model_params.h pour le micrologiciel Edge (TFLM)."""
    os.makedirs(fw_dir, exist_ok=True)
    blob = im.blob
    sha = hashlib.sha256(blob).digest()
    hexs = ",\n  ".join(", ".join(f"0x{b:02x}" for b in blob[i : i + 12]) for i in range(0, len(blob), 12))
    with open(os.path.join(fw_dir, "model_data.cc"), "w") as f:
        f.write('// Généré par edge/model.py — ne pas modifier.\n#include "model_data.h"\n\n'
                f"alignas(16) const unsigned char g_model_data[] = {{\n  {hexs}\n}};\n"
                f"const unsigned int g_model_data_len = {len(blob)};\n")
    gx = norm(golden_raw)
    gout = im.raw(gx)
    fl = lambda a: ", ".join(f"{v:.6f}f" for v in np.ravel(a))  # noqa: E731
    rows = ",\n".join("    { " + fl(g) + " }" for g in golden_raw.reshape(len(golden_raw), -1))
    outs = ",\n".join("    { " + ", ".join(str(int(v)) for v in o) + " }" for o in gout)
    with open(os.path.join(fw_dir, "model_params.h"), "w") as f:
        f.write(f"""// Généré par edge/model.py — ne pas modifier.
#pragma once
#include <stdint.h>

#define EDGE_W 20
#define EDGE_CH 5
#define EDGE_N_IN (EDGE_W * EDGE_CH)
#define EDGE_N_CLASSES {N_CLASSES}

// normalisation par capteur (tension, courant, vitesse, température, vibrations)
static const float EDGE_MEAN[EDGE_CH] = {{ {fl(norm.mu)} }};
static const float EDGE_STD[EDGE_CH]  = {{ {fl(norm.sd)} }};

#define EDGE_IN_SCALE  {im.in_s:.10f}f
#define EDGE_IN_ZP     {im.in_zp}
#define EDGE_OUT_SCALE {im.out_s:.10f}f
#define EDGE_OUT_ZP    {im.out_zp}

static const uint8_t EDGE_MODEL_SHA256[32] = {{ {", ".join(f"0x{b:02x}" for b in sha)} }};

// fenêtres de référence (brutes, W x CH) et sorties INT8 attendues (calculées en Python)
#define EDGE_N_GOLDEN {len(golden_raw)}
static const float EDGE_GOLDEN_RAW[EDGE_N_GOLDEN][EDGE_N_IN] = {{
{rows}
}};
static const int8_t EDGE_GOLDEN_OUT[EDGE_N_GOLDEN][EDGE_N_CLASSES] = {{
{outs}
}};
""")
    return sha.hex()
