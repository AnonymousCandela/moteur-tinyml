#!/usr/bin/env bash
# Reproduit toutes les expériences : 5 graines, puis agrégation (moyenne ± écart type).
# Usage : bash experiments/run_all.sh [graines...]   (défaut : 0 1 2 3 4)
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${*:-"0 1 2 3 4"}
for s in $SEEDS; do
  python -m sim.generate --out data/dataset_s$s.npz --seed $s
  EXPORT=""; [ "$s" = "0" ] && EXPORT="--export firmware/main"
  python -m experiments.run_abc --data data/dataset_s$s.npz --seed $s --out results/s$s $EXPORT > results_s$s.log 2>&1
  echo "graine $s terminée"
done
python -m experiments.aggregate results/s* --out results/summary.json
