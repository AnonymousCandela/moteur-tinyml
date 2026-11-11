# TinyML résilient aux attaques sur les données de capteurs — moteur IoT simulé

Projet CYB6053 (projet 4 – TinyML). Prototype **entièrement logiciel** :

- Un moteur à courant continu simulé par ses équations physiques envoie des mesures signées (HMAC + compteur) par MQTT.
- Un attaquant peut injecter, modifier, rejouer ou bruiter ces mesures.
- Un nœud Edge TinyML les diagnostique selon trois configurations :
  - **A** : seuils classiques ;
  - **B** : réseau de neurones INT8 ;
  - **C** : B protégé par une couche d'intégrité (HMAC et compteur anti-rejeu) et une couche de cohérence physique.

Le même modèle INT8 tourne dans un micrologiciel ESP32-S3 (TFLite Micro) sous l'émulateur QEMU, avec une parité vérifiée par rapport à Python.

```
sim/          motor.py (physique), attacks.py, protocol.py (HMAC), generate.py (jeu de données), scenario.py
edge/         windows.py, model.py (A, B, INT8, export C), protection.py (couche 2), engine.py (flux), qemu_bridge.py
experiments/  run_abc.py (une graine), run_all.sh (5 graines), aggregate.py, paderborn.py (données réelles)
iot/          sensor_node.py, attacker.py, edge_node.py (chaîne MQTT réelle)
dashboard/    app.py (Streamlit)
firmware/     micrologiciel Edge ESP-IDF (main.cc, model_data.cc et model_params.h régénérés par l'expérience)
results/      s0/ (modèle et seuils utilisés par le tableau de bord), summary.json (5 graines)
```

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10 à 3.12
pip install -r requirements.txt
```

## Reproduire toutes les expériences (environ 2 min)
```bash
bash experiments/run_all.sh          # 5 graines : génération, A/B/C, agrégation -> results/summary.json
```

## Tableau de bord
```bash
streamlit run dashboard/app.py
```
Dans la barre latérale, choisir une panne réelle et une attaque (et sa position : nœud compromis ou interception), puis observer les décisions de A, B et C. Exemple parlant : **P1 Surchauffe + C3 Température figée**. A et B répondent « Normal », C répond « Données suspectes ».

## Chaîne MQTT (trois terminaux, plus Mosquitto)
```bash
mosquitto -p 1883
python -m iot.edge_node --seed 7
python -m iot.attacker --attack 2 --onset 60                       # interception (0 = relais transparent)
python -m iot.sensor_node --seed 7 --fault 2 --fault-onset 40 --speed 10
# nœud compromis : --attack 0 côté attaquant, et --compromis-attack 2 --attack-onset 60 côté capteurs
```

## Nœud Edge embarqué (ESP32-S3 sous QEMU)
```bash
. ~/esp/esp-idf/export.sh                      # ESP-IDF ≥ 5.1 (testé avec 5.4.2)
python $IDF_PATH/tools/idf_tools.py install qemu-xtensa && . ~/esp/esp-idf/export.sh
cd firmware
idf.py -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu" set-target esp32s3 build
cd build && python -m esptool --chip esp32s3 merge_bin --fill-flash-size 8MB -o qemu_flash.bin @flash_args && cd ../..
python -m edge.qemu_bridge --image firmware/build/qemu_flash.bin --data data/dataset_s0.npz \
       --model results/s0/model_int8.tflite --state results/s0/edge_state.npz --n 1000
```
Au démarrage, le micrologiciel vérifie le SHA-256 du modèle, charge TFLM et rejoue 5 fenêtres de référence. Ensuite, il classe les fenêtres reçues sur l'UART. `sdkconfig.qemu` remplace les noyaux SIMD par les noyaux C génériques et désactive l'accélérateur SHA, que QEMU n'émule pas.

## Validation sur données réelles (Paderborn)
Télécharger `measures_v2.csv` depuis Kaggle (jeu « Electric Motor Temperature », wkirgsn), puis :
```bash
python -m experiments.paderborn --csv measures_v2.csv --out results/paderborn.json
```

## Résultats vérifiés (11 nov. 2026, 5 graines, moyenne ± écart type)

| Mesure | A (seuils) | B (TinyML INT8) | C (B protégé) |
|---|---|---|---|
| F1 macro, données propres | 0,753 ± 0,026 | 0,877 ± 0,034 | 0,876 ± 0,033 |
| Faux positifs (fenêtres normales) | 9,0 ± 5,0 % | 6,9 ± 3,4 % | 7,0 ± 3,4 % |
| Décisions trompées sous attaque | 45,8 ± 1,7 % | 49,9 ± 1,5 % | — |
| Détection des attaques | — | — | 95,6 ± 0,4 % |
| Pannes réelles prises pour des attaques | — | — | 0,1 ± 0,1 % |
| Masquage (panne + attaque) : rappel de la panne | — | 7,5 ± 0,9 % (68,2 % sans attaque) | 99,9 % de décisions sûres |
| F1 macro, moteurs hors plage | 0,565 ± 0,038 | 0,794 ± 0,025 | fausse suspicion 1,2 % |

**Volet embarqué**

- **Modèle :** 3 845 paramètres, 7 520 octets en INT8, 3 792 multiplications-accumulations par inférence.
- **Mémoire :** arène TFLM de 1 308 octets, mesurée sous QEMU.
- **Quantification :** l'écart de F1 entre le modèle float et INT8 est de 0,0004.
- **Parité C/Python :** 999 décisions identiques sur 1 000 fenêtres de test, avec un écart maximal d'un pas INT8.
- **Latence :** estimée entre 16 et 63 µs à 240 MHz, en supposant 1 à 4 cycles par multiplication-accumulation. C'est une estimation, pas une mesure.

**Par attaque (détection par C)**

- **Interception :** toutes les attaques sont détectées à 100 % par la couche 1.
- **Nœud compromis :**
  - température figée : 100 % ;
  - courant modifié : 99,7 % ;
  - rejeu : 98,8 % ;
  - bruit : 79 % ;
  - injection : 66 %.

## Limites (à discuter dans l'article)
- **Hausse de vibrations injectée par un nœud compromis :** elle est indiscernable d'une vraie panne P4. Il n'y a qu'un accéléromètre, et aucune autre grandeur ne contredit la mesure. C'est la principale cause des 66 % de détection sur C1. Il faudrait un capteur redondant.
- **Attaquant connaissant le modèle physique :** il peut falsifier toutes les mesures de façon cohérente, comme l'ont montré Liu et al. (2009). Ce cas n'est pas testé.
- **Données simulées :** les résultats viennent d'un simulateur. La généralisation est mesurée sur des moteurs hors de la plage d'entraînement et doit être confirmée sur les données de Paderborn. Le script est prêt, mais il n'a été testé que sur un fichier factice au même format.
- **Pas de matériel :** la latence et l'énergie sont estimées. La mémoire, elle, est mesurée exactement.
- **Tatouage actif :** il suppose que l'Edge connaît la petite perturbation pseudo-aléatoire de ±0,25 V appliquée à l'alimentation, à la manière de Mo et Sinopoli (2009).
