# SAR-Based AI-Powered Multi-Sensor Maritime Awareness System
NASA Space Apps Challenge 2026

Detect ships in SAR (radar) satellite images, estimate their direction from wakes, cross-check with AIS broadcasts and flag suspicious vessels.

## Contents
| Path | What it is |
|---|---|
| `mission_console/index.html` | Web mission console (open in a browser, no install). Tactical fusion map, SAR candidate detector, pipeline, roadmap. Demo data is simulated. |
| `code/maritime_awareness.py` | Full SAR + AIS pipeline: georeferencing, time-aware matching, Kalman tracking, wake direction, anomaly scoring. Runs offline in `simulate` mode. |
| `code/sar_detector_ui.py` | Tkinter desktop app, SAR only. Classic CFAR or a trained YOLO model. |
| `code/train_sar_ship.py` | Prepare data (SSDD, SL-SSDD, Kaggle SARscope), train, evaluate and run YOLO. |
| `notebooks/` | Colab and Kaggle training notebooks, and the pipeline notebook. |

## Quick start
```
pip install -r requirements.txt
python code/maritime_awareness.py     # simulated scene, no data needed
python code/sar_detector_ui.py        # desktop app
```
Open `mission_console/index.html` in a browser for the console.

## Training
Upload `notebooks/train_sarscope_kaggle.ipynb` to Kaggle (or the Colab notebooks), attach the dataset, run all cells, then load the resulting `sar_ship_best.pt` in the desktop app.

## Status
Tested on simulated scenes only. Accuracy on real Sentinel-1 data is still to be measured.
