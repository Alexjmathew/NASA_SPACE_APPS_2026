<div align="center">

# 🛰️ SAR-Based AI-Powered Multi-Sensor Maritime Awareness System

**See ships through clouds, darkness and silence.**
Detect vessels in SAR satellite imagery, read their heading from wakes, cross-check against AIS, and flag the ones that don't add up.

`NASA Space Apps Challenge 2026` · `SAR` · `AIS` · `YOLO` · `Kalman tracking` · `Python`

[▶ Demo video](#-demo-video) · [🚀 Quick start](#-quick-start) · [🧠 How it works](#-how-it-works) · [📊 Results](#-results) · [🗺️ Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 🎬 Demo video

[![Watch the demo](https://drive.google.com/thumbnail?id=1KX_T6FP4bg6sBJnxjx5EIBA0qSkS4C8i&sz=w1000)](https://drive.google.com/file/d/1KX_T6FP4bg6sBJnxjx5EIBA0qSkS4C8i/view)

**▶ [Watch the 3-minute demo on Google Drive](https://drive.google.com/file/d/1KX_T6FP4bg6sBJnxjx5EIBA0qSkS4C8i/view)**

> Direct download: `https://drive.google.com/uc?export=download&id=1KX_T6FP4bg6sBJnxjx5EIBA0qSkS4C8i`

---

## 🌊 The problem

Illegal fishing, smuggling and "dark" vessels that switch off their AIS transponders are hard to catch. Optical satellites go blind at night and under cloud. Radar does not.

## 💡 Our solution

A pipeline that fuses **SAR imagery** with **AIS broadcasts** to answer one question: *is every ship I can see also a ship that is telling me where it is?*

1. **Detect** ships in SAR scenes (classic CFAR or a trained YOLO model).
2. **Estimate heading** from the ship's wake.
3. **Match** detections to AIS messages, time-aware (the satellite pass and AIS timestamps rarely align).
4. **Track** with a Kalman filter to project AIS positions to the image time.
5. **Score anomalies**: unmatched ships, heading mismatches, impossible speeds.

```
SAR image ─► Georeference ─► Ship detection ─► Wake heading ─┐
                                                              ├─► Time-aware matching ─► Anomaly score ─► Alerts
AIS stream ─► Kalman tracking / interpolation ───────────────┘
```

## 🧩 Repository contents

| Path | What it is |
| --- | --- |
| `mission_console/index.html` | Web mission console. Open in a browser, no install. Tactical fusion map, SAR candidate detector, pipeline, roadmap. Demo data is simulated. |
| `code/maritime_awareness.py` | Full SAR + AIS pipeline: georeferencing, time-aware matching, Kalman tracking, wake direction, anomaly scoring. Runs offline in `simulate` mode. |
| `code/sar_detector_ui.py` | Tkinter desktop app, SAR only. Classic CFAR or a trained YOLO model. |
| `code/train_sar_ship.py` | Prepare data (SSDD, SL-SSDD, Kaggle SARscope), train, evaluate and run YOLO. |
| `notebooks/` | Colab and Kaggle training notebooks, and the pipeline notebook. |
| `docs/` | Documentation, figures and video thumbnail. |

## 🚀 Quick start

```bash
git clone https://github.com/alexjmt1/NASA_SPACE_APPS_2026.git
cd NASA_SPACE_APPS_2026
pip install -r requirements.txt

python code/maritime_awareness.py     # simulated scene, no data needed
python code/sar_detector_ui.py        # desktop app
```

Then open `mission_console/index.html` in a browser for the console.

**Requirements:** Python 3.9+, plus the packages in `requirements.txt`. Tkinter ships with most Python installs.

## 🎓 Training your own model

1. Upload `notebooks/train_sarscope_kaggle.ipynb` to Kaggle (or use the Colab notebooks).
2. Attach the dataset (SSDD, SL-SSDD or Kaggle SARscope).
3. Run all cells.
4. Load the resulting `sar_ship_best.pt` in the desktop app.

## 🧠 How it works

| Stage | Method | Why |
| --- | --- | --- |
| Detection | CFAR baseline, YOLO for learned detection | CFAR needs no training data; YOLO handles clutter and small ships |
| Heading | Wake direction analysis | SAR gives position, not heading |
| Matching | Time-aware association | Satellite and AIS timestamps differ by minutes |
| Tracking | Kalman filter | Projects AIS reports to the moment of the SAR acquisition |
| Anomalies | Rule + score based | Flags dark ships, heading mismatch, position mismatch |

## 📊 Results

> ⚠️ **Fill this in before submission. Judges reward measured evidence.**

| Dataset | Model | Precision | Recall | mAP@0.5 |
| --- | --- | --- | --- | --- |
| SSDD | YOLO | _TBD_ | _TBD_ | _TBD_ |
| Real Sentinel-1 scene | CFAR | _TBD_ | _TBD_ | _TBD_ |
| Real Sentinel-1 + real AIS | Full pipeline | _TBD_ | _TBD_ | _TBD_ |

Add 2 to 3 screenshots to `docs/` (detections overlaid on a real scene, a flagged dark vessel) and reference them here:

```md
![Detections](docs/detections.png)
```

## 🛰️ Data sources

- **SAR:** Copernicus Sentinel-1 (open access)
- **Training sets:** SSDD, SL-SSDD, Kaggle SARscope
- **AIS:** _add your source (e.g. a public AIS archive)_
- **NASA resources:** _add any NASA Earthdata / SAR products you use_

## 🌍 Impact

- Supports fisheries enforcement and marine protected areas
- Helps search and rescue find vessels that are not broadcasting
- Works day, night and through cloud
- Built entirely on open data and open-source tools

## ✅ Status and limitations

- Tested on **simulated scenes**. Accuracy on real Sentinel-1 data is being measured (see [Results](#-results)).
- Wake-based heading depends on sea state and image resolution.
- AIS coverage varies by region.

## 🗺️ Roadmap

- [ ] Validate on real Sentinel-1 scenes with real AIS
- [ ] Publish precision / recall numbers
- [ ] Live data ingestion in the mission console
- [ ] Alert export (GeoJSON / email)

## 👥 Team

| Name | Role |
| --- | --- |
| _Your name_ | _Role_ |

## 🙏 Acknowledgements

NASA Space Apps Challenge 2026, ESA Copernicus Sentinel-1, and the authors of the SSDD, SL-SSDD and SARscope datasets.

## 📄 License

_Add a license (MIT is a safe default) in a `LICENSE` file._
