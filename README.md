# MLBB Auto — MLBB Vision Pipeline

Deteksi hero otomatis di minimap Mobile Legends menggunakan **YOLOv11n**.

## Pipeline

```
Video → Crop Minimap → YOLOv11n → ByteTrack → Game State
                       2 class:
                       - blue_hero
                       - red_hero
```

## Setup

```bash
# Clone & install
git clone <repo>
cd mlbb-auto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Siapkan video replay di folder videos/
# videos/alpha_1.mp4
# videos/franco_1.mp4
# videos/tigreal_1.mp4
```

**Spesifikasi Video:**
- Resolusi: **2400×1080** (replay MLBB standar)
- Frame rate: **30 fps**
- Format: `.mp4`
- Minimap cropped otomatis via `layout.yaml` di `(80, 0, 350, 340)`

---

## 1️⃣ Labeling Dataset

Buat label hero dot di minimap untuk training YOLO.

```bash
./scripts/label_minimap.sh                     # semua video
./scripts/label_minimap.sh -v alpha_1.mp4       # video tertentu
```

**Controls:**

| Tombol | Aksi |
|--------|------|
| `B` | Mode BLUE (hero biru) |
| `R` | Mode RED (hero merah) |
| Left click | Tambah label sesuai mode |
| `N` | Next frame (auto-save) |
| `S` | Save + Next |
| `Q` | Quit |

**Tips:**
- Label minimal **500 frame** dari berbagai momen (early, mid, late game, teamfight)
- Setiap frame: label **semua hero** yang terlihat (5 biru + 5 merah)
- Frame tanpa hero (fog of war) — skip dengan N
- Variasikan video (alpha_1.mp4, franco_1.mp4, tigreal_1.mp4)

Output disimpan di `trainings/hero_detector/`:
```
trainings/hero_detector/
├── images/train/*.png    ← crop minimap
├── labels/train/*.txt    ← YOLO format (class cx cy w h)
└── data.yaml             ← config dataset
```

---

## 2️⃣ Training Model

```bash
./scripts/train.sh
```

Script otomatis:
- Mendeteksi GPU (CUDA > MPS > CPU)
- Jika sudah pernah training → **fine-tune** dari model sebelumnya
- Jika pertama kali → download pretrained `yolo11n.pt`
- Split 20% data ke validation set
- Early stopping jika tidak ada improvement

**Output:**
```
runs/detect/trainings/hero_detector/yolo11n_minimap/weights/
├── best.pt       ← model terbaik (PyTorch)
├── best.onnx     ← model terbaik (ONNX, faster)
└── last.pt       ← model epoch terakhir
```

**Monitoring mAP50:**

| mAP50 | Status |
|-------|--------|
| > 0.85 | 🟢 Excellent — siap production |
| 0.70–0.85 | 🟢 Good — detection reliable |
| 0.60–0.70 | 🟡 Decent — tracking dasar ok |
| 0.50–0.60 | 🟡 Low — perlu lebih banyak label |
| < 0.50 | 🔴 Poor — label 500+ frame dulu |

---

## 3️⃣ Inference (Test Detection)

```bash
# Test YOLO detection di video
./scripts/inference.sh
./scripts/inference.sh -v alpha_1.mp4
./scripts/inference.sh --conf 0.3     # atur confidence threshold
```

Menampilkan:
- Bounding box + confidence tiap hero
- mAP50 model
- FPS + inference time
- Kualitas model (color-coded)

---

## 4️⃣ Debug Vision (Full Pipeline)

Setelah model cukup akurat (mAP50 > 0.7), bisa dipakai di pipeline lengkap:

```python
from vision.detectors.minimap.minimap_hero_tracker import MinimapHeroTracker

# YOLO model otomatis kepakai jika ada
tracker = MinimapHeroTracker(yolo_model_path="trainings/hero_detector/yolo11n_minimap/weights/best.pt")

# Atau via debug_vision — edit debug_vision.py:
# tracker = MinimapHeroTracker(yolo_model_path="...")
```

Jalankan debug_vision:
```bash
python tools/debug_vision.py
```

---

## Struktur File

```
mlbb-auto/
├── videos/                          ← replay MLBB (.mp4)
├── trainings/hero_detector/         ← dataset + model
│   ├── images/train/                ← crop minimap untuk training
│   ├── labels/train/                ← label YOLO (.txt)
│   └── data.yaml                    ← config dataset
├── tools/
│   ├── label_minimap.py             ← labeling tool
│   ├── train_yolo.py                ← training script
│   └── inference.py                 ← test detection
├── scripts/
│   ├── label_minimap.sh             ← jalankan labeling
│   ├── train.sh                     ← jalankan training
│   └── inference.sh                 ← jalankan inference
├── vision/
│   └── detectors/minimap/
│       ├── minimap_hero_tracker.py  ← tracker (fallback HSV + velocity gate)
│       └── yolo_detector.py         ← YOLO detector wrapper
├── .gitignore
└── requirements.txt
```

## Catatan

- **Label tidak pernah hilang** saat training ulang — file `.txt` di `labels/train/` tetap aman
- **Training ulang = fine-tune** — melanjutkan dari `best.pt` sebelumnya (bukan dari nol)
- **Model files** di-ignore oleh git (`.pt`, `.onnx`, `trainings/`, `runs/`)
