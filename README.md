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
```

## Usage

Tool utama untuk menjalankan pipeline dan utilitas adalah `main.py`.

```bash
# Buka menu interaktif
python main.py

# Atau jalankan tool tertentu langsung
python main.py --tool [nomor_tool atau nama_tool]
```

## Daftar Tool (`main.py`)

| No | Nama Tool | Deskripsi |
|----|-----------|-----------|
| 1 | Label Minimap (Manual) | Label hero & jungle dot secara manual |
| 2 | Auto-Label Heroes (YOLO) | Deteksi hero otomatis pakai YOLO |
| 3 | Auto-Label Jungle (Template) | Deteksi jungle otomatis (template matching) |
| 4 | Auto-Label Full | Gabungan hero (YOLO) + jungle (template) |
| 5 | Train YOLO Model | Training YOLOv11n |
| 6 | Run Inference | Jalankan YOLO detection pada video |
| 7 | Debug Vision | Player video dengan overlay deteksi |
| 8 | Inspect Dataset | Lihat statistik & sample dataset |
| 9 | Clean Dataset | Bersihkan label kosong |
| 10 | Region Editor | Draw polygon regions di minimap |
| 11 | Compress Video | Kompres ukuran file video |
| 12 | Crawl Game Data | Download database game |
| 13 | Extract Jungle Templates | Crop template jungle |
| 14 | Simulate Minimap | Replay posisi hero dari data |
| 15 | Dataset Status | Ringkasan distribusi dataset |
| 16 | Interactive Label Tool | GUI untuk labeling minimap |
| 17 | Train Skill Classifier | Training deteksi cooldown skill |
| 18 | Train CD Number | Training OCR angka cooldown |
| 19 | Crop Skills Dataset | Crop dataset icon skill |
| 20 | Export to Kaggle | Packaging untuk Kaggle |

### Quick Actions

| Key | Nama | Deskripsi |
|-----|------|-----------|
| a | Auto-Label ALL | Auto-label full semua video |
| b | Full Pipeline | Label -> Train -> Export |
| x | Exit | Keluar dari menu |

## Struktur File

```
mlbb-auto/
├── videos/                          ← replay MLBB (.mp4)
├── trainings/hero_detector/         ← dataset + model
├── tools/                           ← script utilitas
├── scripts/                         ← script bash
├── vision/                          ← core pipeline
├── main.py                          ← entry point utama
├── requirements.txt
└── .gitignore
```
