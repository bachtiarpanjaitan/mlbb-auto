#!/bin/bash
# View Parquet — Pilih game state record yang mau dilihat
#
# Usage:
#   ./scripts/view_parquet.sh                          # Pilih file, tampilkan 10 baris
#   ./scripts/view_parquet.sh --head 3                  # Tampilkan 3 baris
#   ./scripts/view_parquet.sh --cols hero_name,hp_pct   # Filter kolom tertentu
#
# Tips filter kolom:
#   --cols all                               → semua 223 kolom
#   --cols hero_name,level,hp_pct,gold       → hero utama
#   --cols mm_norm_x,mm_norm_y               → posisi minimap (semua hero)
#   --cols blue_mm_1_norm_x,blue_mm_1_norm_y → hero biru slot 1
#   --cols jungle_1_name,jungle_1_norm_x     → jungle objectives

set -e
cd "$(dirname "$0")/.."

# Aktifkan virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

HEAD=10
COLS=""

# Parse args
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --head) HEAD="$2"; shift 2 ;;
    --cols) COLS="$2"; shift 2 ;;
    *) echo "❌ Argumen tidak dikenal: $1"; exit 1 ;;
  esac
done

# Cari file
FILES=$(ls data/game_states/*.parquet 2>/dev/null || true)
if [ -z "$FILES" ]; then
  echo "❌ Tidak ada file .parquet di data/game_states/"
  echo "   Record dulu: buka debug_vision.py, tekan P"
  exit 1
fi

echo "📁 Game State Records:"
echo ""

COUNT=0
while IFS= read -r f; do
  COUNT=$((COUNT + 1))
  STEM=$(basename "$f" .parquet)
  SIZE=$(stat -f "%z" "$f" 2>/dev/null | awk '{printf "%.1f KB", $1/1024}')
  echo "  [$COUNT] $STEM  ($SIZE)"
  eval "FILE_$COUNT=\$f"
done <<< "$FILES"

echo ""
read -p "Pilih nomor [1-$COUNT]: " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "$COUNT" ]; then
  echo "❌ Pilihan tidak valid"
  exit 1
fi

eval "SELECTED=\$FILE_$CHOICE"
SELECTED=$(basename "${SELECTED}" .parquet)

CMD="python tools/view_parquet.py \"$SELECTED\" --head $HEAD"
if [ -n "$COLS" ]; then
  CMD="$CMD --cols \"$COLS\""
fi

echo ""
echo "📊 Menampilkan: $SELECTED"
echo "────────────────────────────────────────"
eval "$CMD"
