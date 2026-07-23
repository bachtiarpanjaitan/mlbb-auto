import cv2
import json
import os
import colorsys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH   = PROJECT_ROOT / "assets" / "minimap.png"
OUTPUT_JSON  = PROJECT_ROOT / "assets" / "databases" / "regions.json"

# -----------------------------
# Warna otomatis — Golden Ratio hue distribution
# Setiap index menghasilkan warna cerah & unik (tidak pernah gelap)
# -----------------------------
_GOLDEN_RATIO = 0.618033988749895   # 1/φ
ALPHA = 0.35   # transparansi fill polygon tertutup


def get_color(idx: int) -> tuple[int, int, int]:
    """
    Hasilkan warna BGR yang cerah dan unik berdasarkan index.
    Hue didistribusikan via golden ratio sehingga warna berurutan
    selalu maksimal berbeda satu sama lain.
    """
    hue = (idx * _GOLDEN_RATIO) % 1.0
    sat = 0.82 + (idx % 3) * 0.06   # variasi sedikit: 0.82 / 0.88 / 0.94
    val = 0.92 + (idx % 2) * 0.06   # variasi sedikit: 0.92 / 0.98
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR

# -----------------------------
# Load Image
# -----------------------------
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Gambar '{IMAGE_PATH}' tidak ditemukan.")

orig = image.copy()   # ukuran asli (untuk koordinat yang disimpan)

# -----------------------------
# Scale display 9x (3x lebih besar dari 3x sebelumnya)
# -----------------------------
SCALE = 5
h, w  = orig.shape[:2]
DISP_W, DISP_H = w * SCALE, h * SCALE
base  = cv2.resize(orig, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
image = base.copy()

# -----------------------------
# State
# -----------------------------
current_points: list[tuple[int,int]] = []   # koordinat skala ASLI
regions: list[dict] = []
selected_idx: int | None = None   # index polygon yang di-hover untuk dihapus

# Load existing JSON if available
if OUTPUT_JSON.exists():
    with open(OUTPUT_JSON, "r") as f:
        regions = json.load(f)
    print(f"[INFO] Loaded {len(regions)} region(s) dari {OUTPUT_JSON}")


# -----------------------------
# Helper
# -----------------------------



def point_in_polygon(pt: tuple[int,int], poly: list) -> bool:
    """Cek apakah titik pt berada di dalam polygon (list of [x,y])."""
    pts_arr = np.array(poly, dtype=np.float32)
    result  = cv2.pointPolygonTest(pts_arr, (float(pt[0]), float(pt[1])), False)
    return result >= 0


# -----------------------------
# Draw
# -----------------------------
def redraw():
    global image
    image = base.copy()
    overlay = image.copy()

    # --- Gambar region yang sudah tersimpan ---
    for i, region in enumerate(regions):
        pts_orig = region["points"]
        # scale ke layar display
        pts  = [[p[0]*SCALE, p[1]*SCALE] for p in pts_orig]
        color = get_color(i)
        arr  = np.array(pts, dtype=np.int32)

        # Fill semi-transparan
        cv2.fillPoly(overlay, [arr], color)

        # Border & titik
        cv2.polylines(image, [arr], True, color, 2)
        for p in pts:
            cv2.circle(image, tuple(p), 4, color, -1)

        # Label nama di tengah polygon (centroid)
        M = cv2.moments(arr)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pts[0]

        # Font 3x lebih besar (0.45 → 1.35, thickness 1 → 3)
        text = region["name"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.35, 3)
        cv2.putText(
            image, text,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.35,
            color, 3, cv2.LINE_AA
        )

        # Highlight bila sedang dipilih untuk dihapus
        if i == selected_idx:
            cv2.polylines(image, [arr], True, (0, 0, 255), 3)
            cv2.putText(
                image, "[DEL] hapus",
                (pts[0][0], pts[0][1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 255), 1, cv2.LINE_AA
            )

    # Blend overlay fill
    cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0, image)

    # --- Shortcut keys overlay ---
    shortcuts = [
        "Left Click  : Tambah titik",
        "Right Click : Tutup polygon",
        "Z           : Undo titik",
        "C           : Clear polygon",
        "D           : Hapus polygon",
        "S           : Simpan JSON",
        "ESC         : Keluar",
    ]
    y0 = 30
    for i, txt in enumerate(shortcuts):
        cv2.putText(
            image, txt,
            (12, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    # --- Gambar polygon yang sedang dibuat ---
    n = len(current_points)
    cur_color = get_color(len(regions))

    # scale titik aktif ke layar display
    cur_disp = [(p[0]*SCALE, p[1]*SCALE) for p in current_points]

    for p in cur_disp:
        cv2.circle(image, p, 5, cur_color, -1)

    if n >= 2:
        cv2.polylines(
            image,
            [np.array(cur_disp, dtype=np.int32)],
            False, cur_color, 2
        )


# -----------------------------
# Mouse Callback
# -----------------------------
def mouse(event, x, y, flags, param):
    global current_points, selected_idx

    if event == cv2.EVENT_MOUSEMOVE:
        # Highlight polygon yang sedang di-hover (untuk petunjuk delete)
        # konversi ke skala asli untuk hit-test
        ox, oy = x // SCALE, y // SCALE
        new_sel = None
        for i, region in enumerate(regions):
            if point_in_polygon((ox, oy), region["points"]):
                new_sel = i
                break
        if new_sel != selected_idx:
            selected_idx = new_sel
            redraw()

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Simpan koordinat dalam skala ASLI
        ox, oy = x // SCALE, y // SCALE
        current_points.append((ox, oy))
        redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Tutup polygon aktif → minta nama
        if len(current_points) < 3:
            print("[WARN] Polygon minimal 3 titik.")
            return

        # Tampilkan preview closed polygon (di skala display)
        redraw()
        cur_color  = get_color(len(regions))
        cur_disp   = np.array([(p[0]*SCALE, p[1]*SCALE) for p in current_points], dtype=np.int32)
        cv2.polylines(image, [cur_disp], True, cur_color, 2)
        cv2.imshow("Region Editor", image)

        name = input("\nNama Region : ").strip()
        if not name:
            print("[WARN] Nama tidak boleh kosong. Polygon dibatalkan.")
            current_points = []
            redraw()
            return

        regions.append({
            "id":     name.lower().replace(" ", "_"),
            "name":   name,
            "points": [list(p) for p in current_points]
        })

        current_points = []
        redraw()
        print(f"[OK] Region '{name}' ditambahkan (total: {len(regions)}).")


# -----------------------------
# Window
# -----------------------------
cv2.namedWindow("Region Editor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Region Editor", DISP_W, DISP_H)
cv2.setMouseCallback("Region Editor", mouse)
redraw()

print("""
==============================
 Region Editor — MLBB Minimap
==============================
 Left Click   : Tambah titik
 Right Click  : Tutup polygon (simpan sementara)
 Z            : Undo titik terakhir
 C            : Clear polygon aktif
 D            : Hapus polygon yang di-hover
 S            : Simpan ke JSON
 ESC          : Keluar
==============================
""")

# -----------------------------
# Main Loop
# -----------------------------
while True:
    cv2.imshow("Region Editor", image)
    key = cv2.waitKey(20) & 0xFF

    # ESC → keluar
    if key == 27:
        break

    # Z → undo titik terakhir
    elif key == ord('z'):
        if current_points:
            removed = current_points.pop()
            redraw()
            print(f"[UNDO] Titik {removed} dihapus. Sisa: {len(current_points)} titik.")
        else:
            print("[UNDO] Tidak ada titik untuk di-undo.")

    # C → clear polygon aktif
    elif key == ord('c'):
        current_points = []
        redraw()
        print("[CLEAR] Polygon aktif dibersihkan.")

    # D → hapus polygon yang di-hover
    elif key == ord('d'):
        if selected_idx is not None and 0 <= selected_idx < len(regions):
            removed_name = regions[selected_idx]["name"]
            regions.pop(selected_idx)
            selected_idx = None
            redraw()
            print(f"[DELETE] Region '{removed_name}' dihapus. Sisa: {len(regions)} region.")
        else:
            print("[DELETE] Arahkan kursor ke dalam polygon yang ingin dihapus dulu.")

    # S → simpan polygon aktif + prompt nama → simpan ke JSON
    elif key == ord('s'):
        if not current_points:
            print("[SAVE] Tidak ada polygon aktif untuk disimpan. Buat polygon dulu (left-click).")
            continue

        if len(current_points) < 3:
            print("[SAVE] Polygon minimal 3 titik.")
            continue

        name = input("\nNama Region : ").strip()
        if not name:
            print("[SAVE] Nama tidak boleh kosong. Batal simpan.")
            continue

        # snake_case id
        region_id = name.lower().replace(" ", "_").replace("-", "_")

        # cek duplikat id
        if any(r["id"] == region_id for r in regions):
            print(f"[SAVE] ID '{region_id}' sudah ada. Gunakan nama lain.")
            continue

        regions.append({
            "id":     region_id,
            "name":   name,
            "points": [list(p) for p in current_points]
        })
        current_points = []
        redraw()

        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(regions, f, indent=4)
        print(f"[SAVE] Region '{name}' (id: {region_id}) ditambahkan & disimpan. Total: {len(regions)} region.")

cv2.destroyAllWindows()