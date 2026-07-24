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
selected_vertex: int | None = None  # index vertex yang di-hover (untuk drag)
drag_state: str | None = None     # "vertex" | "polygon" | None
drag_target: int = -1             # index region yang di-drag
drag_vertex: int = -1             # index vertex yang di-drag
drag_start_mouse: tuple[int,int] = (0, 0)  # posisi mouse awal (display coords)
drag_start_points: list = []      # snapshot titik awal region saat drag mulai

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


def find_nearest_vertex(pt: tuple[int,int], poly: list, radius: int = 15) -> int | None:
    """Cari vertex terdekat dalam radius (skala DISPLAY)."""
    for i, p in enumerate(poly):
        dx = abs(p[0] * SCALE - pt[0])
        dy = abs(p[1] * SCALE - pt[1])
        if dx <= radius and dy <= radius:
            return i
    return None


def find_nearest_edge(pt: tuple[int,int], poly: list, radius_sq: int = 400) -> int | None:
    """Cari segmen garis terdekat — return index vertex SEBELUM segmen (skala DISPLAY).
    
    radius_sq = 400 → jarak maks 20px dari garis.
    """
    n = len(poly)
    if n < 2:
        return None
    best_idx = None
    best_dist = float("inf")
    for i in range(n):
        j = (i + 1) % n  # next vertex (wrap around)
        ax, ay = poly[i][0] * SCALE, poly[i][1] * SCALE
        bx, by = poly[j][0] * SCALE, poly[j][1] * SCALE
        # Distance from point pt to line segment ab
        abx, aby = bx - ax, by - ay
        t = ((pt[0] - ax) * abx + (pt[1] - ay) * aby) / (abx * abx + aby * aby + 1e-10)
        t = max(0, min(1, t))
        cx, cy = ax + t * abx, ay + t * aby
        dx, dy = pt[0] - cx, pt[1] - cy
        d_sq = dx * dx + dy * dy
        if d_sq < radius_sq and d_sq < best_dist:
            best_dist = d_sq
            best_idx = i
    return best_idx


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

        # Border
        cv2.polylines(image, [arr], True, color, 2)
        
        # Titik vertex — highlight yang sedang di-drag/hover
        for vi, p in enumerate(pts):
            is_dragging = (drag_state == "vertex" and drag_target == i and drag_vertex == vi)
            is_hover = (selected_idx == i and selected_vertex == vi)
            if is_dragging:
                cv2.circle(image, tuple(p), 8, (0, 255, 255), -1)  # kuning terang
            elif is_hover:
                cv2.circle(image, tuple(p), 7, (255, 255, 255), 2)  # putih outline
            else:
                cv2.circle(image, tuple(p), 5, color, -1)

        # Highlight polygon saat di-drag
        if drag_state == "polygon" and drag_target == i:
            cv2.polylines(image, [arr], True, (0, 255, 255), 3)

        # Highlight edge yang di-hover (untuk insert titik)
        if selected_idx == i and selected_vertex is None and not drag_state:
            pass  # handled globally below

    # --- Global edge hover indicator (for insert point preview) ---
    if selected_idx is not None and selected_vertex is None and not drag_state:
        hover_pts = regions[selected_idx]["points"]
        ei = find_nearest_edge((selected_mx, selected_my), hover_pts)
        if ei is not None:
            j = (ei + 1) % len(hover_pts)
            ax, ay = hover_pts[ei][0] * SCALE, hover_pts[ei][1] * SCALE
            bx, by = hover_pts[j][0] * SCALE, hover_pts[j][1] * SCALE
            abx, aby = bx - ax, by - ay
            t = ((selected_mx - ax) * abx + (selected_my - ay) * aby) / (abx * abx + aby * aby + 1e-10)
            t = max(0, min(1, t))
            cx, cy = int(ax + t * abx), int(ay + t * aby)
            # Highlight the edge
            cv2.line(image, (ax, ay), (bx, by), (0, 255, 255), 3, cv2.LINE_AA)
            # Show insert point
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)
            # Label
            cv2.putText(image, "Klik untuk tambah titik", (cx + 14, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

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
        "Left Click  : Tambah titik / Drag vertex/polygon",
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
selected_mx: int = 0
selected_my: int = 0
def mouse(event, x, y, flags, param):
    global current_points, selected_idx, selected_vertex
    global drag_state, drag_target, drag_vertex, drag_start_mouse, drag_start_points

    # Convert display coords to original
    ox, oy = x // SCALE, y // SCALE

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_points:
            # Sedang membuat polygon baru → tambah titik
            current_points.append((ox, oy))
            redraw()
            return

        # Cek apakah klik di vertex yang sudah ada (mulai drag vertex)
        found_vertex = False
        for i, region in enumerate(regions):
            vi = find_nearest_vertex((x, y), region["points"])
            if vi is not None:
                drag_state = "vertex"
                drag_target = i
                drag_vertex = vi
                drag_start_mouse = (x, y)
                drag_start_points = [list(region["points"][vi])]
                selected_idx = i
                selected_vertex = vi
                found_vertex = True
                redraw()
                print(f"[DRAG] Vertex {vi} region '{region['name']}'")
                break

        if found_vertex:
            return

        # Cek apakah klik di EDGE polygon → insert titik baru
        edge_target = None
        edge_idx = None
        for i, region in enumerate(regions):
            ei = find_nearest_edge((x, y), region["points"])
            if ei is not None:
                edge_target = i
                edge_idx = ei
                break

        if edge_target is not None:
            # Insert new vertex after edge_idx
            region = regions[edge_target]
            j = (edge_idx + 1) % len(region["points"])
            # Interpolate position on the edge
            ax, ay = region["points"][edge_idx]
            bx, by = region["points"][j]
            abx, aby = bx * SCALE - ax * SCALE, by * SCALE - ay * SCALE
            t = ((x - ax * SCALE) * abx + (y - ay * SCALE) * aby) / (abx * abx + aby * aby + 1e-10)
            t = max(0.3, min(0.7, t))  # constrain to middle portion
            new_px = int(ax + t * (bx - ax))
            new_py = int(ay + t * (by - ay))
            region["points"].insert(j, [new_px, new_py])
            selected_idx = edge_target
            selected_vertex = j
            redraw()
            print(f"[INSERT] Titik baru di region '{region['name']}' (now {len(region['points'])} pts)")
            return

        # Cek apakah klik di dalam polygon → drag seluruh polygon
        for i, region in enumerate(regions):
            if point_in_polygon((ox, oy), region["points"]):
                drag_state = "polygon"
                drag_target = i
                drag_start_mouse = (x, y)
                drag_start_points = [list(p) for p in region["points"]]
                selected_idx = i
                selected_vertex = None
                redraw()
                print(f"[DRAG] Polygon '{region['name']}'")
                return

        # Klik di luar → mulai titik baru
        current_points.append((ox, oy))
        redraw()

    elif event == cv2.EVENT_MOUSEMOVE:
        # Track mouse for edge hover indicator
        global selected_mx, selected_my
        selected_mx, selected_my = x, y

        if drag_state == "vertex":
            # Drag vertex — delta dari posisi mouse awal
            dmx, dmy = drag_start_mouse
            dx = (x - dmx) // SCALE
            dy = (y - dmy) // SCALE
            region = regions[drag_target]
            orig_x, orig_y = drag_start_points[0]
            region["points"][drag_vertex] = [
                max(0, orig_x + dx),
                max(0, orig_y + dy)
            ]
            redraw()
        elif drag_state == "polygon":
            # Drag seluruh polygon — delta dari posisi mouse awal
            dmx, dmy = drag_start_mouse
            dx = (x - dmx) // SCALE
            dy = (y - dmy) // SCALE
            region = regions[drag_target]
            for vi in range(len(region["points"])):
                orig_x, orig_y = drag_start_points[vi]
                region["points"][vi] = [
                    max(0, orig_x + dx),
                    max(0, orig_y + dy)
                ]
            redraw()
        else:
            # Hover highlight
            new_sel = None
            new_vtx = None
            # Cek vertex dulu
            for i, region in enumerate(regions):
                vi = find_nearest_vertex((x, y), region["points"])
                if vi is not None:
                    new_sel = i
                    new_vtx = vi
                    break
            if new_vtx is None:
                # Cek polygon body
                for i, region in enumerate(regions):
                    if point_in_polygon((ox, oy), region["points"]):
                        new_sel = i
                        break
            if new_sel != selected_idx or new_vtx != selected_vertex:
                selected_idx = new_sel
                selected_vertex = new_vtx
                redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        if drag_state:
            action = "vertex" if drag_state == "vertex" else "polygon"
            print(f"[DRAG] {action} selesai")
            drag_state = None
            drag_target = -1
            drag_vertex = -1
            drag_start_points = []
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
            # Save existing regions (after drag edits) tanpa prompt nama
            if not regions:
                print("[SAVE] Tidak ada region untuk disimpan.")
                continue
            OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(regions, f, indent=4)
            print(f"[SAVE] {len(regions)} region(s) disimpan ke {OUTPUT_JSON}")
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