#!/usr/bin/env python3
"""
train_cd_number.py — Training CNN untuk baca angka cooldown dari icon skill.

Input:  70×70 crop skill icon (sama dengan skill state classifier)
Output: Angka cooldown (0-120) atau None (kalau gak ada angka/not cooldown)

Dataset: generate synthetic (angka putih di background gelap, mirip UI MLBB)
+ bisa fine-tune dari data real nantinya.

Output ONNX: models/cd_number_classifier.onnx

Cara pakai:
  python tools/train_cd_number.py --epochs 30
"""

from __future__ import annotations

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── Force CPU ──
DEVICE = torch.device("cpu")

# ── Constants ──
INPUT_SIZE = 70
DIGIT_CLASSES = 11  # 0-9 + blank (10=blank/no digit)
MAX_DIGITS = 3      # maksimal 3 digit (max CD = 120)
MAX_CD = 120


# ═══════════════════════════════════════════════════════════════════════
#  Model — CNN kecil untuk digit recognition
# ═══════════════════════════════════════════════════════════════════════

class CDNumberCNN(nn.Module):
    """
    CNN untuk baca angka cooldown dari skill icon 70x70.

    Output: 3 digit (masing-masing 11 classes: 0-9 + blank)
    Angka dibaca dari kiri ke kanan.

    Contoh: "12" → [0, 1, 2]  (blank di awal)
            "5"  → [10, 0, 5]  (blank di kiri)
            ""   → [10, 10, 10] (blank semua)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 70→35
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 35→17
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 17→8
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 8→4
        )
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # 3 digit heads: digit_1 (puluhan), digit_2 (satuan), digit_3 (kosong/padding)
        self.digit_heads = nn.ModuleList([
            nn.Linear(64, DIGIT_CLASSES) for _ in range(MAX_DIGITS)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        return [head(x) for head in self.digit_heads]


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic Dataset
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_cd(cd_value: int) -> np.ndarray:
    """
    Generate 70x70 image mirip CD number di game MLBB.
    - Background: cooldown overlay (gelap + icon texture)
    - Angka: agak tipis, putih redup, center
    """
    # Cooldown overlay background (texture gelap + noise)
    b = random.randint(25, 55)
    img = np.full((INPUT_SIZE, INPUT_SIZE, 3), b, dtype=np.uint8)
    # Icon texture hint (acak)
    texture = np.random.randint(0, 15, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.85, texture, 0.15, 0)

    text = str(cd_value)

    # Font: HERSEY_PLAIN = lebih tipis (simulasi font game)
    # Scale lebih kecil biar angka gak terlalu gedhe
    scale = 0.7 if len(text) <= 1 else (0.6 if len(text) == 2 else 0.5)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 1)[0]
    tx = (INPUT_SIZE - text_size[0]) // 2
    ty = (INPUT_SIZE + text_size[1]) // 2

    # Angka: putih redup (180-200), tipis (thickness=1), biar mirip game
    brightness = random.randint(170, 210)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_PLAIN, scale,
                (brightness, brightness, brightness), 1, cv2.LINE_AA)

    # Kadang blur (simulasi rendering game)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


class SyntheticCDDataset(Dataset):
    """Generate synthetic cooldown number dataset."""

    def __init__(self, size: int = 10000, augment: bool = True,
                 real_data_dir: str | Path | None = None,
                 cache_dir: str | Path | None = None):
        self.size = size
        self.augment = augment
        self.cd_values = []
        self.real_data: list[tuple[np.ndarray, int]] = []
        self._cache: list[np.ndarray] | None = None
        self._cache_dir = Path(cache_dir) if cache_dir else None

        # Load real data jika ada
        if real_data_dir:
            real_path = Path(real_data_dir)
            if real_path.exists():
                for fname in sorted(real_path.glob("*.png")):
                    parts = fname.stem.split("_")
                    for p in parts:
                        if p.startswith("cd") and p[2:].isdigit():
                            val = int(p[2:])
                            img = cv2.imread(str(fname))
                            if img is not None:
                                self.real_data.append((img, val))
                            break
                print(f"  Loaded {len(self.real_data)} real samples from {real_path}")

        # Synthetic distribution
        sync_size = max(size - len(self.real_data), size // 2)

        # Cek cache synthetic di disk
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_files = sorted(self._cache_dir.glob("*.npy"))
            if len(cache_files) >= sync_size:
                import time
                t0 = time.time()
                self._cache = []
                for cf in cache_files[:sync_size]:
                    self._cache.append(np.load(str(cf)))
                    val = int(cf.stem.split("_")[0])
                    self.cd_values.append(val)
                print(f"  Loaded {len(self._cache)} cached synthetic ({time.time()-t0:.1f}s)")
                return  # Skip generation

        for _ in range(sync_size):
            r = random.random()
            if r < 0.70:
                v = random.randint(0, 30)
            elif r < 0.90:
                v = random.randint(31, 60)
            else:
                v = random.randint(61, MAX_CD)
            self.cd_values.append(v)
        self._generate_and_cache()

    def _generate_and_cache(self):
        """Generate synthetic data and cache to disk."""
        import time
        t0 = time.time()
        self._cache = []
        for i, v in enumerate(self.cd_values):
            img = generate_synthetic_cd(v)
            self._cache.append(img)
            if self._cache_dir:
                np.save(str(self._cache_dir / f"{v:03d}_{i:05d}.npy"), img)
        print(f"  Generated + cached {len(self._cache)} synthetic ({time.time()-t0:.0f}s)")

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        # Mix real + synthetic
        if self.real_data and idx < len(self.real_data):
            img, cd_val = self.real_data[idx]
            # Convert grayscale back to RGB (3 channel)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        else:
            sync_idx = idx - len(self.real_data) if self.real_data else idx
            sync_idx = min(sync_idx, len(self.cd_values) - 1)
            cd_val = self.cd_values[max(0, sync_idx)]
            img = generate_synthetic_cd(cd_val)

        # Konversi ke tensor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5

        # Label: [digit1, digit2, digit3] dengan blank=10
        s = str(cd_val).zfill(MAX_DIGITS)  # "012" untuk 12
        labels = []
        for ch in s:
            labels.append(int(ch))
        labels = torch.tensor(labels, dtype=torch.long)

        return tensor, labels, cd_val

    @staticmethod
    def decode(pred_digits: list[torch.Tensor]) -> int | None:
        """Convert model predictions → integer."""
        result = ""
        blank = DIGIT_CLASSES - 1
        started = False
        for head_pred in pred_digits:
            digit = int(torch.argmax(head_pred, dim=1)[0])
            if digit == blank:
                if started:
                    break
                continue
            started = True
            result += str(digit)
        if result:
            return int(result)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════


def auto_scan_hero_dataset(hero_dir: str | Path, cd_dir: str | Path,
                           cd_model_path: str | Path | None = None) -> int:
    """Scan hero_skills/dataset untuk cooldown crops → copy ke cd_number dataset."""
    import sys
    if 'vision' not in sys.modules:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import cv2; from vision.detectors.skills.cd_number import CDNumberDetector
    hero_dir = Path(hero_dir)
    cd_dir = Path(cd_dir)
    cd_dir.mkdir(parents=True, exist_ok=True)

    det = CDNumberDetector()
    if cd_model_path and os.path.isfile(str(cd_model_path)):
        det = CDNumberDetector()
    if not det.available:
        print("  ⚠️  CD model not available, skip auto-scan")
        return 0

    saved = 0
    for hero_entry in sorted(hero_dir.iterdir()):
        if not hero_entry.is_dir(): continue
        for slot_dir in sorted(hero_entry.iterdir()):
            if not slot_dir.is_dir(): continue
            cd_folder = slot_dir / 'cooldown'
            if not cd_folder.exists(): continue
            for img_path in sorted(cd_folder.glob('*.png')):
                img = cv2.imread(str(img_path))
                if img is None: continue
                cd_val = det.read(img)
                if cd_val is not None and 1 <= cd_val <= 120:
                    fname = f'{hero_entry.name}_{slot_dir.name}_cd{cd_val:03d}.png'
                    dst = cd_dir / fname
                    if not dst.exists():
                        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        saved += 1
    print(f"  📋 Auto-scan hero dataset: {saved} new CD samples")
    return saved

def train():
    BASE = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train CD number digit classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset-size", type=int, default=20000)
    parser.add_argument("--real-data", type=str, default=str(BASE / "trainings" / "cd_number" / "real_dataset"),
                        help="Directory with real CD number crops (png with _cdNNN_ in name)")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    ONNX_PATH = BASE / "models" / "cd_number_classifier.onnx"
    CHECKPOINT_DIR = BASE / "trainings" / "cd_number" / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════╗")
    print("║  CD Number Digit Classifier — Training   ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Device:   {DEVICE}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Dataset:  {args.dataset_size} synthetic")

    # ── Auto-scan hero_skills dataset untuk cooldown crops ──
    hero_dataset = BASE / "trainings" / "hero_skills" / "dataset"
    if hero_dataset.exists():
        auto_scan_hero_dataset(hero_dataset, args.real_data if os.path.isdir(args.real_data) else BASE / "trainings" / "cd_number" / "real_dataset")

    # ── Dataset ──
    real_path = args.real_data if os.path.isdir(args.real_data) else str(BASE / "trainings" / "cd_number" / "real_dataset")
    cache_dir = BASE / 'trainings' / 'cd_number' / 'cache'
    ds = SyntheticCDDataset(size=args.dataset_size, real_data_dir=real_path, cache_dir=cache_dir)
    val_size = int(len(ds) * 0.1)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"  Train:  {len(train_ds)} samples")
    print(f"  Val:    {len(val_ds)} samples")
    print()

    # ── Model ──
    model = CDNumberCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: {n_params:,} params, 3×{DIGIT_CLASSES} digit classes")
    print()

    # ── Training ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc = 0.0

    print("🏋️  Training...")
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'Acc':>7} | {'Acc (d1/d2/d3)':>18}")
    print("  " + "-" * 50)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = sum(criterion(o, y[:, i]) for i, o in enumerate(outputs))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        # Evaluate
        model.eval()
        correct_digits = [0, 0, 0]
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                val_loss += sum(criterion(o, y[:, i]) for i, o in enumerate(outputs)).item() * x.size(0)
                total += y.size(0)
                for i, o in enumerate(outputs):
                    correct_digits[i] += (o.argmax(1) == y[:, i]).sum().item()

        avg_loss = total_loss / len(train_ds)
        avg_val_loss = val_loss / len(val_ds)
        acc = [c / total for c in correct_digits]
        avg_acc = sum(acc) / 3

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")

        print(f"  {epoch:5d} | {avg_loss:8.4f} | {avg_acc:7.2%} | "
              f"{acc[0]:.2%}/{acc[1]:.2%}/{acc[2]:.2%}")

    print(f"\n✅ Selesai! Best val acc: {best_acc:.2%}")

    # ── Export ONNX ──
    if not args.no_export:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "best.pt", map_location=DEVICE))
        model.eval()

        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
        ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model, dummy, str(ONNX_PATH),
            input_names=["input"],
            output_names=["digit_0", "digit_1", "digit_2"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=18,
        )
        print(f"  ✅ ONNX: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024:.1f} KB)")

        # Test
        import onnxruntime as ort
        session = ort.InferenceSession(str(ONNX_PATH))
        test_input = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        outputs = session.run(None, {"input": test_input})
        # Decode
        blank = DIGIT_CLASSES - 1
        result = ""
        started = False
        for out in outputs:
            digit = int(np.argmax(out[0]))
            if digit == blank:
                if started:
                    break
                continue
            started = True
            result += str(digit)

        print(f"     Test inference: outputs={[o.shape for o in outputs]}")
        print(f"     Example decode: '{result}' (expected: random)")

    print("\n🎯 Done!")


if __name__ == "__main__":
    train()
