#!/usr/bin/env python3
"""
train_skill_classifier.py — Training CNN untuk deteksi state skill icon.

Menggantikan heuristics CV (brightness, edge, HSV) dengan CNN classifier.
Model belajar langsung dari pixel icon skill (70x70):
  - ready      → skill siap pakai
  - cooldown   → skill dalam cooldown (overlay gelap)
  - available  → skill baru selesai cooldown (border highlight)
  - empty      → slot kosong

Dataset (dari tools/crop_skills_dataset.py):
  trainings/hero_skills/dataset/
    <video_name>/<slot>/<label>/*.png

Output:
  models/skill_classifier.onnx  ← langsung siap pakai di pipeline

Cara pakai:
  python tools/train_skill_classifier.py
  python tools/train_skill_classifier.py --epochs 100 --batch-size 32
  python tools/train_skill_classifier.py --resume --dataset trainings/hero_skills/dataset
"""

from __future__ import annotations

import os
import sys
import argparse
import csv
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ── Force CPU (ringan, cocok untuk training kecil) ──
DEVICE = torch.device("cpu")

# ── Constants ──
CLASSES = ["ready", "cooldown", "available", "empty", "locked"]
NUM_CLASSES = len(CLASSES)
INPUT_SIZE = 70  # px, sesuai ukuran crop dari layout.yaml

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE / "trainings" / "hero_skills" / "dataset"
CHECKPOINT_DIR = BASE / "trainings" / "hero_skills" / "checkpoints"
ONNX_OUTPUT = BASE / "models" / "skill_classifier.onnx"
LOG_PATH = BASE / "trainings" / "hero_skills" / "training_log.csv"


# ═══════════════════════════════════════════════════════════════════════
#  Model — CNN ringan untuk icon 70×70
# ═══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Sequential):
    """Conv3×3 → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SkillStateCNN(nn.Module):
    """
    CNN ringan (~63K params) untuk klasifikasi state icon skill 70×70.

    Arsitektur:
      Conv(3→16) → Pool → Conv(16→32) → Pool →
      Conv(32→64) → Pool → Conv(64→64) → Pool →
      GlobalAvgPool → FC(64→32) → Dropout → FC(32→4)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16), nn.MaxPool2d(2),     # 70→35
            ConvBlock(16, 32), nn.MaxPool2d(2),     # 35→17
            ConvBlock(32, 64), nn.MaxPool2d(2),     # 17→8
            ConvBlock(64, 64), nn.MaxPool2d(2),     # 8→4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════

class SkillDataset(Dataset):
    """Load skill icon crops from dataset directory.

    Struktur (hero-based):
      dataset/<hero>/<slot>/<label>/*.png

    Juga backward compat:
      dataset/<video>/<slot>/<label>/*.png  (tanpa hero)
      dataset/<video>/<hero>/<slot>/<label>/*.png  (video-bersarang)
    """

    def __init__(self, root: str | Path, augment: bool = True,
                 hero_filter: list[str] | None = None):
        """
        Args:
            root: Dataset root directory.
            augment: Enable data augmentation.
            hero_filter: Only load these heroes (None = all available).
        """
        self.samples: list[tuple[str, int, str]] = []  # (path, class_id, hero_name)
        self.augment = augment
        root = Path(root)
        self.hero_filter = [h.lower() for h in hero_filter] if hero_filter else None

        if not root.exists():
            raise FileNotFoundError(f"Dataset not found: {root}")

        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name == "debug_vision":
                continue

            entry_name = entry.name.lower()
            if self.hero_filter and entry_name not in self.hero_filter:
                continue

            # Cek apakah entry adalah hero (berisi slot di dalamnya)
            subdirs = [d for d in entry.iterdir() if d.is_dir()]
            slot_names = {d.name for d in subdirs}

            if slot_names & {"skill_1", "skill_2", "skill_3", "battle_spell"}:
                # Format baru: dataset/<hero>/<slot>/<label>/
                for slot_dir in subdirs:
                    self._load_class_dirs(slot_dir, entry.name)
            else:
                # Format lama: cek level di bawahnya
                for sub_dir in subdirs:
                    sub_name = sub_dir.name
                    if self.hero_filter and sub_name.lower() not in self.hero_filter:
                        continue
                    deeper = [d for d in sub_dir.iterdir() if d.is_dir()]
                    deeper_names = {d.name for d in deeper}
                    if deeper_names & {"skill_1", "skill_2", "skill_3", "battle_spell"}:
                        # Format: dataset/<video>/<hero>/<slot>/<label>/
                        for slot_dir in deeper:
                            self._load_class_dirs(slot_dir, sub_name)
                    else:
                        # Format paling lama: dataset/<video>/<slot>/<label>/
                        self._load_class_dirs(sub_dir, "unknown")

        if not self.samples:
            raise RuntimeError(f"No samples found in {root}")

        counts = {CLASSES[i]: 0 for i in range(NUM_CLASSES)}
        hero_counts = {}
        for _, cid, hero in self.samples:
            counts[CLASSES[cid]] += 1
            hero_counts[hero] = hero_counts.get(hero, 0) + 1
        print(f"  📊 Dataset: {len(self.samples)} samples from {len(hero_counts)} heroes")
        for c, n in counts.items():
            print(f"     {c}: {n}")
        if hero_counts:
            heroes_str = " | ".join(f"{h}={n}" for h, n in sorted(hero_counts.items()))
            print(f"     Heroes: {heroes_str}")

    def _load_class_dirs(self, slot_dir: Path, hero_name: str):
        """Load class subdirectories within a slot directory."""
        for class_dir in sorted(slot_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if label not in CLASSES:
                return
            class_id = CLASSES.index(label)
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((str(img_path), class_id, hero_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, class_id, hero_name = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

        # BGR → RGB → CHW (3, 70, 70) → normalize to [-1, 1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5

        # Augmentasi ringan untuk training
        if self.augment:
            # Random brightness
            if np.random.random() < 0.3:
                tensor = tensor * (1.0 + np.random.uniform(-0.15, 0.15))
                tensor = tensor.clamp(-1.0, 1.0)
            # Random flip
            if np.random.random() < 0.2:
                tensor = tensor.flip(-1)
            # Random grayscale overlay (simulate overlay detection)
            if np.random.random() < 0.1:
                tensor = tensor * 0.85 + 0.15 * tensor.mean(dim=0, keepdim=True)

        return tensor, class_id


# ═══════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class = {c: {"ok": 0, "n": 0} for c in CLASSES}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            for t, p in zip(y, pred):
                cn = CLASSES[t.item()]
                per_class[cn]["n"] += 1
                if t == p:
                    per_class[cn]["ok"] += 1
    acc = correct / total if total > 0 else 0
    class_acc = {c: v["ok"] / v["n"] if v["n"] > 0 else 0 for c, v in per_class.items()}
    return total_loss / total, acc, class_acc


# ═══════════════════════════════════════════════════════════════════════
#  ONNX Export
# ═══════════════════════════════════════════════════════════════════════

def export_onnx(model, output_path: str | Path):
    """Export PyTorch model → ONNX."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    # Coba pastikan onnxscript available
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        print("  ⚠️  onnxscript not installed, installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "onnxscript"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("     Installed!")

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
    )
    print(f"  ✅ ONNX exported: {output_path}")
    print(f"     Size: {output_path.stat().st_size / 1024:.1f} KB")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def list_available_heroes(dataset_dir: str | Path) -> list[str]:
    """List hero names available in dataset directory."""
    heroes = []
    root = Path(dataset_dir)
    if not root.exists():
        return heroes
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name == "debug_vision":
            continue
        # Cek apakah entry berisi slot skill
        subdirs = [d for d in entry.iterdir() if d.is_dir()]
        if {d.name for d in subdirs} & {"skill_1", "skill_2", "skill_3"}:
            heroes.append(entry.name)
        else:
            # Cek nested: <video>/<hero>/...
            for sub in subdirs:
                deeper = [d for d in sub.iterdir() if d.is_dir()]
                if {d.name for d in deeper} & {"skill_1", "skill_2", "skill_3"}:
                    heroes.append(sub.name)
                    break
    return sorted(set(heroes))


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN classifier for skill icon state detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR),
                        help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip ONNX export")
    parser.add_argument("--hero", type=str, default=None, nargs="+",
                        help="Train only specific hero(s). Available: (lihat di bawah)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (.pt file). Bisa lanjut training meski jumlah kelas berbeda.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║  Skill State CNN — Training              ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Device:   {DEVICE}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  LR:       {args.lr}")
    print(f"  Dataset:  {args.dataset}")
    print()

    # ── Tampilkan & pilih hero ──
    available = list_available_heroes(args.dataset)
    selected_heroes = args.hero

    if available:
        print(f"  📋 Heroes available ({len(available)}):")
        cols = 6
        for i, h in enumerate(available):
            marker = " ⬅" if selected_heroes and h in selected_heroes else ""
            print(f"     [{i+1:2d}] {h:12s}{marker}", end="")
            if (i + 1) % cols == 0 or i == len(available) - 1:
                print()
        print()

        # Pilih hero interaktif kalau tidak pakai --hero
        if selected_heroes is None and not args.yes:
            try:
                inp = input("  Pilih hero (nomor dipisah spasi, atau 'all'): ").strip()
            except (EOFError, KeyboardInterrupt):
                inp = ""
                print()
            if inp.lower() in ("all", "a", ""):
                selected_heroes = available
                print(f"  ✅ Semua hero ({len(selected_heroes)})")
            elif inp:
                indices = []
                for token in inp.replace(",", " ").split():
                    if token.isdigit():
                        idx = int(token) - 1
                        if 0 <= idx < len(available):
                            indices.append(idx)
                    else:
                        # Coba match nama hero
                        matches = [i for i, h in enumerate(available) if h.startswith(token.lower())]
                        indices.extend(matches)
                indices = sorted(set(indices))
                selected_heroes = [available[i] for i in indices]
                if selected_heroes:
                    print(f"  ✅ {len(selected_heroes)} hero dipilih: {', '.join(selected_heroes)}")
                else:
                    print("  ⚠️  Tidak ada hero valid, pakai semua")
                    selected_heroes = available
            print()
        elif selected_heroes is None:
            selected_heroes = available

    # ── Dataset ──
    print("📂 Loading dataset...")
    full = SkillDataset(args.dataset, augment=not args.no_augment,
                        hero_filter=selected_heroes)
    val_size = int(len(full) * args.val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print()

    # ── Konfirmasi ──
    hero_str = ", ".join(selected_heroes[:5])
    if len(selected_heroes) > 5:
        hero_str += f" +{len(selected_heroes)-5} others"
    print(f"  🎯 Train dengan {len(selected_heroes)} hero: {hero_str}")
    print(f"     {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print()
    if not args.yes:
        try:
            inp = input("  Mulai training? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            inp = "n"
            print()
        if inp not in ("", "y", "yes"):
            print("  ❌ Dibatalkan.")
            sys.exit(0)
        print()

    # ── Model ──
    model = SkillStateCNN()
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        old_state = ckpt.get("model_state_dict", ckpt)
        # Cek apakah jumlah kelas di checkpoint sama dengan sekarang
        old_out = model.classifier[-1].out_features
        new_out = NUM_CLASSES
        if old_out != new_out:
            print(f"  ⚠️  Checkpoint {old_out} kelas → model {new_out} kelas")
            print("      Load feature weights, re-init classifier...")
            # Load semua weights kecuali classifier layer
            filtered = {k: v for k, v in old_state.items()
                       if not k.startswith("classifier.")}
            model.load_state_dict(filtered, strict=False)
            # Re-init classifier (random)
            model.classifier[-1] = nn.Linear(32, new_out)
        else:
            model.load_state_dict(old_state)
        print(f"  ✅ Resumed from: {args.resume}")
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: {n_params:,} trainable params")
    print()

    # ── Training ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / "best.pt"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val_acc = 0.0
    t0 = time.time()

    print("🏋️  Training...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | "
          f"{'Val Loss':>8} {'Val Acc':>8} | Per-class Accuracy")
    print("  " + "-" * 80)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        cls_str = " ".join(f"{c}={v:.0%}" for c, v in class_acc.items())
        print(f"  {epoch:5d} | {train_loss:10.4f} {train_acc:9.2%} | "
              f"{val_loss:8.4f} {val_acc:8.2%} | {cls_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    elapsed = time.time() - t0
    print()
    print(f"✅ Selesai! ({elapsed:.0f}s)")
    print(f"   Best val accuracy: {best_val_acc:.2%}")
    print(f"   Checkpoint: {best_path}")

    # ── Load best + export ──
    if not args.no_export:
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print()
        print("📦 Exporting to ONNX...")
        export_onnx(model, ONNX_OUTPUT)
        print()
        print("🎯 Model siap! File:")
        print(f"   {ONNX_OUTPUT}")
        print()

        # Test inference with ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(ONNX_OUTPUT))
            inp = session.get_inputs()[0]
            out = session.get_outputs()[0]
            print(f"   ONNX Runtime test:")
            print(f"     Input:  {inp.name} {inp.shape}")
            print(f"     Output: {out.name} {out.shape}")
            dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
            result = session.run([out.name], {inp.name: dummy})
            print(f"     Inference: OK → {result[0].shape}")
        except Exception as e:
            print(f"   ⚠️  ONNX Runtime test skipped: {e}")

    print()
    print("🎯 Done!")


if __name__ == "__main__":
    main()
