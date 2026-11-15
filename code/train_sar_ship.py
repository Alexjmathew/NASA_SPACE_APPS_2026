#!/usr/bin/env python3
"""
SAR ship detection training (SSDD boxes + SL-SSDD sea/land masks) with Ultralytics YOLO.

IMPORTANT: SL-SSDD (github.com/Han-Ke/SL-SSDD) only contains sea/land MASKS (PNG, 0 = land, 255 = sea),
named like the SSDD images. It has NO ship boxes. Ship boxes come from the original SSDD (VOC XML).
So this script uses:
    SSDD     -> images + ship boxes   (training targets)
    SL-SSDD  -> land masks            (used to split val results into offshore / inshore and, optionally,
                                       to suppress detections on land at inference)

Steps
    python train_sar_ship.py prepare --ssdd /data/SSDD --slssdd /data/SL-SSDD --out /data/ssdd_yolo
    python train_sar_ship.py train   --data /data/ssdd_yolo/ssdd.yaml --model yolov8s.pt --epochs 200
    python train_sar_ship.py eval    --weights runs/ssdd/train/weights/best.pt --data /data/ssdd_yolo/ssdd.yaml
    python train_sar_ship.py predict --weights runs/ssdd/train/weights/best.pt --source img_or_folder --slssdd-masks /data/SL-SSDD

Install:  pip install ultralytics opencv-python numpy pyyaml
Colab  :  !pip install ultralytics   then run the commands above with ! in front.
"""
import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------------------------------
# 1. Dataset preparation
# ------------------------------------------------------------------------------------------
def index_files(root, exts):
    """stem -> path for every file under root with an allowed extension (first one wins)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.setdefault(p.stem, p)
    return out


def parse_voc(xml_path):
    """Return (width, height, [(xmin, ymin, xmax, ymax), ...]) from a Pascal-VOC xml (SSDD format)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2, y2))
    return w, h, boxes


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 1 or bh < 1:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def is_ssdd_test(stem):
    """Official SSDD split: image numbers ending in 1 or 9 are the test set (~20 %)."""
    digits = "".join(c for c in stem if c.isdigit())
    return bool(digits) and digits[-1] in ("1", "9")


def prepare(args):
    ssdd, out = Path(args.ssdd), Path(args.out)
    images = index_files(ssdd, IMG_EXT)
    xmls = index_files(ssdd, {".xml"})
    masks = index_files(args.slssdd, {".png", ".bmp", ".tif", ".tiff", ".jpg"}) if args.slssdd else {}
    # SL-SSDD masks live in their own folder; never treat them as SSDD images
    if args.slssdd and Path(args.slssdd).resolve() != ssdd.resolve():
        mask_dirs = {p.parent for p in masks.values()}
        images = {s: p for s, p in images.items() if p.parent not in mask_dirs}
    stems = sorted(s for s in images if s in xmls)
    if not stems:
        sys.exit("No image/xml pairs found under --ssdd. Expected SSDD JPEGImages/ + Annotations/*.xml")
    print(f"SSDD images with annotations: {len(stems)} | SL-SSDD masks found: {len(masks)}")

    if args.split == "official":
        test = {s for s in stems if is_ssdd_test(s)}
    else:
        rnd = random.Random(args.seed)
        shuffled = stems[:]
        rnd.shuffle(shuffled)
        test = set(shuffled[: int(0.2 * len(shuffled))])
    print(f"train {len(stems) - len(test)} | val/test {len(test)} ({args.split} split)")

    if out.exists():
        shutil.rmtree(out)
    n_boxes = n_missing_mask = 0
    for s in stems:
        sp = "val" if s in test else "train"
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (out / "masks" / sp).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(images[s]), cv2.IMREAD_COLOR)
        if img is None:
            print("unreadable:", images[s]); continue
        h, w = img.shape[:2]
        _, _, boxes = parse_voc(xmls[s])
        cv2.imwrite(str(out / "images" / sp / f"{s}.png"), img)          # lossless copy
        lines = []
        for b in boxes:
            y = to_yolo(b, w, h)
            if y:
                lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*y))
        (out / "labels" / sp / f"{s}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)
        if s in masks:
            m = cv2.imread(str(masks[s]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out / "masks" / sp / f"{s}.png"), m)     # 0 = land, 255 = sea
            else:
                n_missing_mask += 1
        else:
            n_missing_mask += 1
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": {0: "ship"}}
    (out / "ssdd.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"ship boxes: {n_boxes} | images without a land mask: {n_missing_mask}")
    print("dataset yaml:", out / "ssdd.yaml")



# ------------------------------------------------------------------------------------------
# 1b. Generic preparation for any Kaggle-style SAR ship dataset (YOLO txt / VOC xml / COCO json)
# ------------------------------------------------------------------------------------------
SPLIT_WORDS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test"}


def split_of(path, root):
    """Guess train/val/test from any folder or file name along the path (None if not stated)."""
    for part in reversed(Path(path).relative_to(root).parts):
        for tok in Path(part).stem.lower().replace("-", "_").split("_"):
            if tok in SPLIT_WORDS:
                return SPLIT_WORDS[tok]
    return None


def read_yolo_boxes(txt):
    """YOLO detect lines (cls cx cy w h) -> normalised boxes; polygon lines (cls x1 y1 x2 y2 ...) -> their bbox."""
    out = []
    for ln in Path(txt).read_text().strip().splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        nums = [float(x) for x in v[1:]]
        if len(nums) == 4:
            out.append(tuple(nums))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            xs, ys = nums[0::2], nums[1::2]
            out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
    return out


def prepare_auto(args):
    root, out = Path(args.src), Path(args.out)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in all_files if p.suffix.lower() in IMG_EXT]
    jsons, xmls, txts = ([p for p in all_files if p.suffix.lower() == e] for e in (".json", ".xml", ".txt"))
    if not images:
        sys.exit(f"No images found under {root}")
    by_stem = {}
    for p in images:
        by_stem.setdefault(p.stem, []).append(p)
    by_name = {}
    for p in images:
        by_name.setdefault(p.name, []).append(p)

    def pick(cands, ref):
        if len(cands) == 1:
            return cands[0]
        sp = split_of(ref, root)
        for c in cands:
            if sp and split_of(c, root) == sp:
                return c
        return cands[0]

    items = []          # (image_path, [(cx,cy,w,h) normalised], split or None)
    fmt = None
    # --- COCO ---
    coco = []
    for j in jsons:
        try:
            d = json.loads(j.read_text())
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                coco.append((j, d))
        except Exception:
            pass
    if coco:
        fmt = "COCO json"
        for j, d in coco:
            sp = split_of(j, root)
            anns = {}
            for a in d["annotations"]:
                if a.get("iscrowd"):
                    continue
                anns.setdefault(a["image_id"], []).append(a["bbox"])          # [x, y, w, h] pixels
            for im in d["images"]:
                c = by_name.get(Path(im["file_name"]).name)
                if not c:
                    continue
                W, H = im.get("width"), im.get("height")
                path = pick(c, j)
                if not (W and H):
                    hh, ww = cv2.imread(str(path)).shape[:2]
                    W, H = ww, hh
                bx = [((x + w / 2) / W, (y + h / 2) / H, w / W, h / H) for x, y, w, h in anns.get(im["id"], [])]
                items.append((path, bx, sp or split_of(path, root)))
    # --- VOC ---
    if not items:
        voc = {}
        for x in xmls:
            try:
                if ET.parse(x).getroot().find("object") is not None or ET.parse(x).getroot().find("size") is not None:
                    voc.setdefault(x.stem, []).append(x)
            except Exception:
                pass
        if voc:
            fmt = "Pascal VOC xml"
            for stem, c in by_stem.items():
                for img in c:
                    xs = voc.get(stem)
                    if not xs:
                        continue
                    xp = pick(xs, img)
                    w, h, boxes = parse_voc(xp)
                    if not (w and h):
                        hh, ww = cv2.imread(str(img)).shape[:2]
                        w, h = ww, hh
                    bx = [y for y in (to_yolo(b, w, h) for b in boxes) if y]
                    items.append((img, bx, split_of(img, root)))
    # --- YOLO txt ---
    if not items:
        lab = {}
        for t in txts:
            lab.setdefault(t.stem, []).append(t)
        for stem, c in by_stem.items():
            for img in c:
                ts = lab.get(stem)
                if not ts:
                    continue
                tp = pick(ts, img)
                try:
                    items.append((img, read_yolo_boxes(tp), split_of(img, root)))
                except ValueError:
                    continue
        if items:
            fmt = "YOLO txt"
    if not items:
        sys.exit("Could not find annotations (looked for COCO json, VOC xml, YOLO txt). "
                 "Print the folder tree and adapt the loader.")
    print(f"Detected format: {fmt} | annotated images: {len(items)} | "
          f"boxes: {sum(len(b) for _, b, _ in items)} | empty images: {sum(1 for _, b, _ in items if not b)}")

    # --- splits: respect train/val/test folders if present, otherwise random 80/20 ---
    known = {sp for _, _, sp in items if sp}
    rnd = random.Random(args.seed)
    if known and None not in {sp for _, _, sp in items} and "train" in known and (known & {"val", "test"}):
        assign = lambda sp: "train" if sp == "train" else "val"
        if "val" in known and "test" in known:                      # keep real val for training-time validation
            assign = lambda sp: "train" if sp == "train" else ("val" if sp == "val" else None)
        print("Using the dataset's own split folders:", sorted(known))
        order = items
        split_names = [assign(sp) for _, _, sp in order]
    else:
        idx = list(range(len(items)))
        rnd.shuffle(idx)
        nv = int(0.2 * len(items))
        vs = set(idx[:nv])
        order = items
        split_names = ["val" if i in vs else "train" for i in range(len(items))]
        print("No usable split folders - random 80/20 split")

    if out.exists():
        shutil.rmtree(out)
    cnt = {"train": 0, "val": 0}
    nbox = {"train": 0, "val": 0}
    for i, ((img_path, boxes, _), sp) in enumerate(zip(order, split_names)):
        if sp is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        name = f"{i:06d}"
        cv2.imwrite(str(out / "images" / sp / f"{name}.png"), img)
        lines = []
        for cx, cy, w, h in boxes:
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {min(w, 1):.6f} {min(h, 1):.6f}")
        (out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
        cnt[sp] += 1
        nbox[sp] += len(lines)
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": {0: "ship"}}
    (out / "ssdd.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"train: {cnt['train']} images / {nbox['train']} ships | val: {cnt['val']} images / {nbox['val']} ships")
    print("dataset yaml:", out / "ssdd.yaml")

# ------------------------------------------------------------------------------------------
# 2. Training
# ------------------------------------------------------------------------------------------
def train(args):
    from ultralytics import YOLO
    last = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last.exists():                 # Colab disconnected? continue where it stopped
        print("Resuming from", last)
        YOLO(str(last)).train(resume=True)
        return
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name=args.name, exist_ok=True, patience=args.patience, workers=args.workers,
        seed=args.seed, cos_lr=True, close_mosaic=15, amp=True,
        # --- augmentation chosen for SAR: small bright targets, no colour, orientation/scale arbitrary ---
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.25,        # no hue/saturation (single channel); mild brightness only
        degrees=0.0, translate=0.1, scale=0.4, shear=0.0,
        flipud=0.5, fliplr=0.5,                  # ships can point in any direction
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("Best weights:", best)


# ------------------------------------------------------------------------------------------
# 3. Evaluation (overall + offshore vs inshore using the SL-SSDD land masks)
# ------------------------------------------------------------------------------------------
def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def box_land_fraction(mask, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = mask.shape
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] == 0).mean())


def land_dilate(mask, px):
    """Land (0) grown by `px` pixels, so ships right at the shoreline count as inshore."""
    if px <= 0:
        return mask
    land = (mask == 0).astype(np.uint8)
    land = cv2.dilate(land, np.ones((2 * px + 1, 2 * px + 1), np.uint8))
    return np.where(land > 0, 0, 255).astype(np.uint8)


def evaluate(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
                  conf=0.001, iou=0.6, plots=True)
    print(f"\nOVERALL  P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")

    # offshore / inshore split at a fixed operating point
    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    img_dir, lab_dir, msk_dir = root / "images/val", root / "labels/val", root / "masks/val"
    if not msk_dir.exists() or not any(msk_dir.iterdir()):
        print("No land masks in the prepared dataset - skipping offshore/inshore breakdown.")
        return
    stats = {"offshore": [0, 0, 0], "inshore": [0, 0, 0]}          # TP, FP, FN
    for ip in sorted(img_dir.iterdir()):
        mp, lp = msk_dir / f"{ip.stem}.png", lab_dir / f"{ip.stem}.txt"
        if not mp.exists():
            continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        mask = land_dilate(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), args.shore_px)
        gts = []
        if lp.exists() and lp.read_text().strip():
            for ln in lp.read_text().strip().splitlines():
                _, cx, cy, bw, bh = (float(v) for v in ln.split())
                gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=0.6, device=args.device, verbose=False)[0]
        preds = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
        pc = [float(c) for c in res.boxes.conf.cpu().numpy()]
        order = np.argsort(pc)[::-1]
        used = set()
        gt_scene = ["inshore" if box_land_fraction(mask, g) > 0 else "offshore" for g in gts]
        for pi in order:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou_xyxy(preds[pi], g)
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj)
                stats[gt_scene[bj]][0] += 1
            else:
                sc = "inshore" if box_land_fraction(mask, preds[pi]) > 0 else "offshore"
                stats[sc][1] += 1
        for j, sc in enumerate(gt_scene):
            if j not in used:
                stats[sc][2] += 1
    print(f"\nPer-scene @ conf={args.conf}, IoU 0.5  (inshore = box touches land within {args.shore_px}px)")
    for sc, (tp, fp, fn) in stats.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"  {sc:<9} TP={tp:<4} FP={fp:<4} FN={fn:<4} P={p:.3f} R={r:.3f} F1={f1:.3f}")


# ------------------------------------------------------------------------------------------
# 4. Inference (+ optional land suppression with an SL-SSDD-style mask)
# ------------------------------------------------------------------------------------------
def predict(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    src = Path(args.source)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mask_idx = index_files(args.slssdd_masks, {".png", ".bmp", ".tif", ".jpg"}) if args.slssdd_masks else {}
    for f in files:
        img = cv2.imread(str(f))
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=0.6, device=args.device, verbose=False)[0]
        mask = cv2.imread(str(mask_idx[f.stem]), cv2.IMREAD_GRAYSCALE) if f.stem in mask_idx else None
        kept = 0
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            if mask is not None and box_land_fraction(mask, b) > args.max_land_frac:
                continue                                   # mostly on land -> drop (false alarm)
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{c:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            kept += 1
        cv2.imwrite(str(out / f.name), img)
        print(f"{f.name}: {kept} ships")


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="convert SSDD (VOC) + SL-SSDD masks into a YOLO dataset")
    p.add_argument("--ssdd", required=True, help="folder with SSDD images and VOC xml (searched recursively)")
    p.add_argument("--slssdd", default=None, help="folder with SL-SSDD sea/land mask PNGs")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["official", "random"], default="official")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=prepare)

    p = sub.add_parser("prepare-auto", help="any SAR ship dataset (YOLO txt / VOC xml / COCO json), e.g. Kaggle SARscope")
    p.add_argument("--src", required=True, help="dataset root folder (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=prepare_auto)

    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m/l.pt, yolo11s.pt, ...")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="0, 0,1, cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--project", default="runs/ssdd")
    p.add_argument("--name", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    p.set_defaults(fn=train)

    p = sub.add_parser("eval")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--shore-px", type=int, default=3)
    p.set_defaults(fn=evaluate)

    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="pred_out")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--slssdd-masks", default=None, help="folder of sea/land masks to drop on-land detections")
    p.add_argument("--max-land-frac", type=float, default=0.5)
    p.set_defaults(fn=predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
