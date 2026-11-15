"""
SAR Ship Detector - Tkinter desktop app (SAR only, no AIS)

Pipeline: adaptive CFAR detection -> blob geometry (length / width / axis)
          -> wake direction + ship heading estimate -> table, overlay, CSV export.

Needs: numpy pandas opencv-python scipy matplotlib  (tkinter ships with Python)
Run :  python sar_detector_ui.py
"""
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ----------------------------------------------------------------------------------
# Core processing (no GUI code - can be imported and tested on its own)
# ----------------------------------------------------------------------------------
def load_image(path, amplitude=False):
    """Load PNG/JPG/TIF as float32 intensity. Tick 'amplitude' if pixel values are amplitude (squared)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if amplitude:
        img = img ** 2
    pos = img[img > 0]
    floor = 1e-2 * float(np.median(pos)) if pos.size else 1.0
    return np.maximum(img, floor)


def make_demo_scene(size=1000, px=15.0, n_ships=8, seed=None):
    """Synthetic speckled sea with ships and wakes, for trying the app without real data."""
    rng = np.random.default_rng(seed)
    H = W = size
    sea = 0.2 * (1 + 0.4 * ndimage.gaussian_filter(rng.standard_normal((H, W)), 60) * 25)
    sea = np.clip(sea, 0.05, None).astype(np.float32)
    speckle = rng.gamma(4.0, 1 / 4.0, (H, W)).astype(np.float32)
    layer = np.zeros((H, W), np.float32)
    for _ in range(n_ships):
        c, r = rng.uniform(0.12 * W, 0.88 * W), rng.uniform(0.12 * H, 0.88 * H)
        hd = rng.uniform(0, 2 * math.pi)
        fwd = np.array([math.sin(hd), -math.cos(hd)])
        L = rng.uniform(80, 260) / px
        Wd = L / 6.5
        moving = rng.random() < 0.8
        if moving:
            n_seg, wake_len = 30, 14 * L + 40
            for k in reversed(range(n_seg)):
                f0, f1 = k / n_seg, (k + 1) / n_seg
                p0 = np.array([c, r]) - fwd * (0.5 * L + wake_len * f0)
                p1 = np.array([c, r]) - fwd * (0.5 * L + wake_len * f1)
                cv2.line(layer, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                         float(3.5 * (1 - f0) ** 1.2), 1)
        ang = math.degrees(math.atan2(fwd[1], fwd[0]))
        cv2.ellipse(layer, (int(c), int(r)), (max(2, int(L / 2)), max(1, int(Wd / 2))), ang, 0, 360, 25.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    return (sea * speckle * (1 + layer)).astype(np.float32)


def cfar_detect(img, px, T=6.0, win=61, guard=21, min_area=6, max_length_m=600.0, bounds=None):
    """Adaptive CFAR: pixel > T x local sea mean (guard ring excluded). Returns (DataFrame, background map)."""
    win += (win % 2 == 0)
    guard += (guard % 2 == 0)
    guard = min(guard, win - 2)
    s_big = cv2.boxFilter(img, -1, (win, win), normalize=False, borderType=cv2.BORDER_REFLECT)
    s_gd = cv2.boxFilter(img, -1, (guard, guard), normalize=False, borderType=cv2.BORDER_REFLECT)
    bg = np.maximum((s_big - s_gd) / (win * win - guard * guard), 1e-6)
    mask = (img > T * bg).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = img.shape
    rows = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels[y0:y0 + h, x0:x0 + w] == k)
        ys, xs = ys + y0, xs + x0
        wts = img[ys, xs]
        col, row = float(np.average(xs, weights=wts)), float(np.average(ys, weights=wts))
        cov = np.cov(np.vstack([xs * px, -ys * px])) if area > 2 else np.eye(2) * px ** 2
        lam, vec = np.linalg.eigh(cov)
        length = math.sqrt(12 * max(lam[1], 0) + px ** 2)
        width = math.sqrt(12 * max(lam[0], 0) + px ** 2)
        if length > max_length_m:                      # too large: land / coast / rain cell
            continue
        axis = math.degrees(math.atan2(vec[0, 1], vec[1, 1])) % 180
        peak = float(img[ys, xs].max() / bg[min(int(row), H - 1), min(int(col), W - 1)])
        rec = dict(col=col, row=row, area_px=int(area), length_m=length, width_m=width,
                   axis_deg=axis, snr_db=10 * math.log10(max(peak, 1e-3)))
        if bounds:
            n_, s_, w_, e_ = bounds
            rec["lat"] = n_ - (row + 0.5) / H * (n_ - s_)
            rec["lon"] = w_ + (col + 0.5) / W * (e_ - w_)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("snr_db", ascending=False).reset_index(drop=True)
        df.insert(0, "id", np.arange(1, len(df) + 1))
    return df, bg


def estimate_wake(img, bg, col, row, length_m, px, max_range_px=150):
    """Find the half-line leaving the ship with the highest mean brightness (= wake bearing).
    Ship heading = wake bearing + 180. Returns None if there is not enough image around the ship."""
    H, W = img.shape
    norm = ndimage.gaussian_filter(img / bg, 1.2)
    r0 = 0.5 * length_m / px + 6
    ts = np.arange(r0, max_range_px, 1.0)
    if len(ts) < 15:
        return None
    scores = np.full(360, np.nan)
    for i in range(360):
        a = math.radians(i)
        cols, rows = col + ts * math.sin(a), row - ts * math.cos(a)
        ok = (cols >= 0) & (cols < W - 1) & (rows >= 0) & (rows < H - 1)
        if ok.sum() < 0.6 * len(ts):
            continue
        v = ndimage.map_coordinates(norm, [rows[ok], cols[ok]], order=1)
        scores[i] = np.minimum(v, 6.0).mean() - 1.0
    if np.isfinite(scores).sum() < 90:
        return None
    s = np.where(np.isfinite(scores), scores, np.nanmedian(scores))
    best = int(np.argmax(s))
    idx = (best + np.arange(-3, 4)) % 360
    w = np.clip(s[idx], 0, None) + 1e-9
    ang = np.radians(idx)
    bearing = math.degrees(math.atan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) % 360
    conf = float((s[best] - np.median(s)) / (s.std() + 1e-9))
    return dict(wake_bearing=bearing, heading=(bearing + 180) % 360, wake_conf=conf)


def to_uint8(img, amplitude):
    """Float intensity -> 8-bit 3-channel BGR image like the SSDD training images."""
    amp = np.sqrt(img) if amplitude else img
    if amp.max() > 255.5:
        amp = amp * (255.0 / np.percentile(amp, 99.9))
    g = np.clip(amp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def yolo_detect(model, img, px, amplitude, conf=0.25, imgsz=640, tile=640, overlap=0.2, bounds=None):
    """Run a trained YOLO model. Large scenes are processed in overlapping tiles, then merged with NMS."""
    bgr = to_uint8(img, amplitude)
    H, W = bgr.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    boxes, scores = [], []
    for y0 in ys:
        for x0 in xs:
            crop = bgr[y0:y0 + tile, x0:x0 + tile]
            r = model.predict(crop, imgsz=imgsz, conf=conf, iou=0.6, verbose=False)[0]
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes.append([float(b[0] + x0), float(b[1] + y0), float(b[2] - b[0]), float(b[3] - b[1])])
                scores.append(float(c))
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.5) if boxes else []
    keep = np.array(keep).flatten().astype(int) if len(keep) else []
    win, guard = 61, 21
    s_big = cv2.boxFilter(img, -1, (win, win), normalize=False, borderType=cv2.BORDER_REFLECT)
    s_gd = cv2.boxFilter(img, -1, (guard, guard), normalize=False, borderType=cv2.BORDER_REFLECT)
    bg = np.maximum((s_big - s_gd) / (win * win - guard * guard), 1e-6)
    rows = []
    for k in keep:
        x, y, w, h = boxes[k]
        col, row = x + w / 2, y + h / 2
        x1, y1, x2, y2 = int(max(x, 0)), int(max(y, 0)), int(min(x + w, W)), int(min(y + h, H))
        patch = img[y1:y2, x1:x2]
        peak = float(patch.max() / bg[min(int(row), H - 1), min(int(col), W - 1)]) if patch.size else 1.0
        rec = dict(col=col, row=row, area_px=int(w * h), length_m=max(w, h) * px, width_m=min(w, h) * px,
                   axis_deg=np.nan, snr_db=10 * math.log10(max(peak, 1e-3)), conf=scores[k])
        if bounds:
            n_, s_, w_, e_ = bounds
            rec["lat"] = n_ - (row + 0.5) / H * (n_ - s_)
            rec["lon"] = w_ + (col + 0.5) / W * (e_ - w_)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("conf", ascending=False).reset_index(drop=True)
        df.insert(0, "id", np.arange(1, len(df) + 1))
    return df, bg


def add_wakes(df, img, bg, px, wake_conf_min=4.0, max_wakes=150):
    df["wake_bearing"], df["heading_deg"], df["wake_conf"] = np.nan, np.nan, np.nan
    for i, r in df.head(max_wakes).iterrows():
        w = estimate_wake(img, bg, r.col, r.row, r.length_m, px)
        if w:
            df.loc[i, "wake_conf"] = w["wake_conf"]
            if w["wake_conf"] >= wake_conf_min:
                df.loc[i, "wake_bearing"], df.loc[i, "heading_deg"] = w["wake_bearing"], w["heading"]
    return df


def run_yolo(model, img, px, amplitude, conf, imgsz, do_wake, bounds):
    df, bg = yolo_detect(model, img, px, amplitude, conf, imgsz, bounds=bounds)
    if len(df):
        df = add_wakes(df, img, bg, px) if do_wake else df.assign(
            wake_bearing=np.nan, heading_deg=np.nan, wake_conf=np.nan)
    return df, bg


def run_detection(img, px, T, win, guard, min_area, max_len, do_wake, bounds, wake_conf_min=4.0, max_wakes=150):
    df, bg = cfar_detect(img, px, T, win, guard, min_area, max_len, bounds)
    if len(df):
        df["wake_bearing"] = np.nan
        df["heading_deg"] = np.nan
        df["wake_conf"] = np.nan
        if do_wake:
            for i, r in df.head(max_wakes).iterrows():
                w = estimate_wake(img, bg, r.col, r.row, r.length_m, px)
                if w:
                    df.loc[i, "wake_conf"] = w["wake_conf"]
                    if w["wake_conf"] >= wake_conf_min:
                        df.loc[i, "wake_bearing"] = w["wake_bearing"]
                        df.loc[i, "heading_deg"] = w["heading"]
    return df, bg


# ----------------------------------------------------------------------------------
# Tkinter application
# ----------------------------------------------------------------------------------
class App:
    COLUMNS = [("id", 40), ("x", 60), ("y", 60), ("lat", 80), ("lon", 80), ("length_m", 75),
               ("width_m", 70), ("snr_db", 65), ("heading", 75), ("wake_conf", 75), ("conf", 55)]

    def __init__(self, root):
        self.root = root
        root.title("SAR Ship Detector")
        root.geometry("1280x820")
        self.img = None            # float32 intensity
        self.disp = None           # dB display array
        self.df = pd.DataFrame()
        self.bg = None
        self.path = None
        self.q = queue.Queue()
        self.sel_artist = None
        self.model = None
        self.model_path = None

        self._build_controls()
        self._build_view()
        self.status("Open a SAR image or load the demo scene.")

    # ---------------- layout ----------------
    def _build_controls(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        f = ttk.LabelFrame(left, text="Image", padding=6)
        f.pack(fill=tk.X, pady=3)
        ttk.Button(f, text="Open SAR image…", command=self.open_image).pack(fill=tk.X)
        ttk.Button(f, text="Load demo scene", command=self.load_demo).pack(fill=tk.X, pady=2)
        self.amp = tk.BooleanVar(value=True)    # 8-bit PNG/JPG are normally amplitude; untick for float/intensity TIFFs
        ttk.Checkbutton(f, text="Pixels are amplitude (square them)", variable=self.amp).pack(anchor="w")
        self.px = self._entry(f, "Pixel size (m)", 15.0)

        f = ttk.LabelFrame(left, text="Detector", padding=6)
        f.pack(fill=tk.X, pady=3)
        self.method = tk.StringVar(value="cfar")
        ttk.Radiobutton(f, text="Classic CFAR (no training)", variable=self.method, value="cfar").pack(anchor="w")
        ttk.Radiobutton(f, text="Trained YOLO model (.pt)", variable=self.method, value="yolo").pack(anchor="w")
        ttk.Button(f, text="Load YOLO weights…", command=self.load_model).pack(fill=tk.X, pady=2)
        self.model_lbl = tk.StringVar(value="no model loaded")
        ttk.Label(f, textvariable=self.model_lbl, foreground="#666", wraplength=230).pack(anchor="w")
        self.yconf = self._entry(f, "YOLO confidence", 0.25)
        self.yimgsz = self._entry(f, "YOLO image size", 640)

        f = ttk.LabelFrame(left, text="Detector (CFAR)", padding=6)
        f.pack(fill=tk.X, pady=3)
        self.T = self._entry(f, "Threshold T (× sea mean)", 6.0)
        self.win = self._entry(f, "Background window (px)", 61)
        self.guard = self._entry(f, "Guard window (px)", 21)
        self.min_area = self._entry(f, "Min area (px)", 6)
        self.max_len = self._entry(f, "Max length (m)", 600)
        self.do_wake = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Estimate wake / heading", variable=self.do_wake).pack(anchor="w")

        f = ttk.LabelFrame(left, text="Geo bounds (optional)", padding=6)
        f.pack(fill=tk.X, pady=3)
        self.north = self._entry(f, "North lat", "")
        self.south = self._entry(f, "South lat", "")
        self.west = self._entry(f, "West lon", "")
        self.east = self._entry(f, "East lon", "")

        self.btn_detect = ttk.Button(left, text="▶  Detect ships", command=self.detect)
        self.btn_detect.pack(fill=tk.X, pady=(10, 3), ipady=4)
        ttk.Button(left, text="Save results (CSV)…", command=self.save_csv).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Save annotated image…", command=self.save_png).pack(fill=tk.X, pady=2)

        self.show_wake = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show wake arrows", variable=self.show_wake,
                        command=self.redraw).pack(anchor="w", pady=(8, 0))
        self.show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show ID labels", variable=self.show_labels,
                        command=self.redraw).pack(anchor="w")

        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, wraplength=230, foreground="#444").pack(
            side=tk.BOTTOM, anchor="w", pady=6)

    def _entry(self, parent, label, default):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        ttk.Entry(row, textvariable=var, width=8).pack(side=tk.RIGHT)
        return var

    def _build_view(self):
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update()
        tb.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tf = ttk.Frame(right)
        tf.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree = ttk.Treeview(tf, columns=[c for c, _ in self.COLUMNS], show="headings", height=8)
        for c, w in self.COLUMNS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    # ---------------- helpers ----------------
    def status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _num(self, var, name, cast=float):
        try:
            return cast(float(var.get()))
        except ValueError:
            raise ValueError(f"'{name}' must be a number")

    def _bounds(self):
        vals = [v.get().strip() for v in (self.north, self.south, self.west, self.east)]
        if not any(vals):
            return None
        if not all(vals):
            raise ValueError("Fill all four geo bounds (or leave all empty)")
        return tuple(float(v) for v in vals)

    # ---------------- image loading ----------------
    def _set_image(self, img, label):
        self.img = img
        db = 10 * np.log10(img / max(float(np.median(img)), 1e-9))
        self.vmin, self.vmax = np.percentile(db, [1, 99.8])
        self.disp = db
        self.df, self.bg = pd.DataFrame(), None
        self._fill_table()
        self.redraw(reset_view=True)
        self.status(f"{label}  ({img.shape[1]}×{img.shape[0]} px)")

    def open_image(self):
        p = filedialog.askopenfilename(title="Open SAR image",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
        if not p:
            return
        try:
            self.path = p
            self._set_image(load_image(p, self.amp.get()), p.split("/")[-1].split("\\")[-1])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_demo(self):
        self.px.set("15.0")
        self.path = None
        self._set_image(make_demo_scene(seed=int(np.random.default_rng().integers(1e6))), "Demo scene")

    def load_model(self):
        p = filedialog.askopenfilename(title="Load trained YOLO weights",
                                       filetypes=[("PyTorch weights", "*.pt"), ("All", "*.*")])
        if not p:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            messagebox.showerror("Missing package", "Install it first:  pip install ultralytics")
            return
        self.status("Loading model…")
        try:
            self.model = YOLO(p)
            self.model_path = p
            self.model_lbl.set(p.replace("\\", "/").split("/")[-1])
            self.method.set("yolo")
            self.status("Model loaded. Press Detect ships.")
        except Exception as e:
            messagebox.showerror("Could not load model", str(e))

    # ---------------- detection (background thread) ----------------
    def detect(self):
        if self.img is None:
            messagebox.showinfo("No image", "Open an image or load the demo scene first.")
            return
        try:
            args = (self.img, self._num(self.px, "Pixel size"), self._num(self.T, "T"),
                    self._num(self.win, "Window", int), self._num(self.guard, "Guard", int),
                    self._num(self.min_area, "Min area", int), self._num(self.max_len, "Max length"),
                    self.do_wake.get(), self._bounds())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        if self.method.get() == "yolo":
            if self.model is None:
                messagebox.showinfo("No model", "Click 'Load YOLO weights…' first (the best.pt from Colab).")
                return
            try:
                yargs = (self.model, self.img, self._num(self.px, "Pixel size"), self.amp.get(),
                         self._num(self.yconf, "YOLO confidence"), self._num(self.yimgsz, "YOLO image size", int),
                         self.do_wake.get(), self._bounds())
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e))
                return
            fn, args = run_yolo, yargs
        else:
            fn = run_detection
        self.btn_detect.state(["disabled"])
        self.status("Detecting…")
        self.root.config(cursor="watch")

        def worker():
            try:
                self.q.put(("ok", fn(*args)))
            except Exception as e:
                self.q.put(("err", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        self.btn_detect.state(["!disabled"])
        self.root.config(cursor="")
        if kind == "err":
            messagebox.showerror("Detection failed", payload)
            self.status("Detection failed.")
            return
        self.df, self.bg = payload
        self._fill_table()
        self.redraw()
        n_h = int(self.df["heading_deg"].notna().sum()) if len(self.df) else 0
        self.status(f"{len(self.df)} ship candidates, {n_h} with wake heading.")

    # ---------------- table / drawing ----------------
    def _fill_table(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.df.itertuples():
            lat = f"{r.lat:.5f}" if hasattr(r, "lat") else "-"
            lon = f"{r.lon:.5f}" if hasattr(r, "lon") else "-"
            hd = "-" if pd.isna(r.heading_deg) else f"{r.heading_deg:.0f}°"
            wc = "-" if pd.isna(r.wake_conf) else f"{r.wake_conf:.1f}"
            cf = f"{r.conf:.2f}" if hasattr(r, "conf") and not pd.isna(r.conf) else "-"
            self.tree.insert("", tk.END, iid=str(r.id), values=(
                r.id, f"{r.col:.0f}", f"{r.row:.0f}", lat, lon, f"{r.length_m:.0f}",
                f"{r.width_m:.0f}", f"{r.snr_db:.1f}", hd, wc, cf))

    def redraw(self, reset_view=False):
        if self.disp is None:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.disp, cmap="gray", vmin=self.vmin, vmax=self.vmax, interpolation="nearest")
        self.ax.axis("off")
        self.sel_artist = None
        rad = max(12, min(self.disp.shape) * 0.015)
        for r in self.df.itertuples():
            self.ax.add_patch(Circle((r.col, r.row), rad, fill=False, ec="#ff3b30", lw=1.6))
            if self.show_labels.get():
                self.ax.text(r.col + rad * 1.1, r.row - rad * 1.1, str(r.id), color="#ff3b30", fontsize=9)
            if self.show_wake.get() and not pd.isna(r.wake_bearing):
                b = math.radians(r.wake_bearing)
                self.ax.annotate("", xy=(r.col + 5 * rad * math.sin(b), r.row - 5 * rad * math.cos(b)),
                                 xytext=(r.col, r.row),
                                 arrowprops=dict(arrowstyle="->", color="yellow", lw=1.4))
        if not reset_view:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def on_select(self, _):
        sel = self.tree.selection()
        if not sel or self.df.empty:
            return
        r = self.df[self.df["id"] == int(sel[0])].iloc[0]
        if self.sel_artist is not None:
            self.sel_artist.remove()
        rad = max(12, min(self.disp.shape) * 0.015) * 2
        self.sel_artist = Circle((r.col, r.row), rad, fill=False, ec="#00e5ff", lw=2.5)
        self.ax.add_patch(self.sel_artist)
        span = 12 * rad                                   # zoom to the selected target
        h, w = self.disp.shape
        self.ax.set_xlim(max(0, r.col - span), min(w, r.col + span))
        self.ax.set_ylim(min(h, r.row + span), max(0, r.row - span))
        self.canvas.draw_idle()

    # ---------------- export ----------------
    def save_csv(self):
        if self.df.empty:
            messagebox.showinfo("Nothing to save", "Run detection first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.df.to_csv(p, index=False)
            self.status(f"Saved {len(self.df)} rows to {p}")

    def save_png(self):
        if self.disp is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if p:
            self.fig.savefig(p, dpi=200, bbox_inches="tight")
            self.status(f"Saved {p}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
