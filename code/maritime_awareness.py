# %% [markdown]
# # AI-Powered Multi-Sensor Maritime Awareness System (v2)
# SAR + live AIS prototype with:
# 1. Georeferenced SAR images
# 2. Time-aware AIS matching
# 3. Kalman-filter tracking
# 4. Wake direction estimation
# 5. Anomaly scoring
#
# Runs in Google Colab or as a plain Python script (Thonny / VS Code).
# MODE = "simulate" needs no API key and no SAR file.

# %% Cell 1 - install (Colab: uncomment)
# !pip install websocket-client pandas numpy folium opencv-python-headless matplotlib scipy

# %% Cell 2 - configuration
import sys, json, time, math, threading, webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

IN_COLAB = "google.colab" in sys.modules

MODE = "simulate"          # "simulate" or "live"
                           #   simulate: synthetic SAR scene + synthetic AIS (fully offline)
                           #   live    : real AISStream AIS + synthetic SAR scene planted around it
SAR_FILE = None            # path to a north-up EPSG:4326 GeoTIFF (needs rasterio), or None

AIS_API_KEY = ""           # keep private! In Colab prefer: userdata.get("AIS_API_KEY")
if IN_COLAB and not AIS_API_KEY:
    try:
        from google.colab import userdata
        AIS_API_KEY = userdata.get("AIS_API_KEY")
    except Exception:
        pass

# SAR footprint (north-up). Kerala coast example, ~16.5 km x 16.5 km
FOOTPRINT = dict(north=8.60, south=8.45, west=76.75, east=76.90)
IMG_SIZE = 1000                                   # pixels (square), ~16.5 m / pixel
# AIS subscription box, [[lat1, lon1], [lat2, lon2]] - a bit larger than the footprint
BOUNDING_BOX = [[8.7, 76.65], [8.35, 77.0]]
LIVE_SECONDS = 90

# matching / tracking parameters
MAX_AIS_AGE_S = 1800        # ignore AIS tracks whose nearest report is older than this
GATE_CHI2 = 9.21            # 99 % gate, 2 degrees of freedom
SAR_GEOLOC_SIGMA_M = 30.0   # SAR geolocation error (1-sigma)
AZIMUTH_SHIFT_FACTOR = 0.0  # extra sigma per m/s of ship speed (SAR Doppler shift). Try ~ 5-15 for real data
KNOT = 0.514444             # m/s per knot

rng = np.random.default_rng(42)


# %% Cell 3 - 1) georeferencing
@dataclass
class GeoRef:
    """North-up georeferencing of a SAR image + acquisition time (UTC)."""
    north: float
    south: float
    west: float
    east: float
    width: int
    height: int
    time_utc: datetime

    def pixel_to_latlon(self, col, row):
        lon = self.west + (np.asarray(col) + 0.5) / self.width * (self.east - self.west)
        lat = self.north - (np.asarray(row) + 0.5) / self.height * (self.north - self.south)
        return lat, lon

    def latlon_to_pixel(self, lat, lon):
        col = (np.asarray(lon) - self.west) / (self.east - self.west) * self.width - 0.5
        row = (self.north - np.asarray(lat)) / (self.north - self.south) * self.height - 0.5
        return col, row

    def contains(self, lat, lon, margin=0.0):
        return (self.south + margin <= lat <= self.north - margin and
                self.west + margin <= lon <= self.east - margin)

    @property
    def center(self):
        return (self.north + self.south) / 2, (self.west + self.east) / 2

    @property
    def pixel_size_m(self):
        lat0, _ = self.center
        mx = (self.east - self.west) * 111_320 * math.cos(math.radians(lat0)) / self.width
        my = (self.north - self.south) * 110_574 / self.height
        return mx, my

    @classmethod
    def from_geotiff(cls, path, time_utc=None):
        """Read a north-up EPSG:4326 GeoTIFF. Reproject first (gdalwarp) for other CRSs."""
        import rasterio
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise ValueError("GeoTIFF must be EPSG:4326 (use gdalwarp -t_srs EPSG:4326)")
            b = ds.bounds
            tags = ds.tags()
            t = time_utc
            if t is None:
                raw = tags.get("ACQUISITION_TIME") or tags.get("TIFFTAG_DATETIME")
                if raw is None:
                    raise ValueError("Give time_utc=... (acquisition time not found in tags)")
                t = pd.to_datetime(raw, utc=True).to_pydatetime()
            img = ds.read(1).astype(np.float32)
            return cls(b.top, b.bottom, b.left, b.right, ds.width, ds.height, t), img


class LocalENU:
    """Flat-earth east/north metres around a reference point (fine for a ~50 km scene)."""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = 111_320 * math.cos(math.radians(lat0))
        self.ky = 110_574

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_latlon(self, e, n):
        return self.lat0 + np.asarray(n) / self.ky, self.lon0 + np.asarray(e) / self.kx


def ang_diff(a, b):
    """Smallest absolute difference between two compass angles (deg)."""
    return abs((a - b + 180) % 360 - 180)


# %% Cell 4 - live AIS collection (AISStream) + helpers
def collect_ais_live(api_key, bbox, seconds=60):
    """Collect ALL position reports (history is kept - needed for Kalman tracking)."""
    import websocket
    rows = []

    def parse_time(meta):
        raw = meta.get("time_utc")
        try:
            return pd.to_datetime(str(raw)[:19], utc=True).to_pydatetime()
        except Exception:
            return datetime.now(timezone.utc)

    def on_message(ws, message):
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            d = json.loads(message)
            if d.get("MessageType") != "PositionReport":
                return
            meta = d.get("MetaData", {})
            pr = d.get("Message", {}).get("PositionReport", {})
            rows.append(dict(
                timestamp=parse_time(meta),
                mmsi=meta.get("MMSI"),
                ship_name=(meta.get("ShipName") or "").strip() or f"MMSI {meta.get('MMSI')}",
                latitude=meta.get("latitude", meta.get("Latitude", pr.get("Latitude"))),
                longitude=meta.get("longitude", meta.get("Longitude", pr.get("Longitude"))),
                speed_knots=pr.get("Sog"), course=pr.get("Cog"), heading=pr.get("TrueHeading")))
        except Exception as e:
            print("Message error:", e)

    def on_open(ws):
        ws.send(json.dumps({"APIKey": api_key, "BoundingBoxes": [bbox],
                            "FilterMessageTypes": ["PositionReport"]}))
        print("Connected to AISStream")

    ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream",
                                on_open=on_open, on_message=on_message,
                                on_error=lambda w, e: print("WS error:", e))
    threading.Thread(target=ws.run_forever, daemon=True).start()
    print(f"Collecting AIS for {seconds}s ...")
    time.sleep(seconds)
    ws.close()
    df = pd.DataFrame(rows)
    if len(df):
        df = df.dropna(subset=["latitude", "longitude"]).sort_values("timestamp").reset_index(drop=True)
    print("Position reports:", len(df), "| unique vessels:", df["mmsi"].nunique() if len(df) else 0)
    return df


# %% Cell 5 - synthetic SAR scene + synthetic AIS (so the whole pipeline is testable offline)
def render_sar(geo, ships, rng):
    """Simulate a SAR intensity image: speckled sea + bright ships + turbulent & Kelvin wakes."""
    H, W = geo.height, geo.width
    mx, my = geo.pixel_size_m
    px = (mx + my) / 2
    sea = 0.2 * (1 + 0.4 * ndimage.gaussian_filter(rng.standard_normal((H, W)), 60) * 25)
    sea = np.clip(sea, 0.05, None).astype(np.float32)
    speckle = rng.gamma(4.0, 1 / 4.0, (H, W)).astype(np.float32)          # 4-look speckle
    layer = np.zeros((H, W), np.float32)

    for s in ships:
        c, r = geo.latlon_to_pixel(s["lat"], s["lon"])
        hd = math.radians(s["heading"])
        fwd = np.array([math.sin(hd), -math.cos(hd)])                      # (dcol, drow)
        Lp, Wp = s["length_m"] / px, s["width_m"] / px
        if s["speed_ms"] > 0.5:                                            # wakes only for moving ships
            n_seg, wake_len = 30, 14 * Lp + 40
            for k in reversed(range(n_seg)):
                f0, f1 = k / n_seg, (k + 1) / n_seg
                p0 = np.array([c, r]) - fwd * (0.5 * Lp + wake_len * f0)
                p1 = np.array([c, r]) - fwd * (0.5 * Lp + wake_len * f1)
                cv2.line(layer, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                         float(3.5 * (1 - f0) ** 1.2), 1)
            for sgn in (-1, 1):                                            # faint Kelvin arms (19.5 deg)
                a = math.radians(sgn * 19.5)
                d = np.array([-(fwd[0] * math.cos(a) - fwd[1] * math.sin(a)),
                              -(fwd[0] * math.sin(a) + fwd[1] * math.cos(a))])
                p0 = np.array([c, r]) + d * 0.3 * Lp
                p1 = np.array([c, r]) + d * (6 * Lp + 30)
                cv2.line(layer, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1.0, 1)
        ang = math.degrees(math.atan2(fwd[1], fwd[0]))
        cv2.ellipse(layer, (int(round(c)), int(round(r))),
                    (max(2, int(Lp / 2)), max(1, int(Wp / 2))), ang, 0, 360, 25.0, -1)

    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    return (sea * speckle * (1 + layer)).astype(np.float32)


def simulate_scene(geo, enu, rng):
    """Truth ships + AIS history with deliberate test cases:
    normal / spoofed-course / dark (no AIS) / AIS-only (no ship in image)."""
    roles = ["normal", "normal", "normal", "normal", "spoof_course", "dark", "ais_only"]
    ships, ais_rows = [], []
    t_sar = geo.time_utc.timestamp()
    e0, e1 = enu.to_xy(geo.center[0], geo.west)[0], enu.to_xy(geo.center[0], geo.east)[0]
    n0, n1 = enu.to_xy(geo.south, geo.center[1])[1], enu.to_xy(geo.north, geo.center[1])[1]
    for i, role in enumerate(roles):
        e = rng.uniform(e0 + 0.15 * (e1 - e0), e1 - 0.15 * (e1 - e0))
        n = rng.uniform(n0 + 0.15 * (n1 - n0), n1 - 0.15 * (n1 - n0))
        sog = rng.uniform(4, 14)
        cog = rng.uniform(0, 360)
        v = np.array([math.sin(math.radians(cog)), math.cos(math.radians(cog))]) * sog * KNOT
        lat, lon = enu.to_latlon(e, n)
        length = rng.uniform(90, 250)
        ship = dict(role=role, mmsi=235000000 + i, name=f"SIM VESSEL {i}", lat=float(lat), lon=float(lon),
                    heading=cog, length_m=length, width_m=length / 6.5, speed_ms=sog * KNOT)
        if role != "ais_only":
            ships.append(ship)
        if role == "dark":
            continue
        age = rng.uniform(5, 90)                       # last AIS report is older than SAR time
        for t in np.arange(t_sar - 600, t_sar - age, 15.0):
            dt = t_sar - t
            pe, pn = e - v[0] * dt + rng.normal(0, 8), n - v[1] * dt + rng.normal(0, 8)
            plat, plon = enu.to_latlon(pe, pn)
            rep_cog = (cog + 130) % 360 if role == "spoof_course" else cog + rng.normal(0, 2)
            ais_rows.append(dict(timestamp=datetime.fromtimestamp(t, timezone.utc), mmsi=ship["mmsi"],
                                 ship_name=ship["name"], latitude=float(plat), longitude=float(plon),
                                 speed_knots=sog + rng.normal(0, 0.2), course=rep_cog % 360, heading=None))
    return ships, pd.DataFrame(ais_rows)


def scene_from_live(df, geo, enu, rng):
    """Plant synthetic SAR ships at the dead-reckoned positions of REAL live AIS vessels (+1 dark ship)."""
    t_sar = geo.time_utc.timestamp()
    ships = []
    last = df.sort_values("timestamp").groupby("mmsi").tail(1)
    for _, v in last.iterrows():
        sog = (v.speed_knots or 0) * KNOT if (v.speed_knots or 0) < 102 else 0
        cog = v.course if (v.course is not None and v.course < 360) else rng.uniform(0, 360)
        e, n = enu.to_xy(v.latitude, v.longitude)
        dt = t_sar - v.timestamp.timestamp()
        e += math.sin(math.radians(cog)) * sog * dt
        n += math.cos(math.radians(cog)) * sog * dt
        lat, lon = enu.to_latlon(e, n)
        if geo.contains(float(lat), float(lon), margin=0.01):
            L = rng.uniform(100, 220)
            ships.append(dict(role="live", mmsi=v.mmsi, name=v.ship_name, lat=float(lat), lon=float(lon),
                              heading=cog, length_m=L, width_m=L / 6.5, speed_ms=sog))
    if len(ships) == 0:
        return None
    lat = rng.uniform(geo.south + 0.02, geo.north - 0.02)
    lon = rng.uniform(geo.west + 0.02, geo.east - 0.02)
    ships.append(dict(role="dark", mmsi=0, name="DARK", lat=lat, lon=lon, heading=rng.uniform(0, 360),
                      length_m=150, width_m=23, speed_ms=6.0))
    return ships


# %% Cell 6 - SAR ship detector (adaptive CFAR) + georeferenced detections
def detect_ships(img, geo, win=61, guard=21, T=6.0, min_area=6):
    """Two-parameter-style CFAR on linear intensity: pixel > T x local sea mean (guard ring excluded).
    Returns detections with lat/lon, size, orientation, plus the local background map."""
    img = img.astype(np.float32)
    s_big = cv2.boxFilter(img, -1, (win, win), normalize=False, borderType=cv2.BORDER_REFLECT)
    s_gd = cv2.boxFilter(img, -1, (guard, guard), normalize=False, borderType=cv2.BORDER_REFLECT)
    bg = np.maximum((s_big - s_gd) / (win * win - guard * guard), 1e-6)
    mask = (img > T * bg).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mx, my = geo.pixel_size_m
    px = (mx + my) / 2
    rows = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels[y0:y0 + h, x0:x0 + w] == k)
        ys, xs = ys + y0, xs + x0
        wts = img[ys, xs]
        col, row = np.average(xs, weights=wts), np.average(ys, weights=wts)
        E, N = xs * mx, -ys * my                                  # east / north metres
        cov = np.cov(np.vstack([E, N]))
        lam, vec = np.linalg.eigh(cov)
        length = math.sqrt(12 * max(lam[1], 0) + px ** 2)
        width = math.sqrt(12 * max(lam[0], 0) + px ** 2)
        axis = math.degrees(math.atan2(vec[0, 1], vec[1, 1])) % 180   # compass axis, 0..180 (ambiguous)
        lat, lon = geo.pixel_to_latlon(col, row)
        rows.append(dict(det_id=len(rows), col=col, row=row, lat=float(lat), lon=float(lon),
                         area_px=int(area), length_m=length, width_m=width, axis_deg=axis,
                         peak_ratio=float(img[ys, xs].max() / bg[int(row), int(col)])))
    return pd.DataFrame(rows), bg


# %% Cell 7 - 4) wake direction estimation
def estimate_wake(img, bg, det, geo, max_range_px=150):
    """Search 360 deg for the half-line (ray) leaving the ship with the highest mean brightness.
    The ray direction = wake bearing; ship heading = wake bearing + 180 (this resolves the 180 deg ambiguity
    that hull orientation alone cannot). Returns None when the scene gives too little room."""
    H, W = img.shape
    norm = ndimage.gaussian_filter(img / bg, 1.2)
    px = np.mean(geo.pixel_size_m)
    r0 = 0.5 * det.length_m / px + 6
    ts = np.arange(r0, max_range_px, 1.0)
    if len(ts) < 15:
        return None
    step = 1
    scores = np.full(360 // step, np.nan)
    for i in range(360 // step):
        a = math.radians(i * step)
        cols, rows = det.col + ts * math.sin(a), det.row - ts * math.cos(a)
        ok = (cols >= 0) & (cols < W - 1) & (rows >= 0) & (rows < H - 1)
        if ok.sum() < 0.6 * len(ts):
            continue
        v = ndimage.map_coordinates(norm, [rows[ok], cols[ok]], order=1)
        scores[i] = np.minimum(v, 6.0).mean() - 1.0
    if np.isfinite(scores).sum() < 90:
        return None
    s = np.where(np.isfinite(scores), scores, np.nanmedian(scores))
    best = int(np.argmax(s))
    idx = (best + np.arange(-3, 4)) % len(s)                         # refine with +-3 deg weighted mean
    w = np.clip(s[idx], 0, None) + 1e-9
    ang = np.radians(idx * step)
    bearing = math.degrees(math.atan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) % 360
    conf = float((s[best] - np.median(s)) / (s.std() + 1e-9))
    return dict(wake_bearing=bearing, heading_est=(bearing + 180) % 360,
                wake_strength=float(s[best]), wake_conf=conf)


# %% Cell 8 - 3) Kalman filter tracker (constant velocity, irregular time steps)
class VesselKalman:
    """State x = [east, north, v_east, v_north] in metres, m/s.  Measurement = AIS position + SOG/COG velocity."""
    def __init__(self, sigma_a=0.03, r_pos=15.0, r_vel=1.0):
        self.q = sigma_a ** 2
        self.R = np.diag([r_pos ** 2, r_pos ** 2, r_vel ** 2, r_vel ** 2])
        self.x, self.P, self.t = None, None, None
        self.history = []                                            # (t, x, P) after every update

    def _FQ(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        a = abs(dt)
        Q = self.q * np.array([[a ** 3 / 3, 0, a ** 2 / 2, 0],
                               [0, a ** 3 / 3, 0, a ** 2 / 2],
                               [a ** 2 / 2, 0, a, 0],
                               [0, a ** 2 / 2, 0, a]])
        return F, Q

    def update(self, t, e, n, sog_ms, cog_deg):
        if sog_ms is None or cog_deg is None or sog_ms >= 102 * KNOT or cog_deg >= 360:
            ve = vn = 0.0                                            # AIS "not available" codes
            R = self.R.copy(); R[2, 2] = R[3, 3] = 5.0 ** 2
        else:
            ve, vn = sog_ms * math.sin(math.radians(cog_deg)), sog_ms * math.cos(math.radians(cog_deg))
            R = self.R
        z = np.array([e, n, ve, vn])
        if self.x is None:
            self.x, self.P, self.t = z, R.copy(), t
        else:
            F, Q = self._FQ(t - self.t)
            x, P = F @ self.x, F @ self.P @ F.T + Q
            S = P + R
            K = P @ np.linalg.inv(S)
            I = np.eye(4)
            self.x = x + K @ (z - x)
            self.P = (I - K) @ P @ (I - K).T + K @ R @ K.T           # Joseph form
            self.t = t
        self.history.append((t, self.x.copy(), self.P.copy()))

    def predict(self, t):
        """Predict state/covariance at time t (forward OR backward) without altering the filter."""
        F, Q = self._FQ(t - self.t)
        return F @ self.x, F @ self.P @ F.T + Q


@dataclass
class Track:
    mmsi: int
    name: str
    kf: VesselKalman
    last_t: float
    last_sog_kn: float
    last_cog: float
    n_reports: int


def build_tracks(df_ais, enu, t_sar, max_age=MAX_AIS_AGE_S):
    tracks = []
    for mmsi, g in df_ais.groupby("mmsi"):
        g = g.sort_values("timestamp")
        ts = np.array([x.timestamp() for x in g["timestamp"]])
        keep = np.abs(ts - t_sar) <= max_age
        if not keep.any():
            continue
        g, ts = g[keep], ts[keep]
        kf = VesselKalman()
        for (_, r), t in zip(g.iterrows(), ts):
            e, n = enu.to_xy(r.latitude, r.longitude)
            sog = None if pd.isna(r.speed_knots) else float(r.speed_knots) * KNOT
            cog = None if pd.isna(r.course) else float(r.course)
            kf.update(t, float(e), float(n), sog, cog)
        last = g.iloc[-1]
        tracks.append(Track(int(mmsi), last.ship_name, kf, ts[-1],
                            0.0 if pd.isna(last.speed_knots) else float(last.speed_knots),
                            np.nan if pd.isna(last.course) else float(last.course), len(g)))
    return tracks


# %% Cell 9 - 2) time-aware matching (Kalman prediction to SAR time + Mahalanobis gate + Hungarian)
def match_sar_ais(dets, tracks, geo, enu, gate=GATE_CHI2, sigma_geo=SAR_GEOLOC_SIGMA_M,
                  az_factor=AZIMUTH_SHIFT_FACTOR):
    t_sar = geo.time_utc.timestamp()
    nd, nt = len(dets), len(tracks)
    preds = [tr.kf.predict(t_sar) for tr in tracks]
    d2 = np.full((nd, nt), np.inf)
    for i, d in enumerate(dets.itertuples()):
        e, n = enu.to_xy(d.lat, d.lon)
        for j, (x, P) in enumerate(preds):
            speed = math.hypot(x[2], x[3])
            sig2 = (sigma_geo + az_factor * speed) ** 2
            S = P[:2, :2] + np.eye(2) * sig2
            r = np.array([e - x[0], n - x[1]])
            d2[i, j] = float(r @ np.linalg.solve(S, r))
    cost = np.where(d2 < gate, d2, 1e6)
    pairs = {}
    if nd and nt:
        for i, j in zip(*linear_sum_assignment(cost)):
            if d2[i, j] < gate:
                pairs[i] = j
    return pairs, d2, preds


# %% Cell 10 - 5) anomaly scoring
def score_anomalies(dets, tracks, pairs, d2, preds, wakes, geo, enu, gate=GATE_CHI2):
    t_sar = geo.time_utc.timestamp()
    rows = []
    for i, d in enumerate(dets.itertuples()):
        w = wakes.get(i)
        wake_ok = w is not None and w["wake_conf"] >= 4.0
        moving_wake = wake_ok and w["wake_strength"] > 0.5
        if i in pairs:
            tr = tracks[pairs[i]]
            x, _ = preds[pairs[i]]
            age = abs(t_sar - tr.last_t)
            resid = min(d2[i, pairs[i]] / gate, 1.0)
            head = 0.0
            if wake_ok and tr.last_sog_kn >= 2 and not np.isnan(tr.last_cog):
                head = min(ang_diff(w["heading_est"], tr.last_cog) / 90.0, 1.0)
            stale = min(age / MAX_AIS_AGE_S, 1.0)
            speed_inc = 1.0 if (tr.last_sog_kn < 0.5 and moving_wake) else 0.0
            score = 0.30 * resid + 0.40 * head + 0.15 * stale + 0.15 * speed_inc
            reasons = []
            if head > 0.5: reasons.append(f"wake heading {w['heading_est']:.0f}° vs AIS COG {tr.last_cog:.0f}°")
            if resid > 0.6: reasons.append("large position residual")
            if speed_inc: reasons.append("AIS says stopped but wake present")
            if stale > 0.5: reasons.append(f"AIS {age/60:.0f} min old")
            rows.append(dict(kind="matched", det_id=d.det_id, mmsi=tr.mmsi, name=tr.name, lat=d.lat, lon=d.lon,
                             score=score, reason="; ".join(reasons) or "consistent"))
        else:
            score = 0.60 + 0.25 * min(d.length_m / 200.0, 1.0) + 0.15 * float(moving_wake)
            rows.append(dict(kind="dark_vessel", det_id=d.det_id, mmsi=None, name="UNMATCHED SAR TARGET",
                             lat=d.lat, lon=d.lon, score=min(score, 1.0),
                             reason=f"no AIS within gate; length~{d.length_m:.0f} m"
                                    + ("; moving (wake)" if moving_wake else "")))
    matched_tracks = set(pairs.values())
    for j, tr in enumerate(tracks):                                  # AIS in footprint but nothing seen
        x, _ = preds[j]
        lat, lon = enu.to_latlon(x[0], x[1])
        if j not in matched_tracks and geo.contains(float(lat), float(lon), margin=0.005):
            rows.append(dict(kind="ais_only", det_id=None, mmsi=tr.mmsi, name=tr.name, lat=float(lat),
                             lon=float(lon), score=0.45,
                             reason="AIS position inside scene but no SAR target (missed / spoofed / small)"))
    out = pd.DataFrame(rows)
    if len(out):
        out["level"] = pd.cut(out["score"], [-1, 0.3, 0.6, 2], labels=["normal", "watch", "ALERT"])
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


# %% Cell 11 - visualisation (SAR plot + interactive map)
LEVEL_COLOR = {"normal": "lime", "watch": "orange", "ALERT": "red"}


def plot_sar(img, dets, wakes, alerts, bg):
    db = 10 * np.log10(np.clip(img / np.median(bg), 1e-3, None))
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(db, cmap="gray", vmin=-8, vmax=18)
    lvl = {r.det_id: r.level for r in alerts.itertuples() if r.det_id is not None and not pd.isna(r.det_id)}
    for d in dets.itertuples():
        c = LEVEL_COLOR.get(str(lvl.get(d.det_id, "normal")), "cyan")
        ax.add_patch(plt.Circle((d.col, d.row), 18, fill=False, ec=c, lw=2))
        ax.text(d.col + 22, d.row - 22, f"#{d.det_id}", color=c, fontsize=10)
        w = wakes.get(d.det_id)
        if w and w["wake_conf"] >= 4:
            b = math.radians(w["wake_bearing"])
            ax.annotate("", xy=(d.col + 90 * math.sin(b), d.row - 90 * math.cos(b)), xytext=(d.col, d.row),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
    ax.set_title("SAR detections (circle = anomaly level, yellow arrow = estimated wake direction)")
    ax.axis("off")
    plt.show()


def make_map(geo, enu, df_ais, tracks, preds, dets, pairs, alerts, path="maritime_map.html"):
    try:
        import folium
    except ImportError:
        print("folium not installed - run: pip install folium  (skipping interactive map)")
        return None
    m = folium.Map(location=list(geo.center), zoom_start=11, tiles="CartoDB positron")
    folium.Rectangle([[geo.south, geo.west], [geo.north, geo.east]], color="blue", fill=False,
                     tooltip="SAR footprint").add_to(m)
    fg_ais = folium.FeatureGroup("AIS tracks (Kalman)").add_to(m)
    fg_det = folium.FeatureGroup("SAR detections").add_to(m)
    fg_link = folium.FeatureGroup("matches").add_to(m)
    for j, tr in enumerate(tracks):
        pts = [tuple(enu.to_latlon(x[0], x[1])) for _, x, _ in tr.kf.history]
        folium.PolyLine([(float(a), float(b)) for a, b in pts], color="gray", weight=2).add_to(fg_ais)
        x, P = preds[j]
        plat, plon = enu.to_latlon(x[0], x[1])
        rad = 2.45 * math.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 1.0)) + SAR_GEOLOC_SIGMA_M
        folium.Circle([float(plat), float(plon)], radius=rad, color="#1f77b4", fill=True, fill_opacity=0.15,
                      tooltip=f"{tr.name} predicted @ SAR time (gate ~{rad:.0f} m)").add_to(fg_ais)
        folium.CircleMarker([float(plat), float(plon)], radius=4, color="#1f77b4", fill=True,
                            popup=f"{tr.name}<br>MMSI {tr.mmsi}<br>{tr.last_sog_kn:.1f} kn / COG {tr.last_cog:.0f}°"
                            ).add_to(fg_ais)
    for r in alerts.itertuples():
        col = LEVEL_COLOR.get(str(r.level), "blue")
        icon = {"matched": "ok-sign", "dark_vessel": "warning-sign", "ais_only": "question-sign"}[r.kind]
        folium.Marker([r.lat, r.lon], icon=folium.Icon(color={"lime": "green"}.get(col, col), icon=icon),
                      tooltip=f"{r.kind} | score {r.score:.2f}",
                      popup=f"<b>{r.name}</b><br>score {r.score:.2f} ({r.level})<br>{r.reason}").add_to(fg_det)
    for i, j in pairs.items():
        x, _ = preds[j]
        plat, plon = enu.to_latlon(x[0], x[1])
        folium.PolyLine([(dets.lat[i], dets.lon[i]), (float(plat), float(plon))], color="green",
                        weight=2, dash_array="4").add_to(fg_link)
    folium.LayerControl().add_to(m)
    m.save(path)
    print("Map saved to", path)
    if not IN_COLAB:
        try: webbrowser.open("file://" + __import__("os").path.abspath(path))
        except Exception: pass
    return m


# %% Cell 12 - run the full pipeline
def run():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    df_ais, ships, img, geo = None, None, None, None

    if SAR_FILE:                                                     # real georeferenced SAR image
        geo, img = GeoRef.from_geotiff(SAR_FILE)
        print("Loaded GeoTIFF:", geo)
        enu = LocalENU(*geo.center)
    else:
        geo = GeoRef(**FOOTPRINT, width=IMG_SIZE, height=IMG_SIZE, time_utc=now)
        enu = LocalENU(*geo.center)

    if MODE == "live":
        df_ais = collect_ais_live(AIS_API_KEY, BOUNDING_BOX, LIVE_SECONDS)
        geo.time_utc = datetime.now(timezone.utc)                    # pretend SAR pass is "now"
        if SAR_FILE is None and len(df_ais):
            ships = scene_from_live(df_ais, geo, enu, rng)
            if ships is None:
                print("No live vessels inside the footprint - falling back to simulation")
    if df_ais is None or len(df_ais) == 0 or (SAR_FILE is None and ships is None):
        ships, df_ais = simulate_scene(geo, enu, rng)
    if img is None:
        img = render_sar(geo, ships, rng)

    t_sar = geo.time_utc.timestamp()
    print(f"SAR time {geo.time_utc:%Y-%m-%d %H:%M:%S} UTC | pixel {geo.pixel_size_m[0]:.1f} m | "
          f"AIS reports {len(df_ais)}")

    dets, bg = detect_ships(img, geo)
    print("SAR detections:", len(dets))
    wakes = {}
    for d in dets.itertuples():
        w = estimate_wake(img, bg, d, geo)
        if w: wakes[d.det_id] = w

    tracks = build_tracks(df_ais, enu, t_sar)
    pairs, d2, preds = match_sar_ais(dets, tracks, geo, enu)
    alerts = score_anomalies(dets, tracks, pairs, d2, preds, wakes, geo, enu)

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print("\n=== ANOMALY REPORT ===")
    print(alerts[["kind", "name", "score", "level", "reason"]].round(2).to_string())

    if ships is not None:                                            # simulation self-check
        print("\n=== GROUND TRUTH (simulation) ===")
        for s in ships:
            c, r = geo.latlon_to_pixel(s["lat"], s["lon"])
            hit = ((dets.col - c) ** 2 + (dets.row - r) ** 2).min() ** 0.5 if len(dets) else 1e9
            print(f"{s['name']:<14} role={s['role']:<13} heading={s['heading']:5.1f}  detected={'yes' if hit < 8 else 'NO'}")

    plot_sar(img, dets, wakes, alerts, bg)
    m = make_map(geo, enu, df_ais, tracks, preds, dets, pairs, alerts)
    return dict(geo=geo, img=img, dets=dets, wakes=wakes, tracks=tracks, pairs=pairs,
                alerts=alerts, df_ais=df_ais, map=m)


if __name__ == "__main__" or IN_COLAB:
    result = run()
    result["map"] if IN_COLAB else None      # last expression shows the map in a notebook cell
