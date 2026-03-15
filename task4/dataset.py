from __future__ import annotations

from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_io import read_record, ECGRecord
from pipeline import preprocess_image, LEAD_GRID


def auto_trace_mask(clean_gray: np.ndarray) -> np.ndarray:
    thr = cv2.adaptiveThreshold(
        clean_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    out = np.zeros_like(thr)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= 6 and (w >= 2 or h >= 2):
            out[labels == i] = 255
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


def random_perspective(img: np.ndarray, mask: np.ndarray):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    jitter = 0.08
    dst = src.copy()
    dst[:, 0] += np.random.uniform(-jitter, jitter, size=4) * w
    dst[:, 1] += np.random.uniform(-jitter, jitter, size=4) * h
    M = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
    mask2 = cv2.warpPerspective(mask, M, (w, h), borderValue=0)
    return img2, mask2


def add_shadow_and_noise(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    out = gray.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
    sigma = np.random.uniform(0.35, 0.9) * max(h, w)
    illum = np.random.uniform(15, 45) * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2 * sigma * sigma))
    out += illum
    out += np.random.normal(0, np.random.uniform(1.5, 7.0), size=out.shape)
    if random.random() < 0.7:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_fold_artifacts(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    out = gray.copy().astype(np.float32)
    for _ in range(random.randint(1, 4)):
        pts = []
        axis = random.choice([0, 1])
        if axis == 0:
            x = np.random.uniform(0.15, 0.85) * w
            for y in np.linspace(0, h - 1, 8):
                pts.append([int(np.clip(x + np.random.uniform(-0.1, 0.1) * w, 0, w - 1)), int(y)])
        else:
            y = np.random.uniform(0.15, 0.85) * h
            for x in np.linspace(0, w - 1, 8):
                pts.append([int(x), int(np.clip(y + np.random.uniform(-0.1, 0.1) * h, 0, h - 1))])
        pts = np.asarray(pts, np.int32)
        cv2.polylines(out, [pts], False, color=float(np.random.uniform(110, 185)), thickness=random.choice([1, 2, 3]))
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_grid(canvas: np.ndarray, minor: int = 8, major_every: int = 5) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    scheme = random.choice([
        ((242, 238, 236), (214, 188, 188), (176, 132, 132)),
        ((245, 240, 228), (205, 175, 146), (145, 118, 92)),
        ((252, 250, 250), (220, 210, 210), (190, 170, 170)),
    ])
    paper, minor_color, major_color = scheme
    out[:] = paper

    for x in range(0, w, minor):
        color = major_color if (x // minor) % major_every == 0 else minor_color
        cv2.line(out, (x, 0), (x, h - 1), color, 1)
    for y in range(0, h, minor):
        color = major_color if (y // minor) % major_every == 0 else minor_color
        cv2.line(out, (0, y), (w - 1, y), color, 1)
    return out


def _lead_index(record: ECGRecord, lead_name: str) -> int:
    return record.lead_names.index(lead_name)


def render_record_to_page(record: ECGRecord, size=(1100, 1600)):
    h, w = size
    page = draw_grid(np.full((h, w, 3), 255, np.uint8))
    mask = np.zeros((h, w), np.uint8)

    seg_len = min(int(record.fs * 2.5), record.n_samples // 4)
    top_h = int(h * 0.74)
    row_h = top_h // 3
    col_w = w // 4
    amp_px_per_mv = row_h * 0.22

    # calibration pulse on left
    def draw_calibration(img, y_mid):
        x0 = 18
        pw = int(col_w * 0.08)
        ph = int(row_h * 0.35)
        pts = np.array([
            [x0, y_mid],
            [x0, y_mid - ph],
            [x0 + pw, y_mid - ph],
            [x0 + pw, y_mid],
        ], np.int32)
        cv2.polylines(img, [pts], False, (32, 32, 32), 2)

    for r, row in enumerate(LEAD_GRID):
        y_mid = int((r + 0.5) * row_h)
        draw_calibration(page, y_mid)
        for c, lead in enumerate(row):
            idx = _lead_index(record, lead)
            start = c * seg_len
            end = min(start + seg_len, record.n_samples)
            sig = record.signal_mv[start:end, idx]
            x0 = c * col_w
            x1 = (c + 1) * col_w
            xs = np.linspace(x0 + 10, x1 - 10, len(sig))
            ys = y_mid - sig * amp_px_per_mv
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(page, [pts], False, (30, 30, 30), 2)
            cv2.polylines(mask, [pts], False, 255, 2)
            cv2.putText(page, lead, (x0 + 65, y_mid + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1, cv2.LINE_AA)

    # long lead II
    y_mid = int(top_h + 0.5 * (h - top_h))
    draw_calibration(page, y_mid)
    idx = _lead_index(record, "II")
    sig = record.signal_mv[: min(record.n_samples, 4 * seg_len), idx]
    xs = np.linspace(10, w - 10, len(sig))
    ys = y_mid - sig * amp_px_per_mv
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(page, [pts], False, (30, 30, 30), 2)
    cv2.polylines(mask, [pts], False, 255, 2)
    cv2.putText(page, "II", (65, y_mid + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(page, "25mm/s", (120, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(page, "10mm/mV", (w // 3, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    return page, mask


class ECGPhotoDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], size=(704, 1024)):
        self.image_paths = [Path(p) for p in image_paths]
        self.size = size
        self.cache = []
        for p in self.image_paths:
            bgr = cv2.imread(str(p))
            _, gray, _ = preprocess_image(bgr)
            mask = auto_trace_mask(gray)
            self.cache.append((gray, mask, p.name))

    def __len__(self):
        return max(200, len(self.cache) * 80)

    def __getitem__(self, idx):
        gray, mask, _ = random.choice(self.cache)
        img = gray.copy()
        m = mask.copy()
        img, m = random_perspective(img, m)
        img = add_shadow_and_noise(img)
        img = add_fold_artifacts(img)
        img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(img).float()[None] / 255.0
        y = torch.from_numpy((m > 0).astype(np.float32))[None]
        return x, y


class SyntheticECGDataset(Dataset):
    def __init__(self, record_paths: list[str | Path], size=(704, 1024), length: int = 1200):
        self.record_paths = [Path(p) for p in record_paths]
        self.records = [read_record(p) for p in self.record_paths]
        self.size = size
        self.length = max(length, len(self.records) * 100)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        record = random.choice(self.records)
        page, mask = render_record_to_page(record)

        if random.random() < 0.85:
            page, mask = random_perspective(page, mask)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        gray = add_shadow_and_noise(gray)
        gray = add_fold_artifacts(gray)
        if random.random() < 0.35:
            gray = cv2.rotate(gray, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
            mask = cv2.rotate(mask, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
        gray = cv2.resize(gray, self.size[::-1], interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(gray).float()[None] / 255.0
        y = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return x, y
