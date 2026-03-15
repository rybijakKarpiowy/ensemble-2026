from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np
import torch

from model import TinyUNet


LEAD_GRID = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
SHORT_LEADS = [lead for row in LEAD_GRID for lead in row]
RHYTHM_LEAD = "II"
LONG_LEAD_KEY = RHYTHM_LEAD + "_long"
SUBMISSION_LEADS = SHORT_LEADS + [LONG_LEAD_KEY]
SHORT_DURATION_S = 2.5
LONG_DURATION_S = 10.0
TARGET_FS = 500
MM_PER_SECOND = 25.0
MM_PER_MV = 10.0
SHORT_MM = SHORT_DURATION_S * MM_PER_SECOND  # 62.5 mm
LONG_MM = LONG_DURATION_S * MM_PER_SECOND    # 250 mm


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_page_quad(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    area_img = img.shape[0] * img.shape[1]
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > best_area and area > 0.2 * area_img:
            best = approx[:, 0, :].astype(np.float32)
            best_area = area
    return order_points(best) if best is not None else None


def rectify_page(img: np.ndarray, out_size=(1600, 1100)) -> np.ndarray:
    quad = find_page_quad(img)
    if quad is None:
        return img.copy()
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, out_size)


def estimate_rotation(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0.0
    angles = []
    for ln in lines[:120]:
        rho, theta = ln[0]
        angle = (theta * 180 / np.pi) - 90
        if -20 < angle < 20:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def rotate_keep(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - w / 2
    M[1, 2] += (nh / 2) - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def crop_to_content(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(8, int(0.01 * (x1 - x0 + 1)))
    pad_y = max(8, int(0.01 * (y1 - y0 + 1)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.shape[1], x1 + pad_x)
    y1 = min(img.shape[0], y1 + pad_y)
    return img[y0:y1, x0:x1]


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    lo = np.percentile(gray, 1)
    hi = np.percentile(gray, 99)
    if hi <= lo:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_color_grid(page_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.abs(a - 128), np.abs(b - 128)).astype(np.uint8)
    k = max(7, int(round(page_bgr.shape[1] / 180)))
    if k % 2 == 0:
        k += 1
    hor = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    ver = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    grid = np.maximum(hor, ver)
    return cv2.GaussianBlur(grid, (0, 0), 1.1)


def make_clean_preview(page_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_gray(gray)
    grid_color = estimate_color_grid(page_bgr)
    cleaned = gray.astype(np.float32)
    lift = np.clip((grid_color.astype(np.float32) - 4.0) * 5.0, 0, 60)
    cleaned = np.clip(cleaned + lift, 0, 255)
    cleaned = cv2.fastNlMeansDenoising(cleaned.astype(np.uint8), None, 9, 7, 21)
    return cleaned


def build_trace_score(page_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_gray(gray)

    sigma = max(7.0, page_bgr.shape[1] / 180.0)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark = np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    vert_boost = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))

    grid_h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8))
    grid_v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8))
    grid_cross = np.minimum(grid_h, grid_v)
    grid_color = estimate_color_grid(page_bgr)

    score = (
        0.60 * dark.astype(np.float32)
        + 0.40 * blackhat.astype(np.float32)
        + 0.20 * vert_boost.astype(np.float32)
        - 0.40 * grid_cross.astype(np.float32)
        - 0.30 * grid_color.astype(np.float32)
    )
    score = np.clip(score, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (0, 0), 0.8)
    clean = make_clean_preview(page_bgr)
    return clean, score


def preprocess_image(img_bgr: np.ndarray, out_size=(1600, 1100)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    page = rectify_page(img_bgr, out_size)
    gray0 = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    angle = estimate_rotation(gray0)
    page = rotate_keep(page, angle)
    page = crop_to_content(page)
    clean, score = build_trace_score(page)
    return page, clean, score


def load_model(weights_path: str | Path, device: str = "cpu") -> TinyUNet:
    model = TinyUNet(in_ch=1, out_ch=1, base=16).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_model_score(gray: np.ndarray, model: TinyUNet, device: str = "cpu") -> np.ndarray:
    img = cv2.resize(gray, (1024, 704), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(img).float()[None, None] / 255.0
    ten = ten.to(device)
    with torch.no_grad():
        logits = model(ten)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(prob * 255.0, 0, 255).astype(np.uint8)


def _pick_top_peaks(signal: np.ndarray, n_peaks: int, min_distance: int) -> list[int]:
    candidates = []
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]:
            candidates.append(i)
    candidates.sort(key=lambda i: float(signal[i]), reverse=True)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) >= min_distance for prev in chosen):
            chosen.append(idx)
        if len(chosen) >= n_peaks:
            break
    return sorted(chosen)


def detect_row_centers(score: np.ndarray) -> list[int]:
    h, w = score.shape
    x0 = int(w * 0.18)
    roi = score[:, x0:]
    row_energy = roi.mean(axis=1).astype(np.float32)
    sigma_y = max(5.0, h / 120.0)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=sigma_y).reshape(-1)

    row_energy[: int(h * 0.12)] = row_energy.min()
    row_energy[int(h * 0.985):] = row_energy.min()

    centers = _pick_top_peaks(row_energy, n_peaks=4, min_distance=max(50, int(h * 0.12)))
    if len(centers) < 4:
        centers = np.linspace(int(h * 0.24), int(h * 0.92), 4).astype(int).tolist()
    return centers


def row_bounds_from_centers(centers: list[int], h: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for i, c in enumerate(centers):
        if i == 0:
            step = centers[i + 1] - c
            y0 = max(0, int(c - 0.7 * step))
        else:
            y0 = int((centers[i - 1] + c) / 2)
        if i == len(centers) - 1:
            step = c - centers[i - 1]
            y1 = min(h, int(c + 0.7 * step))
        else:
            y1 = int((c + centers[i + 1]) / 2)
        bounds.append((y0, y1))
    return bounds


def dp_track_path(score_region: np.ndarray, center_y: int | None = None, max_jump: int | None = None) -> np.ndarray:
    score = score_region.astype(np.float32)
    score = cv2.GaussianBlur(score, (1, 0), sigmaX=0, sigmaY=0.8)

    h, w = score.shape
    if max_jump is None:
        max_jump = max(10, min(28, h // 5))

    jumps = np.arange(-max_jump, max_jump + 1, dtype=np.int16)
    jump_cost = 0.05 * np.abs(jumps).astype(np.float32)

    dp = np.full((h, w), -1e9, dtype=np.float32)
    ptr = np.zeros((h, w), dtype=np.int16)

    ys = np.arange(h, dtype=np.float32)
    if center_y is None:
        dp[:, 0] = score[:, 0]
    else:
        dp[:, 0] = score[:, 0] - 0.01 * (ys - float(center_y)) ** 2

    for x in range(1, w):
        prev = dp[:, x - 1]
        for y in range(h):
            src = y - jumps
            valid = (src >= 0) & (src < h)
            cand = prev[src[valid]] - jump_cost[valid]
            best_idx = int(np.argmax(cand))
            best_src = src[valid][best_idx]
            dp[y, x] = score[y, x] + cand[best_idx]
            ptr[y, x] = np.int16(best_src)

    path = np.zeros(w, dtype=np.int16)
    path[-1] = np.int16(np.argmax(dp[:, -1]))
    for x in range(w - 1, 0, -1):
        path[x - 1] = ptr[path[x], x]

    path = cv2.GaussianBlur(path.astype(np.float32).reshape(1, -1), (0, 0), sigmaX=0.8).reshape(-1)
    return path


def trace_paths(score: np.ndarray) -> tuple[dict[str, dict], dict[str, object]]:
    h, w = score.shape
    centers = detect_row_centers(score)
    bounds = row_bounds_from_centers(centers, h)

    def find_calib_end_idx(path_array: np.ndarray) -> int:
        mm_px = w / LONG_MM
        search_limit = int(20 * mm_px)
        search_limit = min(search_limit, len(path_array) - 1)
        if search_limit < int(5 * mm_px):
            return 0

        segment = path_array[:search_limit]
        baseline = np.median(path_array)
        threshold_y = baseline - (4 * mm_px)
        high_points = np.where(segment < threshold_y)[0]

        if len(high_points) > 0:
            first_high = high_points[0]
            last_high = high_points[-1]
            if (last_high - first_high) > (2.5 * mm_px):
                for i in range(last_high, len(segment)):
                    if path_array[i] > baseline - (1 * mm_px):
                        margin = max(3, int(1.5 * mm_px))
                        return min(i + margin, len(path_array) - 1)
                return min(last_high + int(3 * mm_px), len(path_array) - 1)
        return 0

    def estimate_column_width(score_img: np.ndarray, row_bounds: list[tuple[int, int]], start_offset: int) -> int:
        signal_w = w - start_offset
        expected_w = int(signal_w * 0.94 / 4)
        expected_w = max(expected_w, 32)

        y_top = row_bounds[0][0]
        y_bottom = row_bounds[2][1]
        roi = score_img[y_top:y_bottom, start_offset:]
        col_sum = np.sum(roi, axis=0).astype(np.float32)
        bg = cv2.GaussianBlur(col_sum.reshape(1, -1), (0, 0), sigmaX=max(3.0, expected_w / 3)).reshape(-1)
        sharp_peaks = col_sum - bg

        search_margin = int(expected_w * 0.20)
        markers = []
        for i in range(1, 4):
            exp_x = i * expected_w
            x_start = max(0, int(exp_x - search_margin))
            x_end = min(len(sharp_peaks), int(exp_x + search_margin))
            if x_start >= x_end:
                continue
            window = sharp_peaks[x_start:x_end]
            if np.max(window) > np.std(sharp_peaks):
                peak_idx = x_start + int(np.argmax(window))
                markers.append(peak_idx)

        intervals = []
        if markers:
            intervals.append(markers[0])
            for i in range(len(markers) - 1):
                intervals.append(markers[i + 1] - markers[i])
        if intervals:
            avg_w = float(np.median(intervals))
            if 0.75 * expected_w <= avg_w <= 1.25 * expected_w:
                return int(avg_w)
        return expected_w

    y0_first, y1_first = bounds[0]
    test_path = dp_track_path(score[y0_first:y1_first, :], center_y=(y1_first - y0_first) // 2)
    start_x = find_calib_end_idx(test_path)
    col_w = estimate_column_width(score, bounds, start_x)
    end_x = min(w, start_x + 4 * col_w)

    col_bounds = []
    for i in range(4):
        x0 = start_x + i * col_w
        x1 = min(w, start_x + (i + 1) * col_w)
        if x1 > end_x:
            x1 = end_x
        col_bounds.append((int(x0), int(x1)))

    paths: dict[str, dict] = {}
    for r, (y0, y1) in enumerate(bounds[:3]):
        for c, (x0, x1) in enumerate(col_bounds):
            lead = LEAD_GRID[r][c]
            if x0 >= x1:
                x1 = x0 + 1
            region = score[y0:y1, x0:x1]
            path = dp_track_path(region, center_y=(y1 - y0) // 2)
            paths[lead] = {
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "path_y_local": path.tolist(),
            }

    y0, y1 = bounds[3]
    region_long = score[y0:y1, start_x:end_x]
    rhythm_path = dp_track_path(region_long, center_y=(y1 - y0) // 2)
    paths[LONG_LEAD_KEY] = {
        "bbox": [int(start_x), int(y0), int(end_x), int(y1)],
        "path_y_local": rhythm_path.tolist(),
    }

    layout = {
        "row_bounds": [[int(a), int(b)] for a, b in bounds],
        "col_bounds": [[int(a), int(b)] for a, b in col_bounds],
        "start_x": int(start_x),
        "end_x": int(end_x),
        "col_width_px": int(col_w),
        "width_px": int(w),
        "height_px": int(h),
    }
    return paths, layout


def render_path_mask(shape: tuple[int, int], paths: dict[str, dict], thickness: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for payload in paths.values():
        x0, y0, x1, y1 = payload["bbox"]
        path = np.asarray(payload["path_y_local"], dtype=np.float32)
        pts = np.stack([np.arange(len(path)), np.round(path).astype(np.int32)], axis=1)
        pts[:, 0] += x0
        pts[:, 1] += y0
        pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
        cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thickness=thickness)
    return mask


def overlay_paths(gray: np.ndarray, paths: dict[str, dict]) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_map = {
        "I": (255, 0, 0),
        "II": (255, 0, 0),
        "III": (255, 0, 0),
        "aVR": (0, 128, 255),
        "aVL": (0, 128, 255),
        "aVF": (0, 128, 255),
        "V1": (255, 0, 255),
        "V2": (255, 0, 255),
        "V3": (255, 0, 255),
        "V4": (255, 255, 0),
        "V5": (255, 255, 0),
        "V6": (255, 255, 0),
    }

    for lead, payload in paths.items():
        x0, y0, x1, y1 = payload["bbox"]
        path = np.asarray(payload["path_y_local"], dtype=np.float32)
        pts = np.stack([np.arange(len(path)), np.round(path).astype(np.int32)], axis=1)
        pts[:, 0] += x0
        pts[:, 1] += y0
        color = (0, 255, 0) if lead.endswith("_long") else color_map.get(lead, (0, 0, 255))
        cv2.polylines(rgb, [pts.astype(np.int32)], False, color, thickness=1)

    return rgb


def render_trace_only(shape: tuple[int, int], paths: dict[str, dict]) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    mask = render_path_mask(shape, paths, thickness=1)
    canvas[mask > 0] = 0
    return canvas


def _estimate_px_per_mm(paths: dict[str, dict]) -> float:
    widths = []
    for lead in SHORT_LEADS:
        if lead not in paths:
            continue
        x0, _, x1, _ = paths[lead]["bbox"]
        widths.append(max(1, x1 - x0))
    if widths:
        short_width_px = float(np.median(widths))
    elif LONG_LEAD_KEY in paths:
        x0, _, x1, _ = paths[LONG_LEAD_KEY]["bbox"]
        short_width_px = max(1.0, (x1 - x0) / 4.0)
    else:
        short_width_px = 250.0
    return max(short_width_px / SHORT_MM, 1e-3)


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if signal.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if signal.size == 1:
        return np.full(target_len, float(signal[0]), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, signal.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, signal).astype(np.float32)


def extract_signals(paths: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    px_per_mm = _estimate_px_per_mm(paths)
    px_per_mV = px_per_mm * MM_PER_MV

    for raw_lead, payload in paths.items():
        x0, y0, x1, y1 = payload["bbox"]
        path = np.asarray(payload["path_y_local"], dtype=np.float32)
        baseline = float(np.median(path))
        amp_px = (baseline - path).astype(np.float32)
        duration_s = LONG_DURATION_S if raw_lead == LONG_LEAD_KEY else SHORT_DURATION_S
        n_target = int(round(duration_s * TARGET_FS))
        amp_mv = _resample_1d(amp_px / px_per_mV, n_target)
        out[raw_lead] = {
            "lead": raw_lead,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "x_px": list(range(len(amp_px))),
            "y_px": amp_px.astype(float).tolist(),
            "duration_s": float(duration_s),
            "fs_hz": int(TARGET_FS),
            "signal_mv": amp_mv.astype(float).tolist(),
            "signal_mv_dtype": "float32",
            "submission_dtype": "float16",
        }
    return out


def build_submission_arrays(signals: dict[str, dict], include_long_lead: bool = True) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for lead in SHORT_LEADS:
        if lead in signals:
            arrays[lead] = np.asarray(signals[lead]["signal_mv"], dtype=np.float16)
    if include_long_lead and LONG_LEAD_KEY in signals:
        arrays[LONG_LEAD_KEY] = np.asarray(signals[LONG_LEAD_KEY]["signal_mv"], dtype=np.float16)
    return arrays


def make_submission_record(record_name: str, signals: dict[str, dict], include_long_lead: bool = True) -> dict[str, np.ndarray]:
    arrays = build_submission_arrays(signals, include_long_lead=include_long_lead)
    return {f"{record_name}_{lead}": values for lead, values in arrays.items()}


def classical_trace_mask(gray: np.ndarray) -> np.ndarray:
    dummy_score = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    paths, _ = trace_paths(dummy_score)
    return render_path_mask(gray.shape, paths)


def process_ecg_image(
    image_path: str | Path,
    weights_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    device: str = "cpu",
    save_artifacts: bool = True,
    include_long_lead: bool = True,
) -> dict[str, object]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    page, clean, score = preprocess_image(img)
    if weights_path is not None and Path(weights_path).exists():
        model = load_model(weights_path, device)
        model_score = predict_model_score(clean, model, device)
        score = np.maximum(score, model_score)

    paths, layout = trace_paths(score)
    mask = render_path_mask(clean.shape, paths, thickness=1)
    signals = extract_signals(paths)
    submission_arrays = build_submission_arrays(signals, include_long_lead=include_long_lead)
    trace_only = render_trace_only(clean.shape, paths)
    overlay = overlay_paths(clean, paths)

    result: dict[str, object] = {
        "page": page,
        "clean": clean,
        "score": score,
        "mask": mask,
        "trace_only": trace_only,
        "overlay": overlay,
        "paths": paths,
        "signals": signals,
        "submission_arrays": submission_arrays,
        "layout": layout,
    }

    if save_artifacts:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save_artifacts=True")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(out_dir / f"{stem}_page.png"), page)
        cv2.imwrite(str(out_dir / f"{stem}_clean.png"), clean)
        cv2.imwrite(str(out_dir / f"{stem}_score.png"), score)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_trace_only.png"), trace_only)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        with open(out_dir / f"{stem}_signals.json", "w", encoding="utf-8") as f:
            json.dump(signals, f, ensure_ascii=False)
        with open(out_dir / f"{stem}_layout.json", "w", encoding="utf-8") as f:
            json.dump(layout, f, ensure_ascii=False)

    return result


def run_inference(
    image_path: str | Path,
    weights_path: str | Path | None,
    out_dir: str | Path,
    device: str = "cpu",
):
    result = process_ecg_image(
        image_path=image_path,
        weights_path=weights_path,
        out_dir=out_dir,
        device=device,
        save_artifacts=True,
        include_long_lead=True,
    )
    return result["clean"], result["mask"], result["signals"]
