from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass
class ECGRecord:
    name: str
    fs: int
    n_samples: int
    lead_names: list[str]
    signal_mv: np.ndarray  # [n_samples, n_leads]


_HDR_RE = re.compile(r"^(?P<name>\S+)\s+(?P<n_sig>\d+)\s+(?P<fs>\d+)\s+(?P<n_samples>\d+)")


def read_header(path: str | Path) -> dict:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    m = _HDR_RE.match(lines[0].strip())
    if not m:
        raise ValueError(f"Niepoprawny nagłówek HEA: {path}")
    info = {
        "name": m.group("name"),
        "n_sig": int(m.group("n_sig")),
        "fs": int(m.group("fs")),
        "n_samples": int(m.group("n_samples")),
        "leads": [],
        "gains": [],
        "baselines": [],
    }
    for line in lines[1:1 + info["n_sig"]]:
        parts = line.split()
        # przykładowo: file.dat 16 1000.0(0)/mV 16 0 -50 60504 0 I
        gain_part = parts[2]
        gain = float(gain_part.split("(")[0])
        baseline_match = re.search(r"\(([-+]?\d+)\)", gain_part)
        baseline = int(baseline_match.group(1)) if baseline_match else 0
        lead_name = parts[-1]
        info["leads"].append(lead_name)
        info["gains"].append(gain)
        info["baselines"].append(baseline)
    return info


def read_record(hea_path: str | Path) -> ECGRecord:
    hea_path = Path(hea_path)
    info = read_header(hea_path)
    dat_path = hea_path.with_suffix(".dat")
    raw = np.fromfile(dat_path, dtype=np.int16)
    expected = info["n_samples"] * info["n_sig"]
    if raw.size != expected:
        raise ValueError(
            f"Rozmiar DAT nie zgadza się z HEA: {raw.size} != {expected}"
        )
    raw = raw.reshape(info["n_samples"], info["n_sig"]).astype(np.float32)
    gains = np.asarray(info["gains"], dtype=np.float32)
    baselines = np.asarray(info["baselines"], dtype=np.float32)
    signal_mv = (raw - baselines[None, :]) / gains[None, :]
    return ECGRecord(
        name=info["name"],
        fs=info["fs"],
        n_samples=info["n_samples"],
        lead_names=info["leads"],
        signal_mv=signal_mv,
    )
