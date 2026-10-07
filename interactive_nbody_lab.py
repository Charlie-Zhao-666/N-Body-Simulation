import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, LogLocator, MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import sys
import copy
import json
import secrets
import csv
from io import BytesIO
from datetime import datetime
from dateutil.tz import gettz


import queue
import threading
from stable_orbits import STABLE_ORBITS as BASE_STABLE_ORBITS

# Extend only the interactive catalog; the original program and testers retain the base catalog.
# Li & Liao (2019), Collisionless periodic orbits in the free-fall three-body problem.
# Retain the authors' coordinates without shifting the center of mass; the published initial
# values have about 10 decimal places, so exact periodic closure is not guaranteed.
STABLE_ORBITS = [*BASE_STABLE_ORBITS, {
    "name": "Free-fall F1 (1, 0.8, 0.8) · Li-Liao",
    "N": 3,
    "seed": len(BASE_STABLE_ORBITS) + 1,  # Menu identifier; this seed does not generate the initial conditions.
    "masses": [1.0, 0.8, 0.8],
    "positions": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
                  [0.0009114239, 0.3019805958, 0.0]],
    "velocities": [[0.0, 0.0, 0.0] for _ in range(3)],
    "period": 1.8286248401,
    "dt": 1.8286248401 / 40000,
    "tot_time": 40000,
    "recommended_method": "2",
    "source": "https://numericaltank.sjtu.edu.cn/three-body/free-fall-3b/free-fall-3b-movies.htm",
    "note": "Masses: 1, 0.8, 0.8; initially at rest. RK4 selected; start with one period and adaptive stepping off.",
}]

G = 1.0

def dist(s1, s2):
    return np.linalg.norm(s2["pos"] - s1["pos"])


def dir_vector(s1, s2):
    return (s2["pos"] - s1["pos"]) / dist(s1, s2)


def acc(s1, s2):
    return G * s2["mass"] / dist(s1, s2)**2 * dir_vector(s1, s2)


def net_accelerations(s):
    result = np.zeros((len(s), 3), dtype=float)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            pair_acceleration = acc(s[i], s[j])
            result[i] += pair_acceleration
            result[j] -= pair_acceleration * s[i]["mass"] / s[j]["mass"]
    return result


def adaptive_step_size(s, max_dt, eta=0.03, previous_proposal=None):
    if not np.isfinite(max_dt) or max_dt <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError("The maximum time step and safety factor must be finite positive numbers.")
    proposed = float(max_dt)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            separation = np.linalg.norm(s[j]["pos"] - s[i]["pos"])
            speed = np.linalg.norm(s[j]["v"] - s[i]["v"])
            if not np.isfinite(separation) or separation <= 0 or not np.isfinite(speed):
                raise RuntimeError("Coincident particles or a non-finite state prevent adaptive time-step calculation.")
            mu = G * (s[i]["mass"] + s[j]["mass"])
            if mu > 0:
                gravity_time = separation / np.sqrt(mu / separation)
                proposed = min(proposed, eta * gravity_time)
            if speed > 0:
                flyby_time = separation / speed
                proposed = min(proposed, eta * flyby_time)
    if previous_proposal is not None:
        # Diagnostic output may clip a step; do not treat that as a physical
        # demand for small steps on the next iteration.
        proposed = min(proposed, 1.25 * previous_proposal)
    return proposed


def acceleration_power_band(numerator, denominator, trigger_ratio, max_power):
    """Require ratio > trigger_ratio for the first band.

    Higher exact powers enter their corresponding bands; for example, a ratio
    of 4 with trigger_ratio=2 belongs to band 2. Handle zeros first, then use
    binary search over integer powers to avoid logarithmic boundary errors.
    """
    if numerator == 0 or denominator == 0:
        return 0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        ratio = float(np.float64(numerator) / np.float64(denominator))
    if ratio <= trigger_ratio:
        return 0
    low, high = 1, max_power
    while low < high:
        mid = (low + high + 1) // 2
        if ratio >= trigger_ratio ** mid:
            low = mid
        else:
            high = mid - 1
    return low


class AccelerationRatioController:
    """Track per-star reference levels and use the smallest proposed system step.

    Persistent references detect acceleration growth accumulated across steps.
    Each nonzero initial |a| sets that star's scale. Initially zero stars use
    the system's largest initial |a| as a control scale without changing their
    physical acceleration or using a tiny first nonzero value as a denominator.
    A reference becomes scale * trigger_ratio**level only when its local level
    changes; it is not reset at every step. For nonzero current acceleration,
    reference/current must exceed trigger_ratio before the level decreases.
    This hysteresis avoids repeated switching around a single threshold.
    """
    def __init__(self, initial_acceleration, max_dt, trigger_ratio=2.0, max_power=3):
        self.max_dt = float(max_dt)
        self.trigger_ratio = float(trigger_ratio)
        self.max_power = int(max_power)
        self.scales = np.linalg.norm(np.asarray(initial_acceleration, dtype=float), axis=1)
        if not np.isfinite(self.scales).all() or self.scales.size == 0:
            raise ValueError("Initial accelerations must form a non-empty finite vector.")
        positive = self.scales > 0
        if np.any(positive):
            self.scales[~positive] = np.max(self.scales[positive])
        self.levels = np.zeros(len(self.scales), dtype=int)
        self.references = self.scales.copy()
        self.global_level = 0
        self.last_event = None

    def choose(self, current_acceleration, current_time):
        current = np.linalg.norm(np.asarray(current_acceleration, dtype=float), axis=1)
        if current.shape != self.scales.shape or not np.isfinite(current).all():
            raise ValueError("The current accelerations have an incompatible size or contain non-finite values.")
        self.last_event = None
        # If all initial net accelerations are zero, the first nonzero state only initializes scales.
        uninitialized = self.scales == 0
        if np.any(uninitialized) and np.any(current > 0):
            self.scales[uninitialized] = np.max(current)
            self.references[uninitialized] = self.scales[uninitialized]
        valid = (current > 0) & (self.references > 0) & ~uninitialized
        old_levels = self.levels.copy()
        old_references = self.references.copy()
        for i in np.flatnonzero(valid):
            increase = acceleration_power_band(current[i], self.references[i],
                                               self.trigger_ratio, self.max_power)
            if increase:
                self.levels[i] = min(self.max_power, self.levels[i] + increase)
            else:
                decrease = acceleration_power_band(self.references[i], current[i],
                                                   self.trigger_ratio, self.max_power)
                self.levels[i] = max(0, self.levels[i] - decrease)
            # Integer levels set the reference; reaching the cap does not let spikes raise it further.
            if self.levels[i] != old_levels[i]:
                self.references[i] = self.scales[i] * self.trigger_ratio ** int(self.levels[i])
        next_level = int(np.max(self.levels))
        h = float(self.max_dt / self.trigger_ratio ** next_level)
        if next_level != self.global_level:
            drivers = np.flatnonzero(self.levels == next_level) + 1
            self.last_event = dict(time=float(current_time),
                                   previous_level=self.global_level, level=next_level,
                                   dt=h, controlling_stars=drivers.tolist(),
                                   star_levels=self.levels.tolist(),
                                   acceleration_magnitudes=current.tolist(),
                                   references_before=old_references.tolist(),
                                   references_after=self.references.tolist())
        self.global_level = next_level
        return h, next_level


def Ek(s1):
    return 0.5 * s1["mass"] * np.linalg.norm(s1["v"])**2


def Ep(s1, s2):
    return - G * s1["mass"] * s2["mass"] / dist(s1, s2)


def tot_E(s):
    sigma_Ep = 0
    sigma_Ek = 0
    for i in range(len(s)):
        sigma_Ek += Ek(s[i])
        for j in range(i+1, len(s)):
            sigma_Ep += Ep(s[i], s[j])
    return sigma_Ek + sigma_Ep


def Px(s):
    sigma_Px = 0
    for i in range(len(s)):
        sigma_Px += s[i]["mass"] * s[i]["v"][0]
    return sigma_Px


def Py(s):
    sigma_Py = 0
    for i in range(len(s)):
        sigma_Py += s[i]["mass"] * s[i]["v"][1]
    return sigma_Py


def Pz(s):
    sigma_Pz = 0
    for i in range(len(s)):
        sigma_Pz += s[i]["mass"] * s[i]["v"][2]
    return sigma_Pz


def tot_P(s):
    a = np.array([Px(s), Py(s), Pz(s)])
    return a


def tot_P_mag(s):
    return np.linalg.norm(tot_P(s))


def L(s):
    sigma_L = np.zeros(3)
    for i in range(len(s)):
        sigma_L += np.cross(s[i]["pos"], s[i]["mass"] * s[i]["v"])
    return sigma_L


def CoM_pos(s):
    tot_m = 0
    mr = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mr += s[i]["mass"] * s[i]["pos"]
    return mr/tot_m


def CoM_v(s):
    tot_m = 0
    mv = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mv += s[i]["mass"] * s[i]["v"]
    return mv/tot_m


def test(s):
    return {
        "energy": tot_E(s),
        "momentum": tot_P(s),
        "angular_momentum": L(s),
        "com_pos": CoM_pos(s),
        "com_v": CoM_v(s)
    }


def RMSE(data):
    a = 0
    for i in range(1, len(data)):
        a += (data[0] - data[i]) ** 2
    mse = a / (len(data) - 1)
    return np.sqrt(mse)


import hashlib
import html
from urllib.parse import quote
import os
import re
import tempfile
import threading


RUN_HISTORY_DIRECTORY = Path(__file__).resolve().parent / "graph and data"
LEGACY_HISTORY_PATH = Path(__file__).resolve().parent / "运行记录" / "运行记录.csv"
RUN_HISTORY_COLUMNS = [
    "Date", "Time", "Method", "Initial_condition", "dt_initial",
    "Steps_set", "Steps_actual", "Adaptive_method", "Adaptive_param", "K", "E_RMSE", "E_nRMSE", "Position_RMSE", "Position_NRMSE",
    "Px_RMSE", "Py_RMSE", "Pz_RMSE", "Lx_RMSE", "Ly_RMSE", "Lz_RMSE",
    "Vcmx_RMSE", "Vcmy_RMSE", "Vcmz_RMSE", "Device", "Compute_time_s", "Sim_T", "Plot",
]
_HISTORY_FLOAT_COLUMNS = {4, 8, *range(10, 23), 24, 25}
_HISTORY_INDEX_LOCK = threading.RLock()


def build_run_history_record(metadata, diagnostics):
    """Build the same RMSE summary used by the conservation tab."""
    started = datetime.fromisoformat(metadata["run_started_at"])
    energy = np.asarray([item["energy"] for item in diagnostics], dtype=float)
    error = float(RMSE(energy))
    scale = metadata["energy_scale"]
    adaptive_method = adaptive_param = adaptive_k = "NA"
    if metadata["adaptive_enabled"]:
        if metadata["adaptive_algorithm"] == "acceleration_ratio":
            adaptive_method, adaptive_param, adaptive_k = ("AccelerationRatio",
                metadata["acceleration_trigger_ratio"], metadata["acceleration_max_power"])
        else:
            adaptive_method, adaptive_param = "Timescale", metadata["adaptive_eta"]
    initial = (metadata["case_name"] if metadata["config"]["initial_mode"] == "orbit"
               else str(metadata["seed"]))
    row = [started.strftime("%Y-%m-%d"), started.strftime("%H:%M:%S"),
           _history_english(metadata["method_name"]), _history_english(initial), metadata["dt"],
           metadata["requested_steps"], metadata["steps"], adaptive_method, adaptive_param, adaptive_k,
           error, error / scale if scale else np.nan, "NA", "NA"]
    for key in ("momentum", "angular_momentum", "com_v"):
        values = np.asarray([item[key] for item in diagnostics], dtype=float)
        row.extend(float(RMSE(values[:, axis])) for axis in range(3))
    row.extend([metadata["compute_backend"].upper(), metadata["elapsed_seconds"],
                metadata["simulation_end_time"]])
    row = ["NA" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
           else value.item() if isinstance(value, np.generic) else value for value in row]
    return {"metadata": metadata, "row": row + ["NA"]}


def _history_english(value):
    text = str(value)
    for before, after in (
        ("加速度倍率法（累计参考）", "AccelerationRatio"),
        ("时间尺度法", "Timescale"), ("跳蛙法", "Leapfrog"),
        ("四阶龙格库塔", "RK4"), ("龙格-库塔", "Runge-Kutta"),
        ("自由落体", "Free-fall"), ("原算法", "original"),
        ("固定步长", "Fixed"), ("；", ";"), ("（", "("), ("）", ")"),
    ):
        text = text.replace(before, after)
    return " ".join(text.split())


def _archive_write(path, data, replace=True):
    """Publish a complete file; immutable record files never overwrite another run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def _archive_record_path(directory, run_id):
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()
    return directory / "_records" / (digest + ".json")


def _archive_publish_record(directory, entry):
    path = _archive_record_path(directory, entry["run_id"])
    _archive_write(path, json.dumps(entry, ensure_ascii=False, allow_nan=False,
                                   indent=2).encode("utf-8"), replace=False)
    return json.loads(path.read_text(encoding="utf-8"))


def _history_expand_row(row):
    """Read the old 22/23-column schema without changing its original JSON/CSV."""
    row = list(row)
    if len(row) == 22:
        row.append("NA")
    if len(row) == 23:
        adaptive = _history_english(row[7])
        method = param = power = "NA"
        if "AccelerationRatio" in adaptive or "Timescale" in adaptive:
            method = "AccelerationRatio" if "AccelerationRatio" in adaptive else "Timescale"
            parameter = "N" if method == "AccelerationRatio" else "eta"
            match = re.search(r"\b" + parameter + r"\s*=\s*([^;\s]+)", adaptive)
            param = match.group(1) if match else "NA"
            match = re.search(r"\bK\s*=\s*(\d+)", adaptive)
            power = int(match.group(1)) if method == "AccelerationRatio" and match else "NA"
        elif adaptive != "NA":
            method = adaptive
        row = row[:7] + [method, param, power] + row[8:]
    if len(row) == 25:
        row = row[:12] + ["NA", "NA"] + row[12:]
    if len(row) != 27:
        raise ValueError("Run summary must have 27 columns (or the legacy 22/23/25-column format).")
    return row


def _archive_display_row(row):
    row = _history_expand_row(row)
    display = []
    for i, value in enumerate(row):
        if i == 8 and row[7] == "AccelerationRatio" and value != "NA":
            display.append(f"{float(value):.1f}")
        elif i == 8 and row[7] == "Timescale" and value != "NA":
            from decimal import Decimal
            text = format(Decimal(str(value)), "f")
            display.append(text.rstrip("0").rstrip(".") if "." in text else text)
        elif (i in _HISTORY_FLOAT_COLUMNS and value != "NA"
              and not (i in (12, 13) and str(value).startswith("reference_"))):
            display.append(f"{float(value):.6e}")
        else:
            display.append(" ".join(str(value).split()))
    return display


def rebuild_run_history(directory=None):
    """Serialize index generation while preserving independent record files."""
    with _HISTORY_INDEX_LOCK:
        return _rebuild_run_history(directory)


def _rebuild_run_history(directory=None):
    """Rebuild readable indexes from independent, full-precision run records."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((directory / "_records").glob("*.json"))]
    entries.sort(key=lambda item: (item["row"][0], item["row"][1], item["run_id"]))
    rows = []
    for entry in entries:
        row = _history_expand_row(entry["row"])
        position_path = directory / "_position_comparisons" / _archive_record_path(directory, entry["run_id"]).name
        if position_path.exists():
            position = json.loads(position_path.read_text(encoding="utf-8"))
            if position["run_id"] != entry["run_id"]:
                raise ValueError("Position comparison run ID mismatch.")
            row[12:14] = [position["rmse"], position["nrmse"]]
        rows.append(_archive_display_row(row))
    # Keep stored filenames relative; build file URLs only for the current display location.
    display_rows = [row[:-1] + [((directory / row[-1]).resolve().as_uri()
                               if row[-1] != "NA" else "NA")] for row in rows]
    widths = [max(len(title), *(len(row[i]) for row in display_rows)) if display_rows else len(title)
              for i, title in enumerate(RUN_HISTORY_COLUMNS)]
    numeric = _HISTORY_FLOAT_COLUMNS | {5, 6, 9}
    def aligned(row):
        return "  ".join(value.rjust(widths[i]) if i in numeric else value.ljust(widths[i])
                         for i, value in enumerate(row)).rstrip()
    preamble = [
        "# N-body simulation run history; one completed run per row.",
        "# Code units unless specified. Compute_time_s: computation seconds; Sim_T: physical simulation time.",
        "# dt_initial is the initial dt / adaptive upper limit; Steps_set is the initial step count.",
        "# Conserved-quantity RMSE uses diagnostic samples after t=0, measured against the initial value.",
        "# E_nRMSE = E_RMSE / (initial kinetic energy + abs(initial potential energy)).",
        "# Position_RMSE / Position_NRMSE compare with the selected batch reference; disabled or pending = NA.",
        "# Position metrics use a uniform time grid including t=0; previous-test comparisons are in batch_*_comparisons.dat.",
        "# reference_N marks the reference row; N counts completed other-test comparisons against it in the same batch.",
        "# _position_comparisons/*.json retain reference identity, interval, normalization and full-precision results.",
        "# Adaptive_param is eta for Timescale or N for AccelerationRatio; K is only used by AccelerationRatio.",
        "# NA: not used, unavailable, or undefined. Legacy timestamps are preserved as recorded.",
        "# Plot contains the image file URL for that row; NA means no saved figure is available.",
        "# URL clicking in this text file depends on the editor; index.html provides clickable Plot links.",
        "# Adaptive_param uses compact decimals (eta) or one decimal (N); other numbers use scientific notation; JSON retains stored precision.",
        "", "",
        "# " + aligned(RUN_HISTORY_COLUMNS), "# " + "  ".join("-" * width for width in widths),
    ]
    _archive_write(directory / "runs.dat", ("\n".join(
        preamble + ["  " + aligned(row) for row in display_rows]) + "\n").encode("utf-8"))
    html_rows = []
    for row in rows:
        cells = [html.escape(value) for value in row]
        if row[-1] != "NA":
            image_href = quote(row[-1].replace("\\", "/"), safe="/")
            image_path = (directory / row[-1]).resolve()
            cells[-1] = (f'<a href="{html.escape(image_href, quote=True)}" '
                         f'target="_blank" rel="noopener" title="{html.escape(str(image_path), quote=True)}">'
                         f'{html.escape(image_path.as_uri())}</a>')
        html_rows.append("<tr>" + "".join("<td>" + value + "</td>" for value in cells) + "</tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>N-body run history</title>"
            "<style>body{font:14px system-ui;margin:24px;color:#172437}table{border-collapse:collapse;"
            "font:13px monospace;white-space:nowrap}th,td{padding:9px 12px;border:1px solid #dce2eb}"
            "th{position:sticky;top:0;background:#e9eff8}tr:nth-child(even){background:#f6f8fb}"
            "th:nth-child(9),td:nth-child(9){text-align:right;font-variant-numeric:tabular-nums}"
            "a{color:#155db0}</style><h1>N-body run history</h1>"
            "<p><a href='runs.dat'>Aligned text table</a> &middot; "
            "<a href='README.txt'>Column definitions and precision</a></p>"
            "<p>Click the URL in a row's Plot column to open its conservation and error analysis figure in a new tab. "
            "NA means the original figure is unavailable.</p><table><thead><tr>"
            + "".join("<th>" + html.escape(title) + "</th>" for title in RUN_HISTORY_COLUMNS)
            + "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table></html>")
    _archive_write(directory / "index.html", page.encode("utf-8"))
    notes = ("N-body automatic run archive\n\n"
             "This graph and data folder is created beside the Python script, independent of the working directory.\n"
             "To share the archive, copy this entire folder. index.html uses relative image links.\n"
             "Displayed file URLs in the Plot column are refreshed when the program opens the archive.\n\n"
             "runs.dat: aligned text table, best viewed using a monospaced font. Long names are not cut.\n"
             "index.html: the same history with clickable links to the PNG figures.\n"
             "_records/: one immutable JSON summary per run; full numerical precision and run ID.\n"
             "YYYY-MM-DD_HH-MM-SS.png: conservation and error analysis figure for its recorded run.\n"
             "The filename uses the same date and time as its table row (colons become hyphens).\n"
             "If a second is already occupied, the real start-time microseconds appear in both\n"
             "the filename and Time column; existing figures are never overwritten.\n"
             "The Plot column in each DAT and HTML row displays that run's image file URL.\n"
             "Whether DAT URLs can be clicked depends on the text editor; index.html provides\n"
             "direct links in the Plot cells. HTML targets and stored filenames remain relative,\n"
             "so the links work when the entire archive folder is moved to another computer.\n\n"
             "The original adaptive field is split into method, parameter, and K; the final Plot column links the figure.\n"
             "Date and Time refer to the simulation start. started_at in JSON retains timezone\n"
             "information when available. Legacy CSV timestamps have unknown timezone.\n"
             "dt_initial is the chosen initial dt / adaptive maximum. Steps_set is ceil(Sim_T/dt_initial).\n"
             "Steps_actual counts completed integration steps, including a shortened final step.\n"
             "Adaptive_method is NA, Timescale, or AccelerationRatio. Adaptive_param is eta for Timescale\n"
             "or the acceleration trigger N for AccelerationRatio. K is the maximum cumulative\n"
             "exponent, used only by AccelerationRatio. All three fields are NA for fixed dt.\n"
             "Conserved-quantity RMSE excludes the initial sample and compares diagnostics against the initial value.\n"
             "Position_RMSE and Position_NRMSE use the selected batch reference and a common uniform time grid.\n"
             "Disabled, pending, or unavailable position metrics are NA; reference_N marks a reference and its completed comparison count.\n"
             "_position_comparisons/ stores comparison overlays without altering immutable original run records.\n"
             "Comparison definitions and previous-test results are in the companion batch JSON and comparison DAT.\n"
             "E_nRMSE divides energy RMSE by initial kinetic energy + abs(initial potential energy).\n"
             "P, L, and Vcm mean total momentum, total angular momentum, and center-of-mass velocity.\n"
             "Compute_time_s is elapsed real time for the simulation loop and part of result preparation, in seconds.\n"
             "It excludes subsequent plotting and file saving; Sim_T is physical simulation time in code units.\n"
             "DAT and HTML display scientific notation rounded to six decimal places; JSON preserves\n"
             "all stored digits. NA marks an unused setting or unavailable / undefined value.\n"
             "Old CSV rows are imported read-only, including repeated runs. Their missing figures\n"
             "are marked NA; the original CSV is not modified. The table and HTML can be rebuilt\n"
             "from _records. Independent records prevent concurrent runs from overwriting one another.\n")
    _archive_write(directory / "README.txt", notes.encode("utf-8"))
    return directory / "runs.dat"


def migrate_legacy_history(directory=None, legacy_path=None):
    """Read the old 22-column CSV without editing it; repeated imports are harmless."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    legacy_path = Path(legacy_path) if legacy_path is not None else LEGACY_HISTORY_PATH
    if not legacy_path.exists():
        return 0
    with legacy_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if len(header) != 22:
            raise ValueError("Legacy history must contain exactly 22 columns; source was not modified.")
        imported = 0
        for index, original in enumerate(reader):
            if not original:
                continue
            if len(original) != 22:
                raise ValueError(f"Legacy row {index + 2} does not have 22 columns.")
            identity = json.dumps([index, original], ensure_ascii=False)
            run_id = "legacy_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
            row = _history_expand_row([_history_english(value) for value in original])
            # Strings retain every original digit; the readable table formats only a display copy.
            for column in _HISTORY_FLOAT_COLUMNS | {5, 6, 9}:
                if str(row[column]).lower() in {"nan", "inf", "-inf", ""}:
                    row[column] = "NA"
            path = _archive_record_path(directory, run_id)
            existed = path.exists()
            _archive_publish_record(directory, {"run_id": run_id, "row": row,
                                    "started_at": "NA", "source": "legacy_csv",
                                    "legacy_source": str(legacy_path.resolve())})
            imported += not existed
    rebuild_run_history(directory)
    return imported


def _archive_save_png(directory, record, png):
    """Reserve a date/time filename without reusing another run's image."""
    pending = record.get("pending_plot_filename")
    if pending is not None:
        pending_path = directory / pending
        if pending_path.name != pending:
            raise ValueError("An image filename must not contain a directory.")
        if pending_path.is_file() and pending_path.read_bytes() == png:
            return pending
    row = _history_expand_row(record["row"])
    started = datetime.fromisoformat(record["metadata"]["run_started_at"])
    candidates = [(row[1], f"{row[0]}_{str(row[1]).replace(':', '-')}.png")]
    fractional_time = started.strftime("%H:%M:%S.%f")
    if row[1] != fractional_time:
        candidates.append((fractional_time,
                           f"{row[0]}_{fractional_time.replace(':', '-')}.png"))
    for time_label, filename in candidates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d{1,6})?\.png", filename):
            raise ValueError("The image filename requires a valid date and time from its data row.")
        if _archive_write(directory / filename, png, replace=False):
            record["pending_plot_filename"] = filename
            record["row"][1] = time_label
            return filename
    raise FileExistsError(
        "Both the second and exact microsecond image names are occupied. "
        "Existing images were preserved; move the conflicting file before retrying.")


def save_run_history(record, directory=None):
    """Save PNG first, then its immutable summary, and rebuild both readable indexes."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    metadata = record["metadata"]
    run_id = str(metadata["run_id"])
    record_path = _archive_record_path(directory, run_id)
    with _HISTORY_INDEX_LOCK:
        if record_path.exists():
            entry = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            png = record.get("png_bytes")
            if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("The conservation figure must be rendered before saving this run.")
            filename = _archive_save_png(directory, record, png)
            row = _history_expand_row(record["row"])
            row[-1] = filename
            entry = _archive_publish_record(directory, {"run_id": run_id, "row": row,
                                            "started_at": metadata["run_started_at"], "source": "simulation"})
        path = rebuild_run_history(directory)
    record["row"][1] = entry["row"][1]
    record["row"][-1] = entry["row"][-1]
    metadata.update(history_saved_run_id=run_id, history_dat_path=str(path.resolve()),
                    history_csv_path=str(path.resolve()),
                    history_png_path=str((directory / entry["row"][-1]).resolve()))
    return path

# leapfrog
def leapfrog(stars, step_dt, initial_acceleration=None):
    N = len(stars)
    h = step_dt
    if initial_acceleration is None:
        accs = [np.zeros(3) for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                a = acc(stars[i], stars[j])
                accs[i] += a
                accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]
    else:
        accs = np.asarray(initial_acceleration)


    # half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * h * accs[i]

    # full step position
    for i in range(N):
        stars[i]["pos"] += stars[i]["v"] * h

    # new acceleartion
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = acc(stars[i], stars[j])
            accs[i] += a
            accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]

    # another half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * h * accs[i]


    # These accelerations were evaluated at the completed step's positions.
    return np.asarray(accs)


# Runge-Kutta 4th order method
def rk4(stars, step_dt, initial_acceleration=None):
    N = len(stars)
    h = step_dt
    if initial_acceleration is None:
        accs = [np.zeros(3) for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                a = acc(stars[i], stars[j])
                accs[i] += a
                accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]
    else:
        accs = np.asarray(initial_acceleration)


    # K1
    k1 = accs
    v1 = np.array([star["v"] for star in stars])

    # K2
    v2 = v1 + 0.5 * h * np.array(k1)
    k2 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * h * v1[i],
            "v": v2[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k2[i] += a
            k2[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K3
    v3 = v1 + 0.5 * h * np.array(k2)
    k3 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * h * v2[i],
            "v": v3[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k3[i] += a
            k3[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K4
    v4 = v1 + h * np.array(k3)
    k4 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + h * v3[i],
            "v": v4[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k4[i] += a
            k4[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # Update positions and velocities
    for i in range(N):
        stars[i]["pos"] += h * (v1[i] + 2*v2[i] + 2*v3[i] + v4[i]) / 6
        stars[i]["v"] += h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6




    # k4 uses a predicted position, not the final RK4 position. Record the
    # physical acceleration by evaluating gravity at the completed state.
    return net_accelerations(stars)




# GPU is optional: importing/running the CPU mode never imports CuPy.
_GPU_FIELDS_SOURCE = r'''
// NVRTC provides device math intrinsics without host C headers.
extern "C" __global__
void nbody_fields(const double* p, const double* v, const double* mass,
                  double* acceleration, double* potential, double* timescale,
                  const int n, const double gravity, const int need_times) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double ax = 0.0, ay = 0.0, az = 0.0, u = 0.0, tau = __longlong_as_double(0x7ff0000000000000LL);
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = p[3*j] - p[3*i];
        double dy = p[3*j+1] - p[3*i+1];
        double dz = p[3*j+2] - p[3*i+2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (!(r2 > 0.0) || !isfinite(r2)) {
            ax = ay = az = u = tau = __longlong_as_double(0x7ff8000000000000LL);
            break;
        }
        double r = sqrt(r2);
        double factor = gravity * mass[j] / (r2 * r);
        ax += factor * dx; ay += factor * dy; az += factor * dz;
        // Each pair appears in two rows; halve its contribution here.
        u -= 0.5 * gravity * mass[i] * mass[j] / r;
        if (need_times) {
            double mu = gravity * (mass[i] + mass[j]);
            if (mu > 0.0) tau = fmin(tau, r / sqrt(mu / r));
            double vx = v[3*j] - v[3*i];
            double vy = v[3*j+1] - v[3*i+1];
            double vz = v[3*j+2] - v[3*i+2];
            double speed = sqrt(vx*vx + vy*vy + vz*vz);
            if (speed > 0.0) tau = fmin(tau, r / speed);
        }
    }
    acceleration[3*i] = ax;
    acceleration[3*i+1] = ay;
    acceleration[3*i+2] = az;
    potential[i] = u;
    timescale[i] = tau;
}
'''


class GPUState:
    """Compute direct-summation gravity in float64 while keeping state on the GPU.

    Copy each step's recorded state to the host in a batch for plotting.
    Each CUDA thread handles one star and sums all gravitational contributions.
    Avoiding an N-by-N distance matrix keeps auxiliary device storage O(N);
    force evaluation still requires O(N**2) work. No softening or fast-math
    approximations are added. GPU summation order differs from the CPU, so
    bitwise-identical results are not guaranteed.
    """
    def __init__(self, stars, adaptive=False):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No usable NVIDIA CUDA GPU was detected.")
            self.cp = cp
            cp.cuda.Device(0).use()
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties["name"]
            self.device_name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
            self.n = len(stars)
            self.adaptive = adaptive
            self.host_masses = np.array([s["mass"] for s in stars], dtype=np.float64)
            self.mass = cp.asarray(self.host_masses)
            self.pos = cp.asarray(np.array([s["pos"] for s in stars], dtype=np.float64))
            self.vel = cp.asarray(np.array([s["v"] for s in stars], dtype=np.float64))
            self.kernel = cp.RawKernel(_GPU_FIELDS_SOURCE, "nbody_fields",
                                       options=("--std=c++11", "--fmad=false"))
            self.refresh_snapshot()
        except Exception as exc:
            raise RuntimeError(
                f"GPU initialization failed: {exc}\nCurrent Python: {sys.executable}\n"
                "GPU mode requires a working NVIDIA driver and CuPy. Select CPU mode to continue."
            ) from exc

    def fields(self, pos, vel, need_times=False):
        cp = self.cp
        acceleration = cp.empty_like(pos)
        potential = cp.empty(self.n, dtype=cp.float64)
        timescale = cp.empty(self.n, dtype=cp.float64)
        self.kernel(((self.n + 127) // 128,), (128,),
                    (pos, vel, self.mass, acceleration, potential, timescale,
                     np.int32(self.n), np.float64(G), np.int32(need_times)))
        return acceleration, potential, timescale

    def refresh_snapshot(self):
        cp = self.cp
        self.acceleration, potential, timescale = self.fields(self.pos, self.vel, self.adaptive)
        kinetic = .5 * cp.sum(self.mass[:, None] * self.vel**2)
        scalars = cp.stack((kinetic, cp.sum(potential), cp.min(timescale)))
        packet = cp.asnumpy(cp.concatenate((
            self.pos.ravel(), self.vel.ravel(), self.acceleration.ravel(), scalars)))
        n3 = 3 * self.n
        if not np.isfinite(packet[:-1]).all() or np.isnan(packet[-1]):
            raise RuntimeError("The GPU encountered coincident particles or a non-finite state. Check the initial conditions or reduce dt.")
        self.host_pos = packet[:n3].reshape(self.n, 3)
        self.host_vel = packet[n3:2*n3].reshape(self.n, 3)
        self.host_acceleration = packet[2*n3:3*n3].reshape(self.n, 3)
        self.energy = float(packet[-3] + packet[-2])
        self.energy_scale = float(packet[-3] - packet[-2])
        self.fastest_timescale = float(packet[-1])

    def propose_step(self, max_dt, eta, previous_proposal):
        proposal = min(max_dt, eta * self.fastest_timescale)
        if previous_proposal is not None:
            proposal = min(proposal, 1.25 * previous_proposal)
        return proposal

    def advance(self, h, method):
        if method == "1":
            self.vel += .5 * h * self.acceleration
            self.pos += h * self.vel
            next_acceleration, _, _ = self.fields(self.pos, self.vel)
            self.vel += .5 * h * next_acceleration
        else:
            a1, v1 = self.acceleration, self.vel
            v2 = v1 + .5 * h * a1
            a2, _, _ = self.fields(self.pos + .5 * h * v1, v2)
            v3 = v1 + .5 * h * a2
            a3, _, _ = self.fields(self.pos + .5 * h * v2, v3)
            v4 = v1 + h * a3
            a4, _, _ = self.fields(self.pos + h * v3, v4)
            self.pos += h * (v1 + 2*v2 + 2*v3 + v4) / 6
            self.vel += h * (a1 + 2*a2 + 2*a3 + a4) / 6
        # Re-evaluate the completed state, including pairwise speed timescales.
        self.refresh_snapshot()
        return self.host_acceleration, self.energy

    def diagnostics(self):
        # These O(N) reductions use the already transferred snapshot.
        # The O(N²) potential energy and force have been computed on the GPU.
        momentum = self.host_masses[:, None] * self.host_vel
        total_mass = self.host_masses.sum()
        return {
            "energy": self.energy,
            "momentum": momentum.sum(axis=0),
            "angular_momentum": np.cross(self.host_pos, momentum).sum(axis=0),
            "com_pos": (self.host_masses[:, None] * self.host_pos).sum(axis=0) / total_mass,
            "com_v": momentum.sum(axis=0) / total_mass,
        }



class SimulationCancelled(Exception):
    """Signal user cancellation while preserving the previous successful view."""


def validate_config(config):
    """Validate parameters before computation to catch invalid input early."""
    result = dict(config)
    result.setdefault("adaptive_algorithm", "timescale")
    if result["adaptive_algorithm"] not in ("timescale", "acceleration_ratio"):
        raise ValueError("Select the time-scale or acceleration-ratio algorithm.")
    result.setdefault("compute_backend", "cpu")
    result.setdefault("particle_count", 5)
    if result["compute_backend"] not in ("cpu", "gpu"):
        raise ValueError("Select CPU or GPU computation.")
    for key, title in (("dt", "Time step"), ("total_time", "Simulation duration")):
        try:
            result[key] = float(result[key])
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"{title} must be a number.") from None
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{title} must be finite and positive.")
    for key, title in (("sample_every", "Sampling interval"), ("seed", "Seed"), ("particle_count", "Random particle count")):
        try:
            value = float(result[key])
            if not np.isfinite(value) or value != int(value):
                raise ValueError
            result[key] = int(value)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ValueError(f"{title} must be an integer.") from None
    if not 1 <= result["particle_count"] <= 10000:
        raise ValueError("The random particle count must be an integer from 1 to 10000; complete per-step records use memory.")
    if result["sample_every"] < 1:
        raise ValueError("The sampling interval must be at least one base step.")
    if not 0 <= result["seed"] <= 2**32 - 1:
        raise ValueError("The seed must be an integer from 0 to 4294967295.")
    if result.get("method") not in ("1", "2"):
        raise ValueError("Select Leapfrog or RK4.")
    if result.get("initial_mode") not in ("random", "manual", "orbit"):
        raise ValueError("Select an initial-condition source.")
    if result["initial_mode"] == "orbit":
        index = result.get("orbit_index")
        if not isinstance(index, int) or not 0 <= index < len(STABLE_ORBITS):
            raise ValueError("Select a known orbit.")
    result["adaptive_enabled"] = bool(result.get("adaptive_enabled", False))
    try:
        result["adaptive_eta"] = float(result.get("adaptive_eta", 0.03))
    except (ValueError, TypeError):
        raise ValueError("The safety factor eta must be a number.") from None
    if not np.isfinite(result["adaptive_eta"]) or not 0 < result["adaptive_eta"] <= 0.2:
        raise ValueError("The safety factor eta must satisfy 0 < eta <= 0.2.")
    try:
        result["acceleration_trigger_ratio"] = float(result.get("acceleration_trigger_ratio", 2.0))
    except (ValueError, TypeError):
        raise ValueError("The acceleration trigger ratio must be a number.") from None
    if (not np.isfinite(result["acceleration_trigger_ratio"])
            or result["acceleration_trigger_ratio"] <= 1):
        raise ValueError("The acceleration trigger ratio must be finite and greater than 1.")
    try:
        power = float(result.get("acceleration_max_power", 3))
        if not np.isfinite(power) or power < 1 or power != int(power):
            raise ValueError
        result["acceleration_max_power"] = int(power)
        divisor = result["acceleration_trigger_ratio"] ** result["acceleration_max_power"]
        floor_dt = result["dt"] / divisor
        if not np.isfinite(divisor) or not np.isfinite(floor_dt) or floor_dt <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError("The maximum cumulative exponent must be a positive integer, and initial dt / N^exponent must be representable as a finite positive number.") from None
    interval = result["dt"] * result["sample_every"]
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("The sampling interval is too large or too small. Adjust it.")
    return result


def make_initial_state(config):
    if config["initial_mode"] == "orbit":
        orbit = STABLE_ORBITS[config["orbit_index"]]
        masses = np.asarray(orbit.get("masses", np.ones(orbit["N"])), dtype=float)
        positions = np.asarray(orbit["positions"], dtype=float)
        velocities = np.asarray(orbit["velocities"], dtype=float)
        if (masses.shape != (orbit["N"],) or positions.shape != (orbit["N"], 3)
                or velocities.shape != positions.shape or not np.all(masses > 0)
                or not all(np.isfinite(a).all() for a in (masses, positions, velocities))):
            raise ValueError("Orbit masses, positions and velocities must have matching sizes and finite values; masses must be positive.")
        return [{"mass": float(mass), "pos": pos.copy(), "v": vel.copy()}
                for mass, pos, vel in zip(masses, positions, velocities)]
    # RandomState keeps the original seed -> initial conditions mapping,
    # without mutating NumPy's global random generator during repeat runs.
    rng = np.random.RandomState(config["seed"])
    return [{"mass": 1, "pos": rng.uniform(-50, 50, size=3),
             "v": rng.uniform(-0.1, 0.1, size=3)} for _ in range(config.get("particle_count", 5))]


def run_simulation(config, progress=None, cancel=None):
    """Run one numerical simulation with callbacks, without accessing Tk widgets."""
    config = validate_config(config)
    stars = make_initial_state(config)
    initial_stars = copy.deepcopy(stars)
    base_dt, end_time = config["dt"], config["total_time"]
    adaptive = config["adaptive_enabled"]
    eta = config["adaptive_eta"]
    adaptive_algorithm = config["adaptive_algorithm"]
    sample_every = config["sample_every"]
    nominal_steps = int(np.ceil(end_time / base_dt))
    if cancel is not None and cancel.is_set():
        raise SimulationCancelled()
    gpu = (GPUState(stars, adaptive and adaptive_algorithm == "timescale")
           if config["compute_backend"] == "gpu" else None)
    energy_scale = (gpu.energy_scale if gpu is not None else
                    sum(Ek(s) for s in stars) - sum(
                        Ep(stars[i], stars[j]) for i in range(len(stars)) for j in range(i + 1, len(stars))))
    trajectories = [[s["pos"].copy()] for s in stars]
    velocities = [np.array([s["v"].copy() for s in stars])]
    accelerations = [gpu.host_acceleration if gpu is not None else net_accelerations(stars)]
    diagnostics = [gpu.diagnostics() if gpu is not None else test(stars)]
    energies = [diagnostics[0]["energy"]]
    times, step_sizes, diagnostic_times = [0.0], [0.0], [0.0]
    minimum_dt = max(base_dt * 1e-10, np.nextafter(0.0, 1.0))
    current_time, steps = 0.0, 0
    sample_number = 1
    next_sample_time = min(sample_number * sample_every * base_dt, end_time)
    previous_proposal = None
    ratio_mode = adaptive and adaptive_algorithm == "acceleration_ratio"
    ratio_shrink_count = ratio_grow_count = 0
    ratio_level = ratio_max_level_used = 0
    ratio_floor_dt = base_dt / config["acceleration_trigger_ratio"] ** config["acceleration_max_power"]
    ratio_controller = (AccelerationRatioController(
        accelerations[0], base_dt, config["acceleration_trigger_ratio"], config["acceleration_max_power"])
        if ratio_mode else None)
    ratio_events = []
    if ratio_mode:
        minimum_dt = ratio_floor_dt
    integrator = leapfrog if config["method"] == "1" else rk4
    def advance_state(h):
        if gpu is not None:
            return gpu.advance(h, config["method"])
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            acceleration = (integrator(stars, h, initial_acceleration=accelerations[-1])
                            if ratio_mode else integrator(stars, h))
            energy = tot_E(stars)
        if (not np.isfinite(energy) or not np.isfinite(acceleration).all()
                or any(not np.isfinite(s["pos"]).all() or not np.isfinite(s["v"]).all() for s in stars)):
            raise RuntimeError("The simulation produced non-finite values. Reduce dt or check close encounters.")
        return acceleration, energy

    started = last_report = time.perf_counter()
    if progress:
        progress(0.0, 0, 0.0, 0.0)
    while current_time < end_time:
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()
        if adaptive:
            if adaptive_algorithm == "acceleration_ratio":
                proposal, ratio_level = ratio_controller.choose(accelerations[-1], current_time)
                if ratio_controller.last_event is not None:
                    ratio_events.append(ratio_controller.last_event)
                ratio_max_level_used = max(ratio_max_level_used, ratio_level)
                if previous_proposal is not None:
                    ratio_shrink_count += proposal < previous_proposal
                    ratio_grow_count += proposal > previous_proposal
            else:
                proposal = (gpu.propose_step(base_dt, eta, previous_proposal) if gpu is not None
                            else adaptive_step_size(stars, base_dt, eta, previous_proposal))
            if not np.isfinite(proposal) or proposal < minimum_dt:
                raise RuntimeError(
                    f"t={current_time:.12g} requires dt {proposal:.6e} "
                    f"below the safety limit {minimum_dt:.6e}; this run has been stopped.")
            previous_proposal = proposal
            # Sample ratio-mode diagnostics by actual step count without changing dt; only clip the final step.
            boundary = end_time if ratio_mode else next_sample_time
            remaining = boundary - current_time
            roundoff = 8 * np.finfo(float).eps * max(boundary, base_dt)
            if proposal >= remaining - roundoff:
                h, new_time = remaining, boundary
            else:
                h, new_time = proposal, current_time + proposal
        else:
            # Full fixed steps retain the original h exactly; only the final
            # partial step is shortened to stop at the requested physical time.
            new_time = min((steps + 1) * base_dt, end_time)
            h = base_dt if (steps + 1) * base_dt <= end_time else end_time - current_time
        if h <= 0 or new_time <= current_time:
            raise RuntimeError("The time step is smaller than the floating-point resolution at the current time. Adjust the parameters.")
        new_acceleration, energy = advance_state(h)
        current_time = new_time
        steps += 1
        if gpu is not None:
            for i in range(len(stars)):
                trajectories[i].append(gpu.host_pos[i].copy())
            velocities.append(gpu.host_vel)
        else:
            for i, star in enumerate(stars):
                trajectories[i].append(star["pos"].copy())
            velocities.append(np.array([s["v"].copy() for s in stars]))
        accelerations.append(new_acceleration)
        energies.append(energy)
        times.append(current_time)
        step_sizes.append(h)
        should_sample = ((steps % sample_every == 0 or current_time == end_time)
                         if ratio_mode else current_time == next_sample_time)
        if should_sample:
            diagnostics.append(gpu.diagnostics() if gpu is not None else test(stars))
            diagnostic_times.append(current_time)
            if not ratio_mode:
                sample_number += 1
                next_sample_time = min(sample_number * sample_every * base_dt, end_time)
        now = time.perf_counter()
        if progress and (now - last_report >= .15 or current_time == end_time):
            progress(current_time / end_time, steps, h, now - started)
            last_report = now
    if cancel is not None and cancel.is_set():
        raise SimulationCancelled()
    trajectories = np.asarray(trajectories)
    orbit = STABLE_ORBITS[config["orbit_index"]] if config["initial_mode"] == "orbit" else None
    metadata = {
        "case_name": orbit["name"] if orbit else f"Random initial conditions · seed = {config['seed']}",
        "case_id": f"orbit_{config['orbit_index'] + 1}" if orbit else f"seed_{config['seed']}",
        "seed": orbit["seed"] if orbit else config["seed"],
        "orbit_source": orbit.get("source") if orbit else None,
        "reference_period": orbit.get("period") if orbit else None,
        "method_id": config["method"],
        "method_name": {"1": "Leapfrog (KDK)", "2": "Runge-Kutta 4th order"}[config["method"]],
        "compute_backend": config["compute_backend"],
        "device_name": gpu.device_name if gpu is not None else "CPU",
        "dtype": "float64", "particle_count": len(stars),
        "G": G, "dt": base_dt, "steps": steps, "requested_steps": nominal_steps,
        "simulation_end_time": end_time,
        "adaptive_enabled": adaptive,
        "adaptive_algorithm": adaptive_algorithm if adaptive else "fixed",
        "adaptive_eta": eta if adaptive and adaptive_algorithm == "timescale" else None,
        "zero_acceleration_policy": "current zero: skip; initial zero: use system initial nonzero acceleration scale; all-zero start: initialize once without changing dt"
                                    if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "acceleration_trigger_ratio": config["acceleration_trigger_ratio"]
                                      if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "acceleration_max_power": config["acceleration_max_power"] if ratio_mode else None,
        "ratio_limit_scope": "cumulative relative to initial dt" if ratio_mode else None,
        "ratio_minimum_dt": ratio_floor_dt if ratio_mode else None,
        "ratio_final_level": ratio_level if ratio_mode else None,
        "ratio_max_level_used": ratio_max_level_used if ratio_mode else None,
        "ratio_boundary_policy": "r>N to trigger; higher exact powers enter their own band" if ratio_mode else None,
        "ratio_reference_scales": ratio_controller.scales.tolist() if ratio_mode else None,
        "ratio_star_final_levels": ratio_controller.levels.tolist() if ratio_mode else None,
        "ratio_step_events": ratio_events,
        "ratio_adjustment_policy": "persistent per-star references=initial_scale*N**local_level; integer power bands; independent recovery; global maximum level capped at K"
                                   if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "discarded_trial_steps": 0,
        "ratio_shrink_count": int(ratio_shrink_count),
        "ratio_grow_count": int(ratio_grow_count),
        "ratio_step_reference": "current known acceleration versus persistent per-star reference; reference changes only with local level; no trials"
                                if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "dt_min_used": float(min(step_sizes[1:])), "dt_max_used": float(max(step_sizes[1:])),
        "minimum_allowed_dt": minimum_dt,
        "timestep_criterion": (
            "persistent-reference acceleration power bands; cumulative exponent limit; global minimum"
            if adaptive and adaptive_algorithm == "acceleration_ratio"
            else "pairwise gravity/flyby timescales" if adaptive else "fixed"),
        "energy_scale": float(energy_scale),
        "initial_view_extent": max(2 * float(np.max(np.abs(trajectories))), 1e-6) if orbit else 100,
        "initial_stars": [{"mass": s["mass"], "pos": s["pos"].tolist(), "v": s["v"].tolist()}
                          for s in initial_stars],
        "units": "Original program code units; no SI conversion specified",
        "positions_axes": ["star", "step", "xyz"],
        "accelerations_axes": ["step", "star", "xyz"],
        "velocities_axes": ["step", "star", "xyz"],
        "diagnostic_sampling": sample_every,
        "diagnostic_interval": None if ratio_mode else sample_every * base_dt,
        "diagnostic_sampling_policy": "every N accepted steps, plus final state; no integration clipping"
                                      if ratio_mode else "fixed physical-time intervals",
        "config": config, "elapsed_seconds": time.perf_counter() - started,
    }
    return dict(trajectories=trajectories, accelerations=np.asarray(accelerations),
                velocities=np.asarray(velocities), energies=np.asarray(energies), time_step=base_dt,
                diagnostic_times=np.asarray(diagnostic_times), diagnostics=diagnostics,
                metadata=metadata, record_times=np.asarray(times), step_sizes=np.asarray(step_sizes))



def plot_sample_indices(values, max_points=4000):
    """Keep endpoints and each display bucket's extrema; recordings stay intact."""
    count = len(values)
    if count <= max_points:
        return np.arange(count)
    width = int(np.ceil((count - 2) / ((max_points - 2) // 2)))
    keep = [0, count - 1]
    for start in range(1, count - 1, width):
        chunk = values[start:min(start + width, count - 1)]
        keep.extend((start + int(np.argmin(chunk)), start + int(np.argmax(chunk))))
    return np.unique(keep)


def full_time_statistics(values, times):
    """Return extrema over all recorded samples and the trapezoidal time average.

    Plot decimation and zoom do not affect these statistics. Time weighting
    avoids over-weighting densely sampled intervals.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    duration = times[-1] - times[0]
    mean = (float(np.trapezoid(values, times) / duration)
            if len(times) > 1 and duration > 0 else float(values[0]))
    return {"max": float(np.max(values)), "min": float(np.min(values)), "mean": mean}


def statistics_label(stats, quantity):
    return (f"Max: {stats['max']:.5e}\nMin: {stats['min']:.5e}"
            f"\nTime mean: {stats['mean']:.5e}")


class SharedTimeline(ttk.Frame):
    """One control for computation progress, playback time, and GIF export progress."""
    def __init__(self, parent):
        super().__init__(parent, padding=(12, 2, 12, 5))
        self.columnconfigure(1, weight=1)
        self.viewer = None
        self.mode = "idle"
        self.updating = False
        self.mode_var = tk.StringVar(value="Progress")
        self.value = tk.DoubleVar(value=0)
        self.message = tk.StringVar(value="Choose parameters and click Start / Rerun.")
        ttk.Label(self, textvariable=self.mode_var, width=13).grid(row=0, column=0, sticky="w")
        self.scale = ttk.Scale(self, from_=0, to=100, variable=self.value,
                               orient="horizontal", command=self.seek, state="disabled")
        self.scale.grid(row=0, column=1, sticky="ew")
        ttk.Label(self, textvariable=self.message, width=1, style="Note.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="ew")
        self.scale.bind("<ButtonPress-1>", self.press)
        self.scale.bind("<ButtonRelease-1>", self.release)

    def set_value(self, value):
        self.updating = True
        try:
            self.value.set(value)
        finally:
            self.updating = False

    def begin_computation(self):
        self.mode = "computing"
        self.mode_var.set("Computing")
        self.scale.configure(to=100, state="disabled")
        self.set_value(0)
        if self.viewer is not None:
            self.viewer.save_animation_button.configure(state="disabled")

    def attach(self, viewer):
        self.viewer = viewer
        if viewer is None:
            self.mode = "idle"
            self.mode_var.set("Progress")
            self.scale.configure(to=100, state="disabled")
            self.set_value(0)
            return
        self.mode = "playback"
        self.mode_var.set("Playback time")
        self.scale.configure(to=float(viewer.times[-1]), state="normal")
        viewer.save_animation_button.configure(state="normal")
        self.update_frame(viewer)

    def detach(self, viewer):
        if self.viewer is viewer:
            self.viewer = None
            if self.mode == "playback":
                self.attach(None)

    def update_frame(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.set_value(float(viewer.times[viewer.step]))
            self.message.set(viewer.time_var.get())

    def show_status(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.message.set(viewer.status_var.get())

    def begin_export(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.mode = "export"
            self.mode_var.set("Saving GIF")
            self.scale.configure(to=100, state="disabled")
            self.set_value(0)

    def update_export(self, viewer, current, total):
        if self.viewer is viewer and self.mode == "export":
            self.set_value(100 * (current + 1) / total)
            self.message.set(viewer.status_var.get())

    def finish_export(self, viewer):
        if self.viewer is viewer and self.mode == "export":
            self.attach(viewer)
            self.show_status(viewer)

    def seek(self, value):
        if not self.updating and self.mode == "playback" and self.viewer is not None:
            self.viewer.on_seek(value)

    def press(self, event):
        if self.mode == "playback" and self.viewer is not None:
            self.viewer.on_slider_press(event)

    def release(self, event):
        if self.viewer is not None:
            self.viewer.on_slider_release(event)


class NBodyViewer:
    def __init__(self, trajectories, accelerations, energies, time_step,
                 diagnostic_times, diagnostics, metadata, root=None,
                 record_times=None, step_sizes=None, velocities=None, timeline=None):
        # trajectories[star, step, xyz]; accelerations[step, star, xyz].
        self.trajectories = np.asarray(trajectories)
        self.accelerations = np.asarray(accelerations)
        self.energies = np.asarray(energies)
        self.velocities = np.asarray(velocities, dtype=float)
        self.dt = time_step
        self.n, self.frame_count, _ = self.trajectories.shape
        self.last_step = self.frame_count - 1
        self.times = (np.arange(self.frame_count) * self.dt if record_times is None
                      else np.asarray(record_times, dtype=float))
        self.step_sizes = (np.r_[0.0, np.diff(self.times)] if step_sizes is None
                           else np.asarray(step_sizes, dtype=float))
        self.diagnostic_times = np.asarray(diagnostic_times)
        self.diagnostics = diagnostics
        self.metadata = metadata
        self.adaptive = metadata.get("adaptive_enabled", False)
        if self.adaptive and record_times is None:
            raise ValueError("Adaptive mode requires actual cumulative times, not step count multiplied by a fixed dt.")
        if not self.adaptive:
            self.mode_description = f"Fixed dt = {self.dt:g}"
        elif metadata.get("adaptive_algorithm", "timescale") == "acceleration_ratio":
            self.mode_description = (f"Acceleration ratio (retained reference) · N = {metadata.get('acceleration_trigger_ratio', 2):.1f}"
                                     f" · max exponent K = {metadata.get('acceleration_max_power', 3)}"
                                     f" · dt max = {self.dt:g}")
        else:
            self.mode_description = f"Time-scale method · dt max = {self.dt:g} · eta = {metadata['adaptive_eta']:g}"
        self.mode_description += f" · {metadata.get('compute_backend', 'cpu').upper()} · float64"
        if (self.times.shape != (self.frame_count,) or not np.isfinite(self.times).all()
                or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Cumulative times must start at zero, increase strictly, and match the trajectory records.")
        if (self.step_sizes.shape != (self.frame_count,) or not np.isfinite(self.step_sizes).all()
                or self.step_sizes[0] != 0 or np.any(self.step_sizes[1:] <= 0)):
            raise ValueError("Actual step sizes must be positive and match trajectory records; zero marks the initial state.")
        if self.accelerations.shape != (self.frame_count, self.n, 3):
            raise ValueError("Acceleration records must match every star and trajectory time.")
        if self.velocities.shape != (self.frame_count, self.n, 3):
            raise ValueError("Velocity records must match every star and trajectory time.")
        if self.energies.shape != (self.frame_count,):
            raise ValueError("Energy records must match trajectory times.")

        self.root = root if root is not None else tk.Tk()
        width = min(1600, self.root.winfo_screenwidth() - 60)
        if isinstance(self.root, tk.Tk):
            self.root.title("N-body simulation · Trajectories, energy and net acceleration")
            height = min(1000, self.root.winfo_screenheight() - 100)
            self.root.geometry(f"{width}x{height}")
            self.root.minsize(min(1000, width), min(700, height))
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        # Embedded viewers use the application's shared timeline; standalone viewers own one.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.closed = False
        self.step = 0
        self.star_index = 0
        self.fit_particles_active = False
        self.follow_view_active = False
        self.fixed_view_limits = None
        self.playing = False
        self.dragging = False
        self.updating_slider = False
        self.after_id = None
        self.mouse_press = None
        self.drag_axis = None
        self.hover_selection = None
        self.hover_items = {}
        self.page_readouts = {}
        self.page_readout_widgets = {}
        self.page_viewports = {}
        self.plot_headers = {}
        self.hover_idle_text = "Hover over a plot to inspect its values."
        self.selected_magnitudes = None

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Note.TLabel", foreground="#52606d", font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", foreground="#235b8c")

        header = ttk.Frame(self.root, padding=(12, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        title_label = ttk.Label(header, text="N-body simulation: trajectories, energy and acceleration",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        self.save_animation_button = ttk.Button(
            header, text="Save animation (GIF)", command=self.save_animation)
        self.save_animation_button.grid(row=0, column=1, sticky="ne", padx=(14, 0))
        description_label = ttk.Label(
            header, text=f"{metadata['case_name']}  |  {metadata['method_name']}  |  "
            f"N = {self.n}, {self.mode_description}, {self.last_step:,} steps",
            style="Note.TLabel", wraplength=width - 55)
        description_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        def resize_header(event):
            title_label.configure(wraplength=max(
                240, event.width - self.save_animation_button.winfo_reqwidth() - 50))
            description_label.configure(wraplength=max(400, event.width - 36))
        header.bind("<Configure>", resize_header)
        body = ttk.Frame(self.root, padding=(16, 0, 12, 8))
        body.grid(row=1, column=0, sticky="nsew")
        controls = self.controls = ttk.Frame(body, width=270)
        controls.pack(side="left", fill="y", padx=(0, 12))
        controls.pack_propagate(False)
        scroll = tk.Canvas(controls, width=270, highlightthickness=0,
                           background=style.lookup("TFrame", "background"))
        scrollbar = ttk.Scrollbar(controls, orient="vertical", command=scroll.yview)
        scrollbar.pack(side="right", fill="y")
        scroll.pack(side="left", fill="both", expand=True)
        scroll.configure(yscrollcommand=scrollbar.set)
        sidebar = ttk.Frame(scroll, padding=(0, 0, 8, 8))
        sidebar_window = scroll.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(sidebar_window, width=event.width))

        self.play_button = ttk.Button(sidebar, text="Play", command=self.toggle_play)
        self.play_button.pack(fill="x", pady=(4, 3))
        stepping = ttk.Frame(sidebar)
        stepping.pack(fill="x", pady=3)
        ttk.Button(stepping, text="Previous step", command=lambda: self.single_step(-1)).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(stepping, text="Next step", command=lambda: self.single_step(1)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(sidebar, text="Return to start", command=self.rewind).pack(fill="x", pady=3)
        self.stride_var = tk.StringVar(value=str(max(1, self.last_step // 1500)))
        ttk.Label(sidebar, text="Recorded steps per playback frame").pack(anchor="w", pady=(7, 2))
        ttk.Spinbox(sidebar, from_=1, to=max(1, self.last_step),
                    textvariable=self.stride_var).pack(fill="x")
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="Loop playback", variable=self.loop_var).pack(anchor="w", pady=4)

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Label(sidebar, text="Select a star (or click an orbit marker)").pack(anchor="w")
        self.star_var = tk.StringVar(value="Star 1")
        self.star_combo = ttk.Combobox(sidebar, textvariable=self.star_var, state="readonly",
                                      values=[f"Star {i + 1}" for i in range(self.n)])
        self.star_combo.pack(fill="x", pady=(4, 6))
        self.star_combo.bind("<<ComboboxSelected>>",
                             lambda event: self.select_star(self.star_combo.current()))
        ttk.Label(sidebar, text="Camera mode").pack(anchor="w", pady=(4, 2))
        self.camera_mode_var = tk.StringVar(value="Fixed view")
        self.camera_combo = ttk.Combobox(
            sidebar, textvariable=self.camera_mode_var, state="readonly",
            values=["Fixed view", "Follow selected star", "Fit all current particles"])
        self.camera_combo.pack(fill="x", pady=(0, 4))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_mode_change)
        ttk.Label(sidebar, text="Follow centers the selected star. Fit all current particles adjusts the bounds at each time; rotation remains available.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=(0, 5))
        self.details_var = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.details_var, justify="left",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=4)
        self.components_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sidebar, text="Acceleration: show x, y, z components",
                        variable=self.components_var, command=self.refresh_vectors).pack(
                            anchor="w", pady=(6, 3))
        self.delta_energy_var = tk.BooleanVar(value=False)
        ttk.Button(sidebar, text="Fit all trajectories", command=self.fit_orbits).pack(fill="x", pady=3)
        ttk.Label(sidebar,
                  text="Click a star to pause. Drag the playback slider to inspect another time.\n"
                       "Click or drag the yellow time cursor; hover to read values.\n"
                       "Time, position, energy, acceleration and velocity use code units.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=(6, 3))

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Button(sidebar, text="Save current page (PNG)", command=self.save_image).pack(fill="x", pady=3)
        ttk.Button(sidebar, text="Save all-star data (NPZ)", command=self.save_data).pack(fill="x", pady=3)
        ttk.Label(sidebar, text="Save on demand. Long series use fewer plotted points; hover and data export use complete records.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=5)

        display = self.display = ttk.Frame(body)
        display.pack(side="left", fill="both", expand=True)
        energy_controls = ttk.Frame(display)
        energy_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(energy_controls, text="Energy display:").pack(side="left", padx=(0, 8))
        self.energy_value_radio = ttk.Radiobutton(
            energy_controls, text="Total energy E(t)", variable=self.delta_energy_var,
            value=False, command=self.refresh_energy)
        self.energy_value_radio.pack(side="left", padx=(0, 16))
        self.energy_delta_radio = ttk.Radiobutton(
            energy_controls, text="Offset from initial value: ΔE = E(t) − E(0)",
            variable=self.delta_energy_var, value=True, command=self.refresh_energy)
        self.energy_delta_radio.pack(side="left")
        self.notebook = ttk.Notebook(display)
        self.notebook.pack(fill="both", expand=True)
        self.overview_tab = ttk.Frame(self.notebook)
        self.diagnostics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Trajectory · Energy · Acceleration")
        self.notebook.add(self.diagnostics_tab, text="Conservation & Errors")
        self.build_overview()
        self.build_diagnostics()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.owns_timeline = timeline is None
        self.timeline = SharedTimeline(self.root) if timeline is None else timeline
        if self.owns_timeline:
            self.timeline.grid(row=2, column=0, sticky="ew")
        self.slider = self.timeline.scale
        self.time_var = tk.StringVar(master=self.root)
        self.status_var = tk.StringVar(master=self.root, value="Computation complete.")
        self.status_trace = self.status_var.trace_add(
            "write", lambda *args: self.timeline.show_status(self))
        self.refresh_energy()
        self.refresh_vectors()
        self.draw_frame(0)
        if self.owns_timeline:
            self.timeline.attach(self)
        # Initialize the 3D projection before constrained layout measures its labels.
        self.orbit_axes.draw(self.canvas.get_renderer())
        self.canvas.draw()
        self.diagnostic_canvas.draw()
        self.after_id = self.root.after(40, self.tick)


    def embed_figure(self, figure, parent):
        # The plot uses the available viewport exactly; no oversized scrollable canvas.
        variable = tk.StringVar(value=self.hover_idle_text)
        self.page_readouts[figure] = variable
        area = ttk.Frame(parent)
        area.pack(side="top", fill="both", expand=True)
        area.rowconfigure(0, weight=1)
        area.columnconfigure(0, weight=1)
        viewport = tk.Canvas(area, highlightthickness=0, background="white")
        viewport.grid(row=0, column=0, sticky="nsew")
        canvas = FigureCanvasTkAgg(figure, master=viewport)
        widget = canvas.get_tk_widget()
        window = viewport.create_window(0, 0, window=widget, anchor="nw")
        self.page_viewports[figure] = viewport

        def resize(event):
            # Hidden notebook pages briefly report 1 px; retain their last usable layout.
            if event.width <= 1 or event.height <= 1:
                return
            self.apply_figure_layout(figure, event.width, event.height)
            viewport.itemconfigure(window, width=event.width, height=event.height)
            viewport.configure(scrollregion=(0, 0, event.width, event.height))
            self.update_page_readout(figure)
        viewport.bind("<Configure>", resize)
        inspector = tk.Label(viewport, textvariable=variable, font=("Consolas", 9),
                             anchor="ne", justify="right", padx=7, pady=5,
                             background="#f3f6fa", foreground="#172435",
                             borderwidth=1, relief="solid")
        self.page_readout_widgets[figure] = inspector
        variable.trace_add("write", lambda *args: self.update_page_readout(figure))
        inspector.bind("<Leave>", lambda event: self.hide_page_readout(figure))
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x", before=area)
        canvas.mpl_connect("motion_notify_event", self.on_plot_motion)
        canvas.mpl_connect("figure_leave_event", self.clear_hover)
        canvas.mpl_connect("button_press_event", self.on_plot_press)
        canvas.mpl_connect("button_release_event", self.on_plot_release)
        return canvas, toolbar

    def apply_figure_layout(self, figure, width, height, for_export=False):
        """Use compact screen text while keeping the original font sizes for export."""
        diagnostic = figure is getattr(self, "diagnostic_figure", None)
        base_width, base_height = (1220, 780) if diagnostic else (1000, 650)
        scale = (1.0 if for_export else
                 max(.82, min(1.0, width / base_width, height / base_height)))
        text_items = list(figure.texts)
        for axis in figure.axes:
            text_items.extend(axis.texts)
            text_items.append(axis.title)
            for axis_name in ("xaxis", "yaxis", "zaxis"):
                coordinate = getattr(axis, axis_name, None)
                if coordinate is None:
                    continue
                text_items.append(coordinate.label)
                if not hasattr(coordinate, "_nbody_full_tick_size"):
                    labels = coordinate.get_ticklabels()
                    coordinate._nbody_full_tick_size = labels[0].get_fontsize() if labels else 8.0
                size = coordinate._nbody_full_tick_size * scale
                coordinate.set_tick_params(labelsize=max(6.2, size))
                coordinate.offsetText.set_fontsize(max(6.2, size))
            legend = axis.get_legend()
            if legend is not None:
                text_items.extend(legend.get_texts())
                text_items.append(legend.get_title())
        for item in text_items:
            if not hasattr(item, "_nbody_full_fontsize"):
                item._nbody_full_fontsize = item.get_fontsize()
            item.set_fontsize(max(6.2, item._nbody_full_fontsize * scale))
        engine = figure.get_layout_engine()
        if engine is not None:
            engine.set(w_pad=.035 * scale, h_pad=.035 * scale, wspace=.035, hspace=.045)

    def build_overview(self):
        self.figure = Figure(figsize=(12, 7.5), dpi=100, facecolor="white", constrained_layout=True)
        if not self.adaptive:
            compact_mode = f"dt={self.dt:g}"
        elif self.metadata["adaptive_algorithm"] == "acceleration_ratio":
            compact_mode = (f"ratio N={self.metadata['acceleration_trigger_ratio']:.1f}, "
                            f"K={self.metadata['acceleration_max_power']}; dt_max={self.dt:g}")
        else:
            compact_mode = f"eta={self.metadata['adaptive_eta']:g}; dt_max={self.dt:g}"
        self.figure_caption = (f"{self.metadata['case_name']}\n"
                               f"{self.metadata['method_name']} | N={self.n} | {compact_mode} | "
                               f"{self.metadata.get('compute_backend', 'cpu').upper()} | steps={self.last_step:,}")
        self.figure.suptitle(self.figure_caption, fontsize=9)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1.0, 1.45))
        self.orbit_axes = self.figure.add_subplot(grid[:, 0], projection="3d")
        energy_grid = grid[0, 1].subgridspec(2, 1, height_ratios=(.55, 1), hspace=.02)
        acceleration_grid = grid[1, 1].subgridspec(2, 1, height_ratios=(.55, 1), hspace=.02)
        self.energy_info_axes = self.figure.add_subplot(energy_grid[0])
        self.acceleration_info_axes = self.figure.add_subplot(acceleration_grid[0])
        self.energy_axes = self.figure.add_subplot(energy_grid[1])
        self.acceleration_axes = self.figure.add_subplot(acceleration_grid[1], sharex=self.energy_axes)
        for axis, info in ((self.energy_axes, self.energy_info_axes),
                           (self.acceleration_axes, self.acceleration_info_axes)):
            info.set_axis_off()
            self.plot_headers[axis] = info
        self.energy_title_text = self.energy_info_axes.text(0, .98, "Energy", va="top", fontsize=10)
        self.acceleration_title_text = self.acceleration_info_axes.text(0, .98, "Net acceleration", va="top", fontsize=10)
        self.orbit_axes.set(xlabel="x", ylabel="y", zlabel="z")
        self.orbit_axes.set_title("Trajectories", fontsize=10)
        self.orbit_axes.set_box_aspect((1, 1, 1))
        self.orbit_axes.set_facecolor("#f6f8fb")
        colors = plt.get_cmap("tab10")
        self.lines = []
        self.points = []
        for i in range(self.n):
            color = colors(i % 10)
            self.lines.append(self.orbit_axes.plot([], [], [], color=color, linewidth=1)[0])
            self.points.append(self.orbit_axes.plot([], [], [], "o", color=color,
                                                    markersize=6, label=f"Star {i + 1}")[0])
        self.highlight = self.orbit_axes.plot([], [], [], "o", markersize=12,
                                              markerfacecolor="none", markeredgecolor="#172435",
                                              markeredgewidth=1.8)[0]
        self.time_text = self.orbit_axes.text2D(.98, .98, "", transform=self.orbit_axes.transAxes,
                                               ha="right", va="top", fontsize=10)
        extent = self.metadata["initial_view_extent"]
        for setter in (self.orbit_axes.set_xlim, self.orbit_axes.set_ylim, self.orbit_axes.set_zlim):
            setter(-extent, extent)
        self.energy_line, = self.energy_axes.plot([], [], color="#172435", linewidth=1.3, label="E")
        self.energy_reference = self.energy_axes.axhline(self.energies[0], color="#8a94a3",
                                                         linestyle="--", linewidth=.8, label="Initial value")
        self.acceleration_lines = {}
        for key, color, label in (("magnitude", "#7c3aed", "|a|"),
                                   ("x", "#d62728", "a_x"), ("y", "#2ca02c", "a_y"),
                                   ("z", "#1f77b4", "a_z")):
            line, = self.acceleration_axes.plot([], [], color=color,
                                                linewidth=1.4 if key == "magnitude" else .8, label=label)
            self.acceleration_lines[key] = line
        self.energy_cursor = self.energy_axes.axvline(0, color="#de8a28", linewidth=1, alpha=.7)
        self.acceleration_cursor = self.acceleration_axes.axvline(0, color="#de8a28", linewidth=1, alpha=.7)
        for axis in (self.energy_axes, self.acceleration_axes):
            axis.set_xlabel("Time (code units)", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=.22)
            axis.set_xlim(0, self.times[-1])
        self.build_step_size_axes()
        # Place scientific-notation and offset labels at the right end of each y axis.
        self.energy_axes.yaxis.set_offset_position("right")
        self.acceleration_axes.yaxis.set_offset_position("right")
        self.acceleration_axes.set_ylabel("Net acceleration (code units)", fontsize=9)
        self.energy_statistics_text = self.energy_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)
        self.acceleration_statistics_text = self.acceleration_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)
        self.update_overview_legend(self.energy_axes)
        self.register_hover(self.energy_axes, self.times, {"E": self.energies})
        self.register_hover(self.acceleration_axes, self.times, {})
        self.canvas, self.toolbar = self.embed_figure(self.figure, self.overview_tab)

    def step_plot_data(self):
        """Map h[n] to (t[n-1], t[n]]; omit the initial zero marker from the log axis."""
        if self.adaptive and self.metadata.get("adaptive_algorithm") == "acceleration_ratio":
            # Keep every dt change and its preceding point; steps-pre shows the correct active interval.
            changes = np.flatnonzero(np.diff(self.step_sizes[1:]) != 0) + 1
            indices = np.unique(np.r_[0, len(self.step_sizes) - 2, changes - 1, changes]) + 1
        else:
            indices = plot_sample_indices(self.step_sizes[1:]) + 1
        # Extend the first real step back to t=0. The stored h[0]=0 is only
        # an initial-state marker, not an actual zero-size integration step.
        times = np.r_[self.times[0], self.times[indices]]
        steps = np.r_[self.step_sizes[1], self.step_sizes[indices]]
        return times, steps

    def add_step_size_axis(self, axis, base, show_ratio_levels=False):
        times, steps = self.step_plot_data()
        color = "#b45309"
        low, high = float(np.min(self.step_sizes[1:])), float(np.max(self.step_sizes[1:]))
        padding = max((np.log(high) - np.log(low)) * .08, np.log(1.5))
        y_low = max(np.nextafter(0., 1.), np.exp(max(np.log(np.nextafter(0., 1.)), np.log(low) - padding)))
        y_high = np.exp(min(np.log(np.finfo(float).max), np.log(high) + padding))
        right = axis.twinx()
        line, = right.plot(times, steps, color=color, linestyle="--",
                           linewidth=1.15, alpha=.85, drawstyle="steps-pre",
                           label="Actual dt")
        right.set_yscale("log", base=base)
        right.set_ylabel(f"log DT", color=color, fontsize=9, labelpad=7)
        right.tick_params(axis="y", colors=color, labelsize=8)
        right.spines["right"].set_color(color)
        right.set_ylim(y_low, y_high)
        if show_ratio_levels:
            # Align major ticks with dt0/N**k; show at most six levels even for N near 1 or large K.
            last_level = max(0, int(np.floor(np.log(self.dt / low) / np.log(base))))
            first_level = max(0, int(np.ceil(np.log(self.dt / high) / np.log(base))))
            levels = np.unique(np.rint(np.linspace(first_level, max(first_level, last_level),
                                        min(6, max(1, last_level - first_level + 1)))).astype(int))
            ticks = self.dt / np.power(base, levels.astype(float))
            ticks = ticks[(ticks >= y_low) & (ticks <= y_high)]
            right.yaxis.set_major_locator(FixedLocator(ticks))
        else:
            right.yaxis.set_major_locator(LogLocator(base=base, numticks=6))
        right.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.4g}"))
        right.yaxis.set_minor_locator(NullLocator())
        right.grid(False)
        # Put labels, the yellow cursor and physical curves above dt.
        # Make the upper axis transparent so the secondary line is visible.
        right.set_zorder(axis.get_zorder() - 1)
        axis.patch.set_visible(False)
        self.step_axes[axis] = right
        self.step_lines[axis] = line
        self.time_axis_for[right] = axis

    def build_step_size_axes(self):
        self.step_axes = {}
        self.step_lines = {}
        self.time_axis_for = {}
        self.step_log_base = float(self.metadata.get("acceleration_trigger_ratio") or 10.0)
        ratio_mode = self.adaptive and self.metadata.get("adaptive_algorithm") == "acceleration_ratio"
        for axis in (self.energy_axes, self.acceleration_axes):
            self.add_step_size_axis(axis, self.step_log_base, show_ratio_levels=ratio_mode)

    def update_overview_legend(self, axis, handles=None, ncol=1):
        if handles is None:
            handles, _ = axis.get_legend_handles_labels()
        handles = list(handles) + [self.step_lines[axis]]
        diagnostic = axis is getattr(self, "diagnostic_energy_axes", None)
        self.plot_headers[axis].legend(
            handles=handles, fontsize=7, ncol=min(3, len(handles)),
            loc="center" if diagnostic else "lower left",
            bbox_to_anchor=(.63, .68) if diagnostic else (0, -.05), borderaxespad=0,
            frameon=False, handlelength=1.5, columnspacing=.9)

    def build_diagnostics(self):
        self.diagnostic_figure = Figure(figsize=(15, 10), dpi=100, constrained_layout=True)
        grid = self.diagnostic_figure.add_gridspec(5, 4, height_ratios=(.6, 1.25, 1, 1, 1))
        self.diagnostic_figure.suptitle("Conservation and error analysis\n" + self.figure_caption, fontsize=9)
        self.diagnostic_cursors = []
        self.diagnostic_info_axes = self.diagnostic_figure.add_subplot(grid[0, :])
        self.diagnostic_info_axes.set_axis_off()
        energy_axis = self.diagnostic_figure.add_subplot(grid[1, :])
        self.plot_headers[energy_axis] = self.diagnostic_info_axes
        self.diagnostic_title_text = self.diagnostic_info_axes.text(0, .98, "Energy", va="top", fontsize=10)
        self.diagnostic_error_text = self.diagnostic_info_axes.text(0, .62, "", va="top", fontsize=8)
        self.diagnostic_statistics_text = self.diagnostic_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)

        values = np.array([item["energy"] for item in self.diagnostics])
        self.diagnostic_energy_axes = energy_axis
        self.diagnostic_energy_values = values
        self.diagnostic_energy_line, = energy_axis.plot(
            self.diagnostic_times, values, color="#172435", linewidth=1, label="E")
        self.diagnostic_energy_reference = energy_axis.axhline(
            self.energies[0], color="#8a94a3", linestyle="--", linewidth=.9, label="Initial energy E(0)")
        self.add_step_size_axis(energy_axis, base=10.0)
        self.update_overview_legend(energy_axis, ncol=3)
        error = float(RMSE(values))
        scale = self.metadata["energy_scale"]
        normalized = error / scale if scale else np.nan
        self.diagnostic_energy_error_label = f"RMSE = {error:.5e}    Normalized RMSE = {normalized:.5e}"

        energy_axis.set_ylabel("E")
        energy_axis.grid(alpha=.22)
        self.register_hover(energy_axis, self.diagnostic_times, {"E": values})
        self.diagnostic_cursors.append(energy_axis.axvline(0, color="#de8a28", linewidth=.8))
        vector_info = [
            ("momentum", "Total momentum", "P"),
            ("angular_momentum", "Total angular momentum", "L"),
            ("com_v", "COM velocity", "V_com"),
        ]
        for row, (key, title, symbol) in enumerate(vector_info, start=2):
            values = np.array([item[key] for item in self.diagnostics])
            for col, (suffix, color) in enumerate((("x", "tab:red"), ("y", "tab:green"),
                                                    ("z", "tab:blue"), ("Magnitude", "tab:purple"))):
                series = values[:, col] if col < 3 else np.linalg.norm(values, axis=1)
                axis = self.diagnostic_figure.add_subplot(grid[row, col], sharex=energy_axis)
                axis.plot(self.diagnostic_times, series, color=color, linewidth=1)
                axis.set_title(f"{title}：{suffix}\nRMSE = {float(RMSE(series)):.5e}", fontsize=8)
                # Shared time coordinates need labels only on the bottom row.
                axis.set_xlabel("Time (code units)" if row == 4 else "", fontsize=8)
                axis.tick_params(axis="x", labelbottom=(row == 4))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.set_ylabel(f"{symbol}_{suffix}" if col < 3 else f"|{symbol}|", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.grid(alpha=.22)
                self.register_hover(axis, self.diagnostic_times, {f"{symbol} {suffix}": series})
                self.diagnostic_cursors.append(axis.axvline(0, color="#de8a28", linewidth=.8))
        energy_axis.set_xlabel("")
        energy_axis.tick_params(labelsize=7, axis="both")
        energy_axis.tick_params(axis="x", labelbottom=False)
        energy_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        energy_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self.diagnostic_canvas, self.diagnostic_toolbar = self.embed_figure(
            self.diagnostic_figure, self.diagnostics_tab)

    def register_hover(self, axis, times, series, title=None):
        if axis in getattr(self, "step_axes", {}):
            series = dict(series)
            indices = np.clip(np.searchsorted(self.times, np.asarray(times)), 0, self.frame_count - 1)
            actual_steps = self.step_sizes[indices].copy()
            actual_steps[indices == 0] = np.nan
            series["dt"] = actual_steps
        if title is None:
            title = axis.get_title().split("\n")[0] or "Energy"
        self.hover_items[axis] = dict(times=np.asarray(times), series=series, title=title)
        if axis.figure in self.page_readouts:
            self.page_readouts[axis.figure].set(self.hover_idle_text)
        self.hover_selection = None

    @staticmethod
    def format_readout(title, fields):
        rows = [title]
        rows.extend("   ".join(fields[i:i + 2]) for i in range(0, len(fields), 2))
        return "\n".join(rows)

    def update_page_readout(self, figure):
        inspector = self.page_readout_widgets.get(figure)
        if inspector is None:
            return
        text = self.page_readouts[figure].get()
        if not text or text == self.hover_idle_text:
            inspector.place_forget()
            return
        viewport = self.page_viewports[figure]
        inspector.configure(wraplength=max(180, min(470, viewport.winfo_width() - 18)))
        inspector.place(relx=1.0, x=-7, y=7, anchor="ne")
        inspector.lift()

    def hide_page_readout(self, figure):
        self.page_readouts[figure].set(self.hover_idle_text)
        self.hover_selection = None

    def clear_hover(self, event):
        figure = event.canvas.figure
        inspector = self.page_readout_widgets.get(figure)
        if inspector is not None and inspector.winfo_ismapped():
            # Entering the inspector itself is not leaving the plot page.
            x, y = inspector.winfo_pointerxy()
            if (inspector.winfo_rootx() <= x < inspector.winfo_rootx() + inspector.winfo_width()
                    and inspector.winfo_rooty() <= y < inspector.winfo_rooty() + inspector.winfo_height()):
                return
        variable = self.page_readouts.get(figure)
        if variable is not None:
            variable.set(self.hover_idle_text)
        self.hover_selection = None

    @staticmethod
    def nearest_index(times, value):
        index = min(int(np.searchsorted(times, value)), len(times) - 1)
        if index > 0 and abs(value - times[index - 1]) <= abs(value - times[index]):
            index -= 1
        return index

    def refresh_energy(self):
        offset = self.delta_energy_var.get()
        values = self.energies - self.energies[0] if offset else self.energies
        indices = plot_sample_indices(values)
        self.energy_line.set_data(self.times[indices], values[indices])
        label = "ΔE" if offset else "E"
        self.energy_line.set_label(label)
        reference = 0 if offset else self.energies[0]
        self.energy_reference.set_ydata([reference, reference])
        self.energy_reference.set_label("Initial")
        self.energy_title_text.set_text("Energy offset ΔE" if offset else "Total energy E")
        self.energy_statistics = full_time_statistics(values, self.times)
        self.energy_statistics_text.set_text(statistics_label(self.energy_statistics, label))
        self.energy_axes.set_ylabel(f"{label} (code units)", fontsize=9)
        self.energy_axes.relim(); self.energy_axes.autoscale_view(scalex=False)
        self.update_overview_legend(self.energy_axes)
        self.register_hover(self.energy_axes, self.times, {label: values}, "Energy offset" if offset else "Total energy")
        if hasattr(self, "diagnostic_energy_line"):
            diagnostic_values = self.diagnostic_energy_values - self.energies[0] if offset else self.diagnostic_energy_values
            self.diagnostic_energy_line.set_ydata(diagnostic_values)
            self.diagnostic_energy_line.set_label("Energy offset" if offset else "Energy")
            self.diagnostic_energy_reference.set_ydata([reference, reference])
            self.diagnostic_energy_reference.set_label("Initial offset = 0" if offset else "Initial energy")
            axis = self.diagnostic_energy_axes
            axis.set_ylabel(label)
            self.diagnostic_title_text.set_text("Energy offset ΔE = E(t) - E(0)" if offset else "Total energy E(t)")
            self.diagnostic_error_text.set_text(self.diagnostic_energy_error_label)
            self.diagnostic_statistics_text.set_text(statistics_label(self.energy_statistics, label))
            axis.relim(); axis.autoscale_view(scalex=False)
            self.update_overview_legend(axis, ncol=3)
            self.register_hover(axis, self.diagnostic_times, {label: diagnostic_values},
                                "Energy offset" if offset else "Total energy")
            self.apply_figure_layout(self.diagnostic_figure, *self.diagnostic_figure.bbox.size)
            self.diagnostic_canvas.draw_idle()
        self.apply_figure_layout(self.figure, *self.figure.bbox.size)
        self.canvas.draw_idle()

    def refresh_acceleration(self):
        vectors = self.accelerations[:, self.star_index, :]
        self.selected_magnitudes = np.linalg.norm(vectors, axis=1)
        series = {"|a|": self.selected_magnitudes}
        for col, key in enumerate(("x", "y", "z")):
            values = vectors[:, col]
            indices = plot_sample_indices(values)
            line = self.acceleration_lines[key]
            line.set_data(self.times[indices], values[indices])
            line.set_visible(self.components_var.get())
            if self.components_var.get():
                series[f"a_{key}"] = values
        indices = plot_sample_indices(self.selected_magnitudes)
        self.acceleration_lines["magnitude"].set_data(self.times[indices],
                                                      self.selected_magnitudes[indices])
        self.acceleration_title_text.set_text(f"Star {self.star_index + 1}: |a|")
        self.acceleration_statistics = full_time_statistics(self.selected_magnitudes, self.times)
        self.acceleration_statistics_text.set_text(statistics_label(self.acceleration_statistics, "|a|"))
        self.acceleration_axes.relim(visible_only=True)
        self.acceleration_axes.autoscale_view(scalex=False)
        visible_lines = [line for line in self.acceleration_lines.values() if line.get_visible()]
        self.update_overview_legend(self.acceleration_axes, visible_lines, ncol=3)
        self.register_hover(self.acceleration_axes, self.times, series, f"Star {self.star_index + 1}: net acceleration")
        self.apply_figure_layout(self.figure, *self.figure.bbox.size)
        self.canvas.draw_idle()

    def refresh_vectors(self):
        self.refresh_acceleration()

    def select_star(self, index):
        if not 0 <= index < self.n:
            return
        self.pause()
        self.star_index = int(index)
        self.star_var.set(f"Star {self.star_index + 1}")
        self.refresh_vectors()
        self.draw_frame(self.step)
        follow_note = "The view now follows this star. " if self.follow_view_active else ""
        self.status_var.set(f"Selected Star {self.star_index + 1}. {follow_note}Drag the playback slider to inspect its state at another time.")

    def orbit_limits(self):
        return np.array([self.orbit_axes.get_xlim3d(), self.orbit_axes.get_ylim3d(),
                         self.orbit_axes.get_zlim3d()], dtype=float)

    def set_orbit_limits(self, limits):
        for setter, bounds in zip(
                (self.orbit_axes.set_xlim3d, self.orbit_axes.set_ylim3d, self.orbit_axes.set_zlim3d),
                limits):
            setter(*bounds)

    def on_camera_mode_change(self, event=None):
        mode = self.camera_mode_var.get()
        follow = mode == "Follow selected star"
        fit = mode == "Fit all current particles"
        was_dynamic = self.follow_view_active or self.fit_particles_active
        if (follow or fit) and not was_dynamic:
            self.fixed_view_limits = self.orbit_limits().copy()
        elif not (follow or fit) and was_dynamic and self.fixed_view_limits is not None:
            self.set_orbit_limits(self.fixed_view_limits)
        self.follow_view_active = follow
        self.fit_particles_active = fit
        self.draw_frame(self.step)
        self.status_var.set("Fitting all current particles with a margin at each time."
                            if fit else f"Following Star {self.star_index + 1}." if follow else
                            "Fixed view restored to the bounds used before automatic camera mode.")

    def update_follow_view(self, position):
        if self.fit_particles_active:
            current = self.trajectories[:, self.step, :]
            current = current[np.all(np.isfinite(current), axis=1)]
            if len(current):
                low, high = current.min(axis=0), current.max(axis=0)
                center = low / 2 + high / 2
                # Equal axis spans preserve geometry; padding includes isolated or planar states.
                extent = max(float(np.max(high - low)) * 0.6, 1e-6,
                             float(np.max(np.abs(center))) * 1e-12)
                self.set_orbit_limits(np.column_stack((center - extent, center + extent)))
            self.orbit_axes.set_title("All current particles", fontsize=10)
        elif self.follow_view_active:
            # Translate the viewing volume only. Read the current spans so
            # manual zoom is retained; leave elevation/azimuth/roll unchanged.
            limits = self.orbit_limits()
            half_spans = (limits[:, 1] - limits[:, 0]) / 2
            self.set_orbit_limits(np.column_stack((position - half_spans, position + half_spans)))
            self.orbit_axes.set_title(f"Following Star {self.star_index + 1}", fontsize=10)
        else:
            self.orbit_axes.set_title("Trajectories", fontsize=10)

    def draw_frame(self, step, redraw=True):
        self.step = int(np.clip(step, 0, self.last_step))
        # Cap only the displayed trail, always keeping its current endpoint.
        stride = max(1, int(np.ceil((self.step + 1) / 4000)))
        indices = np.arange(0, self.step + 1, stride)
        if indices[-1] != self.step:
            indices = np.append(indices, self.step)
        for i in range(self.n):
            path = self.trajectories[i, indices]
            self.lines[i].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
            position = self.trajectories[i, self.step]
            self.points[i].set_data_3d([position[0]], [position[1]], [position[2]])
        position = self.trajectories[self.star_index, self.step]
        self.highlight.set_data_3d([position[0]], [position[1]], [position[2]])
        self.update_follow_view(position)
        current_time = self.times[self.step]
        self.selected_step_distance = (float(np.linalg.norm(
            position - self.trajectories[self.star_index, self.step - 1]))
            if self.step > 0 else 0.0)
        distance_text = (f"Step displacement = {self.selected_step_distance:.6g}"
                         if self.step > 0 else "Initial state; no step taken")
        self.time_text.set_text(
            f"t = {current_time:.6g}\n{self.step:,} / {self.last_step:,} steps\n{distance_text}")
        self.energy_cursor.set_xdata([current_time, current_time])
        self.acceleration_cursor.set_xdata([current_time, current_time])
        for line in self.diagnostic_cursors:
            line.set_xdata([current_time, current_time])
        acceleration = self.accelerations[self.step, self.star_index]
        velocity = self.velocities[self.step, self.star_index]
        self.details_var.set(
            f"Star {self.star_index + 1} · t = {current_time:.8g}\n"
            + (f"Step dt = {self.step_sizes[self.step]:.8e}\n" if self.step else "Initial state\n") +
            f"a_x = {acceleration[0]:.8e}\n"
            f"a_y = {acceleration[1]:.8e}\n"
            f"a_z = {acceleration[2]:.8e}\n"
            f"|a| = {np.linalg.norm(acceleration):.8e}\n\n"
            f"v_x = {velocity[0]:.8e}\n"
            f"v_y = {velocity[1]:.8e}\n"
            f"v_z = {velocity[2]:.8e}\n"
            f"|v| = {np.linalg.norm(velocity):.8e}\n\n"
            f"{distance_text}\n(code length units)\n\n"
            f"E = {self.energies[self.step]:.10e}\n"
            f"ΔE = {self.energies[self.step] - self.energies[0]:.8e}")
        self.time_var.set(f"Time t = {current_time:.10g} / {self.times[-1]:.10g}    "
                          f"Step {self.step:,} / {self.last_step:,} steps    "
                          f"Step dt = {self.step_sizes[self.step]:.6g}")
        self.updating_slider = True
        try:
            self.timeline.update_frame(self)
        finally:
            self.updating_slider = False
        if redraw:
            self.active_canvas().draw_idle()
        return self.lines + self.points + [self.highlight, self.time_text,
                                          self.energy_cursor, self.acceleration_cursor]

    def active_canvas(self):
        return (self.diagnostic_canvas if self.notebook.select() == str(self.diagnostics_tab)
                else self.canvas)

    def pause(self):
        self.playing = False
        self.play_button.configure(text="Play")

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def single_step(self, direction):
        self.pause()
        self.draw_frame(self.step + direction)

    def rewind(self):
        self.pause()
        self.draw_frame(0)

    def tick(self):
        if self.closed:
            return
        if self.playing and not self.dragging:
            try:
                stride = max(1, int(self.stride_var.get()))
            except ValueError:
                stride = 1
            if self.step >= self.last_step:
                if self.loop_var.get():
                    self.draw_frame(0)
                else:
                    self.pause()
            else:
                self.draw_frame(min(self.last_step, self.step + stride))
        self.after_id = self.root.after(40, self.tick)

    def on_seek(self, value):
        if not self.updating_slider:
            self.draw_frame(self.nearest_index(self.times, float(value)))

    def on_slider_press(self, event):
        self.pause()
        self.dragging = True

    def on_slider_release(self, event):
        self.dragging = False

    def on_tab_change(self, event):
        if hasattr(self, "diagnostic_canvas"):
            if self.notebook.select() == str(self.diagnostics_tab):
                self.controls.pack_forget()
            elif not self.controls.winfo_manager():
                self.controls.pack(side="left", fill="y", padx=(0, 12), before=self.display)
            self.active_canvas().draw_idle()

    def on_hover(self, event):
        axis = self.time_axis_for.get(event.inaxes, event.inaxes)
        if axis is self.orbit_axes:
            index = self.star_index
            position = self.trajectories[index, self.step]
            fields = [f"t = {self.times[self.step]:.9g}", f"x = {position[0]:.8e}",
                      f"y = {position[1]:.8e}", f"z = {position[2]:.8e}",
                      f"|v| = {np.linalg.norm(self.velocities[self.step, index]):.8e}",
                      f"|a| = {np.linalg.norm(self.accelerations[self.step, index]):.8e}"]
            self.page_readouts[self.figure].set(self.format_readout(f"Orbit | selected Star {index + 1}", fields))
            self.hover_selection = (axis, self.step)
            return
        item = self.hover_items.get(axis)
        if item is None or event.xdata is None:
            self.clear_hover(event)
            return
        index = self.nearest_index(item["times"], event.xdata)
        selection = (axis, index)
        if selection == self.hover_selection:
            return
        self.hover_selection = selection
        fields = [f"t = {item['times'][index]:.9g}"]
        for key, values in item["series"].items():
            fields.append("dt = NA (initial state)" if key == "dt" and index == 0
                          else f"{key} = {values[index]:.8e}")
        self.page_readouts[axis.figure].set(self.format_readout(item["title"], fields))

    def seek_plot_event(self, event):
        if self.drag_axis is None or event.x is None:
            return
        # Transform pixels through the axis where dragging started, even outside it.
        time_value = self.drag_axis.transData.inverted().transform((event.x, event.y or 0))[0]
        time_value = float(np.clip(time_value, self.times[0], self.times[-1]))
        index = self.nearest_index(self.times, time_value)
        if index != self.step:
            self.draw_frame(index)

    def on_plot_motion(self, event):
        if self.drag_axis is not None:
            self.seek_plot_event(event)
        else:
            self.on_hover(event)

    def on_plot_press(self, event):
        toolbar = self.toolbar if event.canvas is self.canvas else self.diagnostic_toolbar
        self.mouse_press = None
        self.drag_axis = None
        if event.button != 1 or toolbar.mode or event.inaxes is None:
            return
        axis = self.time_axis_for.get(event.inaxes, event.inaxes)
        if axis in self.hover_items:
            self.pause()
            self.dragging = True
            self.drag_axis = axis
            self.seek_plot_event(event)
        elif event.inaxes is self.orbit_axes:
            self.mouse_press = (event.canvas, event.inaxes, event.x, event.y)

    def on_plot_release(self, event):
        if self.drag_axis is not None:
            self.seek_plot_event(event)
            self.drag_axis = None
            self.dragging = False
            return
        press, self.mouse_press = self.mouse_press, None
        if press is None or event.button != 1 or event.inaxes is not press[1]:
            return
        if event.x is None or event.y is None or np.hypot(event.x - press[2], event.y - press[3]) > 6:
            return
        positions = self.trajectories[:, self.step, :]
        x, y, _ = proj3d.proj_transform(positions[:, 0], positions[:, 1], positions[:, 2],
                                       self.orbit_axes.get_proj())
        pixels = self.orbit_axes.transData.transform(np.column_stack((x, y)))
        distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= 14:
            self.select_star(index)

    def fit_orbits(self):
        center = (self.trajectories[self.star_index, self.step] if self.follow_view_active
                  else np.zeros(3))
        # Fit the complete trajectory at the current frame without changing
        # the chosen follow target or allocating a second full trajectory.
        extent = 1e-6
        for dim in range(3):
            values = self.trajectories[:, :, dim]
            finite = values[np.isfinite(values)]
            if finite.size:
                extent = max(extent, 1.15 * abs(float(finite.min()) - center[dim]),
                             1.15 * abs(float(finite.max()) - center[dim]))
        self.set_orbit_limits(np.column_stack((center - extent, center + extent)))
        self.canvas.draw_idle()

    def choose_save_path(self, suffix, label, extra=""):
        if suffix == ".gif":
            output_path = self.animation_archive_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return filedialog.asksaveasfilename(
                parent=self.root, title=label, initialdir=str(output_path.parent),
                initialfile=output_path.name, defaultextension=suffix,
                filetypes=[(label, "*" + suffix)])
        output = RUN_HISTORY_DIRECTORY
        output.mkdir(parents=True, exist_ok=True)
        base = (f"{self.metadata['case_id']}_method_{self.metadata['method_id']}"
                f"_{self.metadata.get('compute_backend', 'cpu')}"
                f"_{self.metadata.get('adaptive_algorithm', 'timescale') if self.adaptive else 'fixed'}"
                f"_dtmax_{self.dt:g}_steps_{self.last_step}{extra}")
        if self.adaptive and self.metadata.get("adaptive_algorithm") == "acceleration_ratio":
            base += (f"_trigger_{self.metadata.get('acceleration_trigger_ratio', 2):g}"
                     f"_maxpower_{self.metadata.get('acceleration_max_power', 3)}_reference")
        return filedialog.asksaveasfilename(parent=self.root, title=label,
                                            initialdir=str(output), initialfile=base + suffix,
                                            defaultextension=suffix, filetypes=[(label, "*" + suffix)])


    def diagnostics_png(self):
        figure = self.diagnostic_figure
        original_size = figure.get_size_inches().copy()
        visibility = [line.get_visible() for line in self.diagnostic_cursors]
        try:
            for line in self.diagnostic_cursors:
                line.set_visible(False)
            # Archives use a fixed pixel size independent of the window dimensions.
            figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(figure, 1060, 900, for_export=True)
            output = BytesIO()
            figure.savefig(output, format="png", dpi=100, facecolor="white", bbox_inches=None)
            return output.getvalue()
        finally:
            figure.set_size_inches(original_size, forward=False)
            self.apply_figure_layout(figure, *(original_size * figure.dpi))
            for line, visible in zip(self.diagnostic_cursors, visibility):
                line.set_visible(visible)
            self.diagnostic_canvas.draw_idle()


    def save_image(self):
        self.pause()
        diagnostic = self.notebook.select() == str(self.diagnostics_tab)
        path = self.choose_save_path(".png", "PNG image", "_diagnostics" if diagnostic else "_overview")
        if not path:
            return
        figure = self.diagnostic_figure if diagnostic else self.figure
        original_size = figure.get_size_inches().copy()
        try:
            figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(figure, 1060, 900, for_export=True)
            figure.savefig(path, dpi=100, facecolor="white", bbox_inches=None)
            self.status_var.set(f"Image saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        finally:
            figure.set_size_inches(original_size, forward=False)
            self.apply_figure_layout(figure, *(original_size * figure.dpi))
            self.active_canvas().draw_idle()

    def save_data(self):
        self.pause()
        path = self.choose_save_path(".npz", "NumPy data")
        if not path:
            return
        try:
            np.savez_compressed(
                path, times=self.times, step_sizes=self.step_sizes, positions=self.trajectories,
                accelerations=self.accelerations, velocities=self.velocities, energies=self.energies,
                diagnostic_times=self.diagnostic_times,
                diagnostic_energy=np.array([item["energy"] for item in self.diagnostics]),
                momentum=np.array([item["momentum"] for item in self.diagnostics]),
                angular_momentum=np.array([item["angular_momentum"] for item in self.diagnostics]),
                com_pos=np.array([item["com_pos"] for item in self.diagnostics]),
                com_v=np.array([item["com_v"] for item in self.diagnostics]),
                metadata=json.dumps(self.metadata, ensure_ascii=False),
            )
            self.status_var.set(f"Saved every star and time step: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)

    def animation_archive_path(self):
        """Reuse the persisted PNG name, including any same-second timestamp suffix."""
        png_path = self.metadata.get("history_png_path")
        saved_id = self.metadata.get("history_saved_run_id")
        if not png_path or not saved_id or saved_id != str(self.metadata.get("run_id")):
            raise ValueError(
                "Save this run's history first using Retry saving run history. "
                "The animation will then use exactly the same timestamp as its data and image.")
        return Path(png_path).with_suffix(".gif")

    def save_animation(self):
        self.pause()
        try:
            path = self.choose_save_path(".gif", "Save animation (GIF)")
        except Exception as exc:
            messagebox.showerror("Animation unavailable", str(exc), parent=self.root)
            return
        if not path:
            return
        frame_count = simpledialog.askinteger(
            "Animation frames", "How many frames should be exported? More frames take longer and produce larger files.\n"
                        "The animation covers the full simulation interval without changing the original records.",
            parent=self.root, initialvalue=min(300, self.frame_count),
            minvalue=2, maxvalue=min(2000, self.frame_count))
        if frame_count is None:
            return
        original_step = self.step
        hover_texts = [(variable, variable.get()) for variable in self.page_readouts.values()]
        for variable, _ in hover_texts:
            variable.set("")
        # Sample GIF frames in physical time, not adaptive step number.
        frame_times = np.linspace(self.times[0], self.times[-1], frame_count)
        frames = np.array([self.nearest_index(self.times, t) for t in frame_times])
        temporary_path = None
        original_figure_size = self.figure.get_size_inches().copy()
        self.save_animation_button.configure(state="disabled")
        self.timeline.begin_export(self)
        try:
            self.status_var.set(f"Saving animation ({len(frames)} frames). Please wait...")
            self.root.update_idletasks()
            destination = Path(path)
            handle, temporary_name = tempfile.mkstemp(
                prefix="." + destination.stem + ".", suffix=".gif", dir=destination.parent)
            os.close(handle)
            temporary_path = Path(temporary_name)
            self.figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(self.figure, 1060, 900, for_export=True)
            writer = PillowWriter(fps=20)
            with writer.saving(self.figure, temporary_path, dpi=100):
                for index, frame in enumerate(frames):
                    self.draw_frame(frame, redraw=False)
                    writer.grab_frame()
                    self.export_progress(index, len(frames))
            # Publish only the completed GIF; a failed export preserves any existing animation.
            os.replace(temporary_path, destination)
            self.status_var.set(f"Animation saved: {path}")
        except Exception as exc:
            self.status_var.set("Animation export failed; existing files were preserved.")
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.figure.set_size_inches(original_figure_size, forward=False)
            self.apply_figure_layout(self.figure, *(original_figure_size * self.figure.dpi))
            self.save_animation_button.configure(state="normal")
            for variable, text in hover_texts:
                variable.set(text)
            self.draw_frame(original_step)
            self.timeline.finish_export(self)

    def export_progress(self, current, total):
        self.status_var.set(f"Saving animation: {current + 1} / {total} frames")
        self.timeline.update_export(self, current, total)
        self.root.update_idletasks()

    def close(self):
        self.closed = True
        self.timeline.detach(self)
        self.status_var.trace_remove("write", self.status_trace)
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        for canvas in (self.canvas, self.diagnostic_canvas):
            idle_id = getattr(canvas, "_idle_draw_id", None)
            if idle_id:
                # Cancel through the widget that registered the Tk command,
                # so its callback registry is cleaned before window destruction.
                canvas.get_tk_widget().after_cancel(idle_id)
                canvas._idle_draw_id = None
        self.figure.clear()
        self.diagnostic_figure.clear()
        self.root.destroy()

    def run(self):
        self.root.mainloop()




def prepare_batch_plan(plan):
    """Resolve every override before starting; freeze the seed across the batch."""
    if not isinstance(plan, dict) or not {"base", "runs"} <= set(plan) or set(plan) - {"base", "runs", "comparison"}:
        raise ValueError('The plan must contain "base", "runs", and optionally "comparison".')
    base, runs = plan["base"], plan["runs"]
    if not isinstance(base, dict) or not isinstance(runs, list) or not runs:
        raise ValueError("base must be an object; runs must be a nonempty list.")
    allowed = {"initial_mode", "seed", "orbit_index", "dt", "total_time", "method",
               "adaptive_enabled", "adaptive_algorithm", "adaptive_eta",
               "acceleration_trigger_ratio", "acceleration_max_power",
               "sample_every", "compute_backend", "particle_count"}
    resolved = []
    for index, override in enumerate(runs, 1):
        if not isinstance(override, dict):
            raise ValueError(f"Run {index}: parameters must be an object.")
        config = dict(base)
        config.update(override)
        unknown = set(config) - allowed - {"label"}
        if unknown:
            raise ValueError(f"Run {index}: unknown parameters: {sorted(unknown)}")
        label = str(config.pop("label", f"Run {index}"))
        if "adaptive_enabled" in config and not isinstance(config["adaptive_enabled"], bool):
            raise ValueError(f"Run {index}: adaptive_enabled must be true or false.")
        config["method"] = {"Leapfrog": "1", "RK4": "2"}.get(config.get("method"), config.get("method"))
        try:
            config = validate_config(config)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Run {index}: {exc}") from exc
        if config["initial_mode"] == "random":
            config["initial_mode"] = "manual"
        resolved.append({"label": label, "config": config})
    return resolved


def save_batch_arrays(result):
    """Atomically archive all retained arrays beside this run's timestamped PNG."""
    metadata = result["metadata"]
    path = Path(metadata["history_png_path"]).with_suffix(".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"]))["run_id"] == metadata["run_id"]:
                return path
        raise FileExistsError(f"Refusing to overwrite another run: {path}")
    diagnostics = result["diagnostics"]
    arrays = dict(times=result["record_times"], step_sizes=result["step_sizes"],
                  positions=result["trajectories"], accelerations=result["accelerations"],
                  velocities=result["velocities"], energies=result["energies"],
                  diagnostic_times=result["diagnostic_times"],
                  metadata=json.dumps(metadata, ensure_ascii=False))
    for key in ("energy", "momentum", "angular_momentum", "com_pos", "com_v"):
        arrays["diagnostic_energy" if key == "energy" else key] = np.asarray([d[key] for d in diagnostics])
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        # Windows rename refuses to overwrite an existing destination.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def publish_batch_position_columns(manifest, directory):
    """Overlay reference metrics on history without modifying immutable run summaries."""
    options = manifest.get("comparison", {})
    if not (options.get("rmse") or options.get("nrmse")):
        return
    ref = options["reference"]
    jobs = manifest["jobs"]
    for i, job in enumerate(jobs, 1):
        if job["status"] != "saved":
            continue
        row = next((r for r in manifest.get("comparisons", []) if r["test"] == i and r["against"] == ref), {})
        count = sum(r["against"] == ref and r["test"] != ref and r["status"] == "saved"
                    for r in manifest.get("comparisons", []))
        marker = f"reference_{count}"
        rmse = (marker if i == ref else row.get("rmse")) if options["rmse"] else None
        nrmse = (marker if i == ref else row.get("nrmse")) if options["nrmse"] else None
        entry = dict(run_id=job["run_id"], reference_test=ref, reference_run_id=jobs[ref-1].get("run_id"),
                     reference_comparison_count=count if i == ref else None,
                     rmse="NA" if rmse is None else rmse, nrmse="NA" if nrmse is None else nrmse,
                     status="saved" if i == ref else row.get("status", "pending"), settings=options)
        path = directory / "_position_comparisons" / _archive_record_path(directory, job["run_id"]).name
        _archive_write(path, json.dumps(entry, indent=2, allow_nan=False).encode("utf-8"))
    rebuild_run_history(directory)


def format_acceleration_ratio(variable):
    """Display the trigger ratio in fixed-point notation with one decimal place."""
    try:
        value = float(variable.get())
        if np.isfinite(value):
            variable.set(f"{value:.1f}")
    except (ValueError, TypeError):
        pass


def scientific_dt_text(value):
    """Format dt without rounding or fixing the number of mantissa digits."""
    from decimal import Decimal
    return format(Decimal(str(value)), "e")


def adjust_time_step(variable, factor, parent, scientific=False):
    """Change the visible dt without starting a run or changing adaptive settings."""
    from decimal import Decimal, InvalidOperation, localcontext
    try:
        entered = Decimal(variable.get())
        multiplier = Decimal(str(factor))
        with localcontext() as context:
            context.prec = max(28, len(entered.as_tuple().digits) + len(multiplier.as_tuple().digits) + 2)
            value = entered * multiplier
        if not value.is_finite() or value <= 0 or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("Time step must remain finite and positive.")
        variable.set(scientific_dt_text(value) if scientific else str(value.normalize()))
        return True
    except (ValueError, InvalidOperation, OverflowError) as exc:
        messagebox.showerror("Invalid time step", str(exc), parent=parent)
        return False


def time_step_controls(parent, variable, dialog_parent, scientific=False):
    frame = ttk.Frame(parent)
    buttons = []
    for text, factor in (("÷2", .5), ("×2", 2), ("÷5", .2), ("×5", 5), ("÷10", .1), ("×10", 10)):
        button = ttk.Button(frame, text=text, width=3, padding=1,
                            command=lambda f=factor: adjust_time_step(variable, f, dialog_parent, scientific=scientific))
        button.pack(side="left", padx=(0, 2))
        buttons.append(button)
    return frame, buttons


def validate_comparison_settings(settings, jobs):
    """Use one reference, normalization length and physical time grid for the batch."""
    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise ValueError("comparison must be an object.")
    allowed = {"rmse", "nrmse", "previous", "reference", "samples", "auto_reference"}
    if set(settings) - allowed:
        raise ValueError("Unknown comparison settings.")
    result = dict(rmse=False, nrmse=False, previous=False, reference=1, samples=1001, auto_reference=True)
    result.update(settings)
    for key in ("rmse", "nrmse", "previous", "auto_reference"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false.")
    for key, low, high in (("reference", 1, len(jobs)), ("samples", 2, 100000)):
        value = float(result[key])
        if not np.isfinite(value) or value != int(value) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
        result[key] = int(value)
    if result["auto_reference"]:
        result["reference"] = min(range(len(jobs)), key=lambda i: jobs[i]["config"]["dt"]) + 1
    if result["rmse"] or result["nrmse"]:
        reference = make_initial_state(jobs[result["reference"] - 1]["config"])
        def state_array(stars):
            return np.asarray([[star["mass"], *star["pos"], *star["v"]] for star in stars])
        state = state_array(reference)
        for i, job in enumerate(jobs, 1):
            if not np.array_equal(state, state_array(make_initial_state(job["config"]))):
                raise ValueError(f"Test {i}: trajectory comparison requires identical initial masses, positions, velocities and particle order. Disable comparison to run unrelated cases.")
        masses, positions = state[:, 0], state[:, 1:4]
        center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
        scale = float(np.sqrt(np.mean(np.sum((positions - center)**2, axis=1))))
        if result["nrmse"] and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("NRMSE needs a positive initial RMS radius about the center of mass. Use RMSE only for this initial state.")
        result.update(length_scale=scale, time_start=0.0,
                      time_end=min(job["config"]["total_time"] for job in jobs),
                      alignment="cubic Hermite interpolation using saved positions and velocities; exact saved nodes retained",
                      normalization="unweighted initial RMS radius about the mass-weighted center of mass of the reference test",
                      metric="sqrt(mean over stars and uniform time samples of squared 3D position distance); endpoints included")
    return result


def batch_execution_order(jobs, reference):
    """Run the reference first, then stably sort all remaining jobs by configured dt."""
    return [reference] + sorted((i for i in range(len(jobs)) if i != reference),
                                key=lambda i: jobs[i]["config"]["dt"])


def ordered_batch_plan(jobs, comparison=None):
    """Resolve the reference before ordering, and preserve its identity in saved plans."""
    jobs = prepare_batch_plan(dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in jobs]))
    options = validate_comparison_settings(comparison, jobs)
    order = batch_execution_order(jobs, options["reference"] - 1)
    options["reference"] = order.index(options["reference"] - 1) + 1
    jobs = [jobs[i] for i in order]
    settings = {key: options[key] for key in ("rmse", "nrmse", "previous", "reference", "samples", "auto_reference")}
    plan = dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in jobs], comparison=settings)
    return jobs, options, plan


def positions_on_grid(path, grid):
    """Align each saved trajectory in physical time, without extrapolation."""
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float)
        positions = np.asarray(data["positions"], dtype=float).transpose(1, 0, 2)
        velocities = np.asarray(data["velocities"], dtype=float)
        if (times.ndim != 1 or len(times) < 2 or positions.shape != velocities.shape
                or positions.shape[0] != len(times) or positions.shape[-1] != 3
                or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
                or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities))):
            raise ValueError(f"Invalid saved trajectory: {path}")
        if grid[0] < times[0] or grid[-1] > times[-1]:
            raise ValueError("Comparison times must lie inside both trajectories; extrapolation is disabled.")
        right = np.clip(np.searchsorted(times, grid, side="right"), 1, len(times)-1)
        left = right-1
        width = (times[right]-times[left])[:, None, None]
        u = ((grid-times[left])/(times[right]-times[left]))[:, None, None]
        aligned = ((2*u**3-3*u**2+1)*positions[left] + (u**3-2*u**2+u)*width*velocities[left]
                   + (-2*u**3+3*u**2)*positions[right] + (u**3-u**2)*width*velocities[right])
        # Avoid interpolation at exactly recorded sample times.
        nodes = np.searchsorted(times, grid)
        matched = times[np.minimum(nodes, len(times)-1)] == grid
        aligned[matched] = positions[nodes[matched]]
        if not np.all(np.isfinite(aligned)):
            raise ValueError("Non-finite interpolated positions.")
        return aligned


def update_batch_comparisons(manifest, directory):
    """Compute newly available pairs; a later reference is resolved automatically."""
    options = manifest.get("comparison", {})
    if not (options.get("rmse") or options.get("nrmse")):
        return
    jobs = manifest["jobs"]
    reference = options["reference"]
    old = {(row["test"], row["against"]): row for row in manifest.get("comparisons", [])}
    pairs = {}
    for test in range(1, len(jobs)+1):
        if test != reference:
            pairs.setdefault((test, reference), []).append("reference")
        if options["previous"] and test > 1:
            pairs.setdefault((test, test-1), []).append("previous")
    grid = np.linspace(options["time_start"], options["time_end"], options["samples"])
    rows = []
    for (test, against), kinds in pairs.items():
        row = old.get((test, against), dict(test=test, against=against, status="pending"))
        row["kind"] = "+".join(kinds)
        if row["status"] != "saved" and jobs[test-1]["status"] == jobs[against-1]["status"] == "saved":
            row.update(status="pending", message="")
            try:
                left = positions_on_grid(directory / jobs[test-1]["data"], grid)
                right = positions_on_grid(directory / jobs[against-1]["data"], grid)
                if left.shape != right.shape:
                    raise ValueError("Particle counts do not match.")
                squared = np.sum((left-right)**2, axis=2)
                per_star = np.sqrt(np.mean(squared, axis=0))
                rms = float(np.sqrt(np.mean(squared)))
                if not np.isfinite(rms):
                    raise ValueError("Non-finite position RMS; inspect trajectory scales.")
                row.update(status="saved", rmse=rms if options["rmse"] else None,
                           nrmse=rms/options["length_scale"] if options["nrmse"] else None,
                           per_star_rmse=per_star.tolist() if options["rmse"] else None,
                           per_star_nrmse=(per_star/options["length_scale"]).tolist() if options["nrmse"] else None)
            except Exception as exc:
                row.update(status="error", message=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    manifest["comparisons"] = rows


def comparison_dat(manifest):
    """Keep a readable comparison table with the definitions needed to reproduce it."""
    options = manifest["comparison"]
    lines = ["# Position comparisons: relative trajectory differences, not exact-solution errors.",
             f"# Reference: Test {options['reference']}; previous means the preceding executed run (reference first, then ascending dt).",
             f"# Common time interval: [{options['time_start']:.16g}, {options['time_end']:.16g}]; uniform samples: {options['samples']} (including endpoints).",
             f"# Length scale: {options['length_scale']:.16g}; {options['normalization']}.",
             f"# Alignment: {options['alignment']}.", f"# Metric: {options['metric']}.",
             "# Per-star values and full settings are stored in the companion batch JSON.", "", "",
             f"{'Test':>6} {'Against':>8} {'Kind':<20} {'Star':>6} {'RMSE':>23} {'NRMSE':>23} {'Status':<10}"]
    def number(value):
        return "NA" if value is None else f"{value:.16e}"
    for row in manifest.get("comparisons", []):
        def add(star, rms, nrms):
            lines.append(f"{row['test']:>6} {row['against']:>8} {row['kind']:<20} {str(star):>6} {number(rms):>23} {number(nrms):>23} {row['status']:<10}")
        add("All", row.get("rmse"), row.get("nrmse"))
        per = row.get("per_star_rmse") or row.get("per_star_nrmse") or []
        for i in range(len(per)):
            add(i+1, row['per_star_rmse'][i] if row.get('per_star_rmse') else None,
                row['per_star_nrmse'][i] if row.get('per_star_nrmse') else None)
        if row.get("message"):
            lines.append(f"# Test {row['test']} vs {row['against']}: {row['message']}")
    return "\n".join(lines) + "\n"


class BatchPlanEditor:
    """Edit numbered test cases with the same parameter choices as the main page."""
    def __init__(self, app, base):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Test mode · Configure runs, then start the batch")
        self.window.geometry("1120x800")
        self.window.minsize(920, 730)
        self.window.transient(app.root)
        self.window.grab_set()
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.jobs = [dict(label="Test 1", config=copy.deepcopy(base))]
        self.current = 0
        self.vars = {}
        self.widgets = {}
        top = ttk.Frame(self.window, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Select test").pack(side="left")
        self.selector = ttk.Combobox(top, state="readonly", width=32)
        self.selector.pack(side="left", padx=8)
        self.selector.bind("<<ComboboxSelected>>", self.select_test)
        ttk.Button(top, text="Save this test", command=self.save_current).pack(side="left", padx=4)
        ttk.Button(top, text="Add next test", command=self.add_test).pack(side="left", padx=4)
        ttk.Button(top, text="Delete this test", command=self.delete_test).pack(side="left", padx=4)
        form = ttk.LabelFrame(self.window, text="Parameters for the selected test", padding=12)
        form.grid(row=1, column=0, sticky="ew", padx=12)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        fields = [
            ("label", "Test label", None),
            ("initial_mode", "Initial conditions", ["Specified seed", "Known orbit"]),
            ("seed", "Seed", None),
            ("orbit_index", "Known orbit", [f"{i+1}. {o['name']}" for i, o in enumerate(STABLE_ORBITS)]),
            ("dt", "dt / upper limit", None), ("total_time", "Simulation duration T", None),
            ("method", "Integrator", ["Leapfrog", "RK4"]),
            ("sample_every", "Diagnostic interval (steps)", None),
            ("adaptive_enabled", "Enable adaptive time step", ["Off", "On"]),
            ("adaptive_algorithm", "Adaptive algorithm", ["Time-scale method", "Acceleration ratio (retained reference)"]),
            ("adaptive_eta", "Safety factor eta", None),
            ("acceleration_trigger_ratio", "Acceleration trigger ratio N", None),
            ("acceleration_max_power", "Maximum exponent K", None),
            ("compute_backend", "Compute device", ["CPU", "GPU (NVIDIA / CuPy)"]),
            ("particle_count", "Random particle count", None),
        ]
        for i, (key, title, choices) in enumerate(fields):
            row, col = i // 2, 2 * (i % 2)
            ttk.Label(form, text=title).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
            variable = tk.StringVar()
            self.vars[key] = variable
            host = ttk.Frame(form) if key == "dt" else form
            if key == "dt":
                host.grid(row=row, column=col+1, sticky="ew", padx=(0, 12), pady=4)
                host.columnconfigure(0, weight=1)
            widget = (ttk.Combobox(host, textvariable=variable, values=choices, state="readonly", width=28)
                      if choices else ttk.Entry(host, textvariable=variable, width=28))
            if key == "dt":
                widget.grid(row=0, column=0, sticky="ew")
                controls, self.dt_buttons = time_step_controls(host, variable, self.window, scientific=True)
                widget.bind("<FocusOut>", self.format_dt)
                widget.bind("<Return>", self.format_dt)
                controls.grid(row=1, column=0, sticky="w", pady=(3, 0))
            else:
                widget.grid(row=row, column=col+1, sticky="ew", padx=(0, 12), pady=4)
            self.widgets[key] = widget
            if key == "acceleration_trigger_ratio":
                widget.bind("<FocusOut>", lambda event: format_acceleration_ratio(self.vars["acceleration_trigger_ratio"]))
                widget.bind("<Return>", lambda event: format_acceleration_ratio(self.vars["acceleration_trigger_ratio"]))
            if choices:
                widget.bind("<<ComboboxSelected>>", self.sync_fields)
        self.widgets["orbit_index"].bind("<<ComboboxSelected>>", self.orbit_defaults)
        ttk.Button(form, text="New random seed", command=lambda: self.vars["seed"].set(str(secrets.randbits(32)))).grid(
            row=7, column=3, sticky="w")
        self.auto_reference_var = tk.BooleanVar(value=True)
        self.rmse_var = tk.BooleanVar(value=False)
        self.nrmse_var = tk.BooleanVar(value=False)
        self.previous_var = tk.BooleanVar(value=False)
        self.samples_var = tk.StringVar(value="1001")
        comparison = ttk.LabelFrame(self.window, text="Position comparison · Shared by all tests", padding=8)
        comparison.grid(row=3, column=0, sticky="ew", padx=12)
        ttk.Checkbutton(comparison, text="Position RMSE", variable=self.rmse_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(comparison, text="Position NRMSE", variable=self.nrmse_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(comparison, text="Also compare with previous test", variable=self.previous_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=12)
        ttk.Label(comparison, text="Reference test").grid(row=1, column=0, sticky="w", pady=5)
        self.reference_combo = ttk.Combobox(comparison, state="readonly", width=30)
        self.reference_combo.grid(row=1, column=1, sticky="ew")
        self.reference_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        ttk.Button(comparison, text="Use selected test as reference", command=self.use_selected_reference).grid(row=1, column=2, padx=12)
        ttk.Label(comparison, text="Uniform time samples").grid(row=1, column=3)
        ttk.Entry(comparison, textvariable=self.samples_var, width=8).grid(row=1, column=4, padx=8)
        ttk.Label(comparison, text="Same initial state required. Shared interval: 0 to shortest duration. NRMSE uses one fixed initial RMS radius.").grid(row=2, column=0, columnspan=5, sticky="w")
        ttk.Checkbutton(comparison, text="Automatic reference: smallest configured dt", variable=self.auto_reference_var,
                        command=self.refresh).grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(comparison, text="Reference runs first; remaining tests run from smallest to largest dt.").grid(row=3, column=3, columnspan=2, sticky="w")
        preview = ttk.Frame(self.window)
        preview.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        columns = (
            ("order", "Order", 55, "center"),
            ("role", "Role", 85, "w"),
            ("test", "Test / Label", 125, "w"),
            ("source", "Initial conditions", 300, "w"),
            ("method", "Method", 85, "w"),
            ("dt", "DT", 185, "e"),
            ("duration", "Duration T", 100, "e"),
            ("adaptive", "Adaptive method", 140, "w"),
            ("device", "Device", 65, "center"),
        )
        self.summary = ttk.Treeview(
            preview, columns=[item[0] for item in columns],
            show="headings", height=5, selectmode="none",
        )
        for key, title, width, anchor in columns:
            self.summary.heading(key, text=title, anchor=anchor)
            self.summary.column(key, width=width, minwidth=width,
                                anchor=anchor, stretch=(key == "source"))
        self.summary.tag_configure("reference", background="#e6f0ff")
        self.summary.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview, orient="vertical", command=self.summary.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(preview, orient="horizontal", command=self.summary.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.summary.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.status = tk.StringVar(value="Edit Test 1, save it, then add the next test. New tests copy the selected test's parameters.")
        ttk.Label(self.window, textvariable=self.status, wraplength=1000, padding=(12, 2)).grid(row=4, column=0, sticky="ew")
        bar = ttk.Frame(self.window, padding=12)
        bar.grid(row=5, column=0, sticky="ew")
        ttk.Button(bar, text="Load plan...", command=self.load_plan).pack(side="left")
        ttk.Button(bar, text="Save plan...", command=self.save_plan).pack(side="left", padx=8)
        ttk.Label(bar, text="Output: graph and data (PNG + NPZ + run summary)").pack(side="left", padx=8)
        ttk.Button(bar, text="Start all tests", command=self.start).pack(side="right")
        self.refresh()
        self.load_current()

    def sync_fields(self, event=None):
        orbit = self.vars["initial_mode"].get() == "Known orbit"
        enabled = self.vars["adaptive_enabled"].get() == "On"
        timescale = self.vars["adaptive_algorithm"].get() == "Time-scale method"
        for key, active in (("seed", not orbit), ("particle_count", not orbit), ("orbit_index", orbit),
                            ("adaptive_algorithm", enabled), ("adaptive_eta", enabled and timescale),
                            ("acceleration_trigger_ratio", enabled and not timescale),
                            ("acceleration_max_power", enabled and not timescale)):
            widget = self.widgets[key]
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox) else "normal") if active else "disabled")

    def orbit_defaults(self, event=None):
        orbit = STABLE_ORBITS[self.widgets["orbit_index"].current()]
        self.vars["dt"].set(scientific_dt_text(orbit["dt"]))
        self.vars["total_time"].set(str(orbit["dt"] * orbit["tot_time"]))
        self.vars["particle_count"].set(str(orbit["N"]))
        if orbit.get("recommended_method"):
            self.vars["method"].set("Leapfrog" if str(orbit["recommended_method"]) == "1" else "RK4")
        self.sync_fields()

    def format_dt(self, event=None):
        # Wait until editing is complete, so intermediate text such as "1e-" is allowed.
        from decimal import InvalidOperation
        try:
            value = scientific_dt_text(self.vars["dt"].get())
        except (InvalidOperation, ValueError):
            return
        self.vars["dt"].set(value)

    def load_current(self):
        job = self.jobs[self.current]
        config = job["config"]
        for key, variable in self.vars.items():
            variable.set(str(config.get(key, "")))
        self.vars["dt"].set(scientific_dt_text(config["dt"]))
        format_acceleration_ratio(self.vars["acceleration_trigger_ratio"])
        self.vars["label"].set(job["label"])
        self.vars["initial_mode"].set("Known orbit" if config["initial_mode"] == "orbit" else "Specified seed")
        self.widgets["orbit_index"].current(max(0, config.get("orbit_index", 0)))
        self.vars["method"].set("Leapfrog" if config["method"] == "1" else "RK4")
        self.vars["adaptive_enabled"].set("On" if config["adaptive_enabled"] else "Off")
        self.vars["adaptive_algorithm"].set("Time-scale method" if config["adaptive_algorithm"] == "timescale" else "Acceleration ratio (retained reference)")
        self.vars["compute_backend"].set("CPU" if config["compute_backend"] == "cpu" else "GPU (NVIDIA / CuPy)")
        self.sync_fields()

    def read_current(self):
        self.format_dt()
        config = {key: variable.get() for key, variable in self.vars.items()}
        config["initial_mode"] = "orbit" if config["initial_mode"] == "Known orbit" else "manual"
        config["orbit_index"] = self.widgets["orbit_index"].current()
        config["adaptive_enabled"] = config["adaptive_enabled"] == "On"
        config["adaptive_algorithm"] = "timescale" if config["adaptive_algorithm"] == "Time-scale method" else "acceleration_ratio"
        config["compute_backend"] = "cpu" if config["compute_backend"] == "CPU" else "gpu"
        if config["initial_mode"] == "orbit":
            orbit = STABLE_ORBITS[config["orbit_index"]]
            config.update(seed=orbit["seed"], particle_count=orbit["N"])
        return prepare_batch_plan(dict(base={}, runs=[config]))[0]

    def save_current(self):
        try:
            self.jobs[self.current] = self.read_current()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid test parameters", str(exc), parent=self.window)
            return False
        self.refresh()
        self.status.set(f"Test {self.current + 1} settings saved. Save plan exports all settings for reuse.")
        return True

    def use_selected_reference(self):
        self.auto_reference_var.set(False)
        self.reference_combo.configure(state="readonly")
        self.reference_combo.current(self.current)
        self.refresh()

    def refresh(self):
        reference = self.reference_combo.current()
        if self.auto_reference_var.get():
            reference = min(range(len(self.jobs)), key=lambda i: self.jobs[i]["config"]["dt"])
        self.reference_combo.configure(state="disabled" if self.auto_reference_var.get() else "readonly")
        self.reference_combo.configure(values=[f"{i+1}. {job['label']}" for i, job in enumerate(self.jobs)])
        self.reference_combo.current(min(max(reference, 0), len(self.jobs)-1))
        self.selector.configure(values=[f"{i+1}. {job['label']}" for i, job in enumerate(self.jobs)])
        self.selector.current(self.current)
        self.summary.delete(*self.summary.get_children())
        ordered = [(i+1, self.jobs[i]) for i in batch_execution_order(self.jobs, self.reference_combo.current())]
        for rank, (i, job) in enumerate(ordered, 1):
            c = job["config"]
            source = STABLE_ORBITS[c["orbit_index"]]["name"] if c["initial_mode"] == "orbit" else f"seed {c['seed']}"
            method = "Leapfrog" if c["method"] == "1" else "RK4"
            adaptive = c["adaptive_algorithm"] if c["adaptive_enabled"] else "Off"
            self.summary.insert(
                "", "end",
                values=("★" if rank == 1 else str(rank),
                        "Reference" if rank == 1 else "",
                        f"{i}: {job['label']}", source, method,
                        scientific_dt_text(c["dt"]), f"{c['total_time']:g}",
                        adaptive, c["compute_backend"].upper()),
                tags=("reference",) if rank == 1 else (),
            )

    def select_test(self, event=None):
        selected = self.selector.current()
        if self.save_current():
            self.current = selected
            self.refresh()
            self.load_current()

    def add_test(self):
        if not self.save_current():
            return
        job = copy.deepcopy(self.jobs[self.current])
        job["label"] = f"Test {len(self.jobs) + 1}"
        self.jobs.append(job)
        self.current = len(self.jobs) - 1
        self.refresh()
        self.load_current()

    def delete_test(self):
        if len(self.jobs) == 1:
            self.status.set("Keep at least one test in the plan.")
            return
        reference = self.reference_combo.current()
        removed = self.current
        self.jobs.pop(removed)
        # Preserve the referenced job when deleting an earlier test.
        self.reference_combo.current(0 if removed == reference else reference - (removed < reference))
        self.current = min(self.current, len(self.jobs) - 1)
        self.refresh()
        self.load_current()

    def comparison_settings(self):
        return dict(rmse=self.rmse_var.get(), nrmse=self.nrmse_var.get(), previous=self.previous_var.get(),
                    reference=self.reference_combo.current()+1, samples=self.samples_var.get(), auto_reference=self.auto_reference_var.get())

    def plan(self):
        return dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in self.jobs],
                    comparison=self.comparison_settings())

    def save_plan(self):
        if not self.save_current():
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("Test plan", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.plan(), indent=2), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Plan not saved", str(exc), parent=self.window)

    def load_plan(self):
        path = filedialog.askopenfilename(parent=self.window, filetypes=[("Test plan", "*.json")])
        if not path:
            return
        try:
            plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if "plan" in plan:
                plan = plan["plan"]
            jobs = prepare_batch_plan(plan)
            settings = validate_comparison_settings(plan.get("comparison"), jobs)
        except (ValueError, TypeError, OSError) as exc:
            messagebox.showerror("Invalid plan", str(exc), parent=self.window)
            return
        self.jobs, self.current = jobs, 0
        self.auto_reference_var.set(settings["auto_reference"])
        self.refresh()
        self.reference_combo.current(settings["reference"]-1)
        self.rmse_var.set(settings["rmse"])
        self.nrmse_var.set(settings["nrmse"])
        self.previous_var.set(settings["previous"])
        self.samples_var.set(str(settings["samples"]))
        self.refresh()
        self.load_current()

    def start(self):
        if not self.save_current():
            return
        try:
            self.app.start_batch(copy.deepcopy(self.jobs), self.comparison_settings())
        except Exception as exc:
            messagebox.showerror("Batch not started", str(exc), parent=self.window)
            return
        if self.app.running:
            self.window.destroy()


class NBodyApp:
    """Combine a parameter panel, background simulation, and replaceable result view."""
    def __init__(self, root=None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("N-body simulation · Settings and interactive playback")
        width = min(1640, self.root.winfo_screenwidth() - 60)
        height = min(1080, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1100, width), min(740, height))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.viewer = None
        self.worker = None
        self.cancel_event = threading.Event()
        self.messages = queue.Queue()
        self.pending_history = []
        self.history_migration_pending = True
        self.closed = False
        self.poll_id = None
        self.running = False
        self.batch_jobs = []
        self.batch_index = 0
        self.batch_manifest = None
        self.batch_path = None
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=4)
        style.configure("Note.TLabel", foreground="#52606d", font=("Microsoft YaHei UI", 9))
        self.configuration_area = ttk.Frame(self.root)
        self.configuration_area.grid(row=0, column=0, sticky="ew")
        self.configuration_area.columnconfigure(0, weight=1)
        self.settings_visible = True
        self.settings_toggle = ttk.Button(
            self.configuration_area, text="Hide settings", command=self.toggle_settings)
        self.settings_toggle.grid(row=0, column=0, sticky="w", padx=12, pady=(3, 0))
        self.batch_button = ttk.Button(self.configuration_area, text="Test mode...", command=self.open_batch_editor)
        self.batch_button.grid(row=0, column=0, sticky="e", padx=12, pady=(3, 0))
        self.comparison_button = ttk.Button(self.configuration_area, text="Comparison results...", command=self.show_comparisons)
        self.comparison_button.grid(row=0, column=0, sticky="e", padx=(0, 125), pady=(3, 0))
        self.settings = ttk.LabelFrame(self.configuration_area,
                                      text="Simulation settings · Edit, then click Start / Rerun", padding=(12, 6))
        self.settings.grid(row=1, column=0, sticky="ew", padx=12, pady=(3, 0))
        for col in (1, 3, 5, 7):
            self.settings.columnconfigure(col, weight=1)
        self.mode_var = tk.StringVar(value="Specified seed")
        self.seed_var = tk.StringVar(value=str(secrets.randbits(32)))
        self.orbit_var = tk.StringVar()
        self.dt_var = tk.StringVar(value="1")
        self.duration_var = tk.StringVar(value="10000")
        self.method_var = tk.StringVar(value="Leapfrog (KDK)")
        self.sample_var = tk.StringVar(value="10")
        self.adaptive_var = tk.BooleanVar(value=False)
        self.adaptive_algorithm_var = tk.StringVar(value="Time-scale method")
        self.eta_var = tk.StringVar(value="0.03")
        self.trigger_ratio_var = tk.StringVar(value="2.0")
        self.max_power_var = tk.StringVar(value="3")
        self.backend_var = tk.StringVar(value="CPU")
        self.particle_count_var = tk.StringVar(value="5")
        self.record_run_var = tk.BooleanVar(value=True)
        self.form_widgets = []

        def label(text, row, col):
            ttk.Label(self.settings, text=text).grid(row=row, column=col, sticky="w", padx=(0, 5), pady=3)

        def entry(variable, row, col, width=11):
            host = ttk.Frame(self.settings) if variable is self.dt_var else self.settings
            if variable is self.dt_var:
                host.grid(row=row, column=col, sticky="ew", padx=(0, 12), pady=3)
                host.columnconfigure(0, weight=1)
            widget = ttk.Entry(host, textvariable=variable, width=width)
            if variable is self.dt_var:
                widget.grid(row=0, column=0, sticky="ew")
                controls, buttons = time_step_controls(host, variable, self.root)
                controls.grid(row=1, column=0, sticky="w", pady=(3, 0))
                self.dt_buttons = buttons
                self.form_widgets.extend(buttons)
            else:
                widget.grid(row=row, column=col, sticky="ew", padx=(0, 12), pady=3)
            self.form_widgets.append(widget)
            return widget

        label("Initial conditions", 0, 0)
        self.mode_combo = ttk.Combobox(self.settings, textvariable=self.mode_var, state="readonly",
                                      values=["Random seed", "Specified seed", "Known orbit"], width=17)
        self.mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.mode_combo)
        label("Seed", 0, 2)
        self.seed_entry = entry(self.seed_var, 0, 3)
        label("Known orbit", 0, 4)
        self.orbit_combo = ttk.Combobox(
            self.settings, textvariable=self.orbit_var, state="readonly",
            values=[f"{i + 1}. {orbit['name']}" for i, orbit in enumerate(STABLE_ORBITS)], width=34)
        self.orbit_combo.grid(row=0, column=5, columnspan=3, sticky="ew", pady=3)
        self.orbit_combo.current(0)
        self.form_widgets.append(self.orbit_combo)
        label("dt / upper limit", 1, 0)
        entry(self.dt_var, 1, 1)
        label("Simulation duration T", 1, 2)
        entry(self.duration_var, 1, 3)
        label("Integrator", 1, 4)
        self.method_combo = ttk.Combobox(self.settings, textvariable=self.method_var, state="readonly",
                                        values=["Leapfrog (KDK)", "RK4 (fourth order)"], width=22)
        self.method_combo.grid(row=1, column=5, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.method_combo)
        label("Diagnostic interval (steps)", 1, 6)
        entry(self.sample_var, 1, 7, width=8)
        self.adaptive_check = ttk.Checkbutton(
            self.settings, text="Enable adaptive time step", variable=self.adaptive_var, command=self.sync_fields)
        self.adaptive_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        self.form_widgets.append(self.adaptive_check)
        label("Safety factor eta", 2, 2)
        self.eta_entry = entry(self.eta_var, 2, 3)
        self.run_button = ttk.Button(self.settings, text="Start / Rerun", command=self.start_run)
        self.run_button.grid(row=2, column=4, columnspan=2, sticky="ew", padx=(0, 12), pady=3)
        self.cancel_button = ttk.Button(self.settings, text="Cancel run", command=self.cancel_run, state="disabled")
        self.cancel_button.grid(row=2, column=6, columnspan=2, sticky="ew", pady=3)
        label("Compute device", 3, 0)
        self.backend_combo = ttk.Combobox(
            self.settings, textvariable=self.backend_var, state="readonly",
            values=["CPU", "GPU (NVIDIA / CuPy)"], width=22)
        self.backend_combo.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.backend_combo)
        label("Random particle count", 3, 2)
        self.particle_count_entry = entry(self.particle_count_var, 3, 3)
        ttk.Label(self.settings, text="Double precision (float64); CPU may be faster for small particle counts.",
                  style="Note.TLabel").grid(row=3, column=4, columnspan=4, sticky="w")
        label("Adaptive algorithm", 4, 0)
        self.adaptive_algorithm_combo = ttk.Combobox(
            self.settings, textvariable=self.adaptive_algorithm_var, state="disabled",
            values=["Time-scale method", "Acceleration ratio (retained reference)"], width=23)
        self.adaptive_algorithm_combo.grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.adaptive_algorithm_combo.bind("<<ComboboxSelected>>", lambda event: self.sync_fields())
        self.form_widgets.append(self.adaptive_algorithm_combo)
        label("Acceleration trigger ratio N", 4, 2)
        self.trigger_ratio_entry = entry(self.trigger_ratio_var, 4, 3)
        self.trigger_ratio_entry.bind("<FocusOut>", lambda event: format_acceleration_ratio(self.trigger_ratio_var))
        self.trigger_ratio_entry.bind("<Return>", lambda event: format_acceleration_ratio(self.trigger_ratio_var))
        label("Maximum exponent K", 4, 4)
        self.max_power_entry = entry(self.max_power_var, 4, 5)
        ttk.Label(self.settings, text="Minimum suggested dt = initial dt / N^K",
                  style="Note.TLabel").grid(row=4, column=6, columnspan=2, sticky="w")
        self.algorithm_note_var = tk.StringVar()
        ttk.Label(self.settings, textvariable=self.algorithm_note_var,
                  style="Note.TLabel", wraplength=width - 70).grid(
                      row=5, column=0, columnspan=8, sticky="w")
        self.hint_var = tk.StringVar()
        ttk.Label(self.settings, textvariable=self.hint_var, style="Note.TLabel",
                  wraplength=width - 70).grid(row=6, column=0, columnspan=8, sticky="w", pady=(4, 0))
        self.record_check = ttk.Checkbutton(self.settings, text="Record this run (single-run mode)", variable=self.record_run_var)
        self.record_check.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.form_widgets.append(self.record_check)
        ttk.Label(self.settings, text="Off: view results without automatic history or PNG. Test mode keeps its batch archives.",
                  style="Note.TLabel").grid(row=7, column=3, columnspan=5, sticky="w")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_source_change)
        self.orbit_combo.bind("<<ComboboxSelected>>", self.use_orbit_defaults)
        self.sync_fields()

        self.settings.bind("<Configure>", lambda event: [
            child.configure(wraplength=max(450, event.width - 28))
            for child in self.settings.winfo_children()
            if isinstance(child, ttk.Label) and int(child.cget("wraplength") or 0) > 0])
        self.result_host = ttk.Frame(self.root)
        self.result_host.grid(row=1, column=0, sticky="nsew")
        self.result_host.columnconfigure(0, weight=1)
        self.result_host.rowconfigure(0, weight=1)
        self.placeholder = ttk.Label(
            self.result_host, text="Select initial conditions and parameters above, then click Start / Rerun.\n\n"
            "Completed runs display trajectories, energy and per-star acceleration. Velocity values remain available for inspection and export.\n"
            "Drag the playback slider at the bottom or the yellow time cursor in a plot.",
            anchor="center", justify="center", font=("Microsoft YaHei UI", 13))
        self.placeholder.grid(row=0, column=0, sticky="nsew")
        self.timeline = SharedTimeline(self.root)
        self.progress_footer = self.timeline
        self.progress_footer.grid(row=2, column=0, sticky="ew")
        self.progress_var = self.timeline.value
        self.progress_bar = self.timeline.scale
        self.progress_text = self.timeline.message
        self.history_status = tk.StringVar()
        # Successful automatic saves reserve no footer space; show Retry only on a failure.
        self.retry_history_button = ttk.Button(
            self.progress_footer, text="Retry saving run history", command=self.save_pending_history,
            state="disabled")
        self.retry_history_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.retry_history_button.grid_remove()
        self.save_pending_history()
        self.poll_id = self.root.after(100, self.poll_worker)

    def show_comparisons(self):
        manifest = self.batch_manifest
        if not manifest or not manifest.get("comparisons"):
            messagebox.showinfo("Position comparisons", "Enable Position RMSE or NRMSE in Test mode and configure at least two tests.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Position comparison results")
        window.geometry("980x460")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        options = manifest["comparison"]
        ttk.Label(window, text=f"Reference: Test {options['reference']} | Common time: 0 to {options['time_end']:.8g} | Samples: {options['samples']} | Length scale: {options['length_scale']:.8g}\n"
                  "Relative differences, not exact-solution errors. Test numbers follow execution order (reference first, then ascending dt). Previous = preceding run. Reopen to refresh.", padding=10).grid(row=0, column=0, sticky="ew")
        columns = ("test", "against", "kind", "star", "rmse", "nrmse", "status")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for key in columns:
            tree.heading(key, text=key.upper())
            tree.column(key, width=90 if key in ("test", "against", "star") else 145, anchor="center")
        tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(window, command=tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        for row in manifest["comparisons"]:
            def add(star, rms, nrms):
                tree.insert("", "end", values=(row['test'], row['against'], row['kind'], star,
                            "NA" if rms is None else f"{rms:.8e}", "NA" if nrms is None else f"{nrms:.8e}", row['status']))
            add("All", row.get("rmse"), row.get("nrmse"))
            per = row.get("per_star_rmse") or row.get("per_star_nrmse") or []
            for i in range(len(per)):
                add(i+1, row['per_star_rmse'][i] if row.get('per_star_rmse') else None,
                    row['per_star_nrmse'][i] if row.get('per_star_nrmse') else None)
        errors = [f"Test {r['test']} vs {r['against']}: {r.get('message', '')}" for r in manifest['comparisons'] if r['status'] == 'error']
        ttk.Label(window, text="\n".join(errors) if errors else f"Saved table: {self.batch_manifest.get('comparison_file', '')}", wraplength=930, padding=10).grid(row=2, column=0, sticky="ew")

    def open_batch_editor(self):
        if self.running:
            return
        try:
            base = self.read_config()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid base parameters", str(exc), parent=self.root)
            return
        self.seed_var.set(str(base["seed"]))
        BatchPlanEditor(self, base)

    def start_batch(self, jobs, comparison=None):
        if self.running:
            return
        if self.pending_history:
            messagebox.showerror("Archive pending", "Retry saving the previous run before starting a batch.", parent=self.root)
            return
        jobs, options, plan = ordered_batch_plan(jobs, comparison)
        stamp = datetime.now(gettz("America/Toronto")).strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.batch_path = RUN_HISTORY_DIRECTORY / f"batch_{stamp}.json"
        self.batch_manifest = dict(plan=plan, jobs=[dict(**job, status="pending") for job in jobs], comparison=options)
        update_batch_comparisons(self.batch_manifest, self.batch_path.parent)
        self.write_batch_manifest()
        self.batch_jobs = jobs
        self.batch_index = 0
        self.start_run(config=jobs[0]["config"])

    def write_batch_manifest(self):
        if self.batch_manifest.get("comparison", {}).get("rmse") or self.batch_manifest.get("comparison", {}).get("nrmse"):
            dat_path = self.batch_path.with_name(self.batch_path.stem + "_comparisons.dat")
            _archive_write(dat_path, comparison_dat(self.batch_manifest).encode("utf-8"))
            self.batch_manifest["comparison_file"] = dat_path.name
        _archive_write(self.batch_path, json.dumps(self.batch_manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        publish_batch_position_columns(self.batch_manifest, self.batch_path.parent)

    def batch_finished_run(self, success, message=""):
        if not self.batch_jobs:
            return
        entry = self.batch_manifest["jobs"][self.batch_index]
        entry["status"] = "saved" if success else "stopped"
        entry["message"] = message
        if success:
            entry["plot"] = Path(self.viewer.metadata["history_png_path"]).name
            entry["data"] = Path(entry["plot"]).with_suffix(".npz").name
            entry["run_id"] = self.viewer.metadata["run_id"]
        try:
            update_batch_comparisons(self.batch_manifest, self.batch_path.parent)
            self.write_batch_manifest()
        except Exception as exc:
            success = False
            messagebox.showerror("Batch manifest not saved", str(exc), parent=self.root)
        total = len(self.batch_jobs)
        self.batch_index += 1
        if success and self.batch_index < total:
            self.start_run(config=self.batch_jobs[self.batch_index]["config"])
        else:
            self.batch_jobs = []
            comparison_errors = sum(row["status"] == "error" for row in self.batch_manifest.get("comparisons", []))
            if comparison_errors:
                messagebox.showerror("Position comparison failed", f"{comparison_errors} comparison(s) failed. Simulation files are saved. Open Comparison results for details.", parent=self.root)
            self.progress_text.set(f"Batch complete: {total}/{total} runs saved." if success else
                                   f"Batch stopped at run {self.batch_index}/{total}. Completed archives are retained. {message}")

    def toggle_settings(self, visible=None):
        self.settings_visible = not self.settings_visible if visible is None else bool(visible)
        if self.settings_visible:
            self.settings.grid()
        else:
            self.settings.grid_remove()
        self.settings_toggle.configure(text="Hide settings" if self.settings_visible else "Show settings")

    def save_pending_history(self, viewer=None):
        try:
            if self.history_migration_pending:
                imported = migrate_legacy_history()
                rebuild_run_history()
                self.history_migration_pending = False
                if imported:
                    self.history_status.set(f"Imported {imported} previous runs. Their missing plots are marked NA.")
            while self.pending_history:
                record = self.pending_history[0]
                if "png_bytes" not in record:
                    if viewer is not None and record["metadata"] is viewer.metadata:
                        record["png_bytes"] = viewer.diagnostics_png()
                    else:
                        # Render from retained arrays if an earlier export failed.
                        # A closed viewer's figure is cleared, so it must not be queued by reference.
                        temporary_root = tk.Toplevel(self.root)
                        temporary_root.withdraw()
                        temporary_viewer = None
                        try:
                            temporary_viewer = NBodyViewer(root=temporary_root, **record["result"])
                            record["png_bytes"] = temporary_viewer.diagnostics_png()
                        finally:
                            if temporary_viewer is not None:
                                temporary_viewer.close()
                            else:
                                temporary_root.destroy()
                path = save_run_history(record)
                if record.get("save_full_data"):
                    save_batch_arrays(record["result"])
                record.pop("result", None)
                self.pending_history.pop(0)
                self.history_status.set(f"Saved data + conservation plot. Clickable image index: {path.parent / 'index.html'}")
        except Exception as exc:
            self.history_status.set(
                f"Archive pending ({len(self.pending_history)} new runs): {exc}. "
                "Resolve the issue and click Retry before closing.")
            self.retry_history_button.configure(state="normal")
            self.retry_history_button.grid()
            messagebox.showerror("Run archive not saved", self.history_status.get(), parent=self.root)
            return False
        self.retry_history_button.configure(state="disabled")
        self.retry_history_button.grid_remove()
        return True

    def sync_fields(self):
        if self.running:
            return
        mode = self.mode_var.get()
        self.seed_entry.configure(state="normal" if mode == "Specified seed" else "disabled")
        self.particle_count_entry.configure(state="disabled" if mode == "Known orbit" else "normal")
        self.orbit_combo.configure(state="readonly" if mode == "Known orbit" else "disabled")
        use_timescale = self.adaptive_algorithm_var.get() == "Time-scale method"
        enabled = self.adaptive_var.get()
        self.adaptive_algorithm_combo.configure(state="readonly" if enabled else "disabled")
        self.eta_entry.configure(state="normal" if enabled and use_timescale else "disabled")
        self.trigger_ratio_entry.configure(state="normal" if enabled and not use_timescale else "disabled")
        self.max_power_entry.configure(state="normal" if enabled and not use_timescale else "disabled")
        self.algorithm_note_var.set(
            "Selects dt from pairwise gravitational and encounter time scales; smaller eta gives more conservative steps."
            if use_timescale else
            "Each star is compared with its retained reference, so accumulated growth also triggers adjustment. The largest level controls dt, capped at K. Zero values do not automatically reduce dt.")
        seed_hint = ("Generates a new seed for each run; use Specified seed to reproduce it."
                     if mode == "Random seed" else
                     "Known orbits use preset positions and velocities, independent of random seeds."
                     if mode == "Known orbit" else "The same seed recreates the same initial positions and velocities.")
        sampling_hint = ("Ratio method: diagnostics are sampled every specified number of actual steps, without shortening dt."
                         if enabled and not use_timescale else "Diagnostic sampling interval = entered value × dt.")
        self.hint_var.set(seed_hint + "  " + sampling_hint + "Full trajectories, velocities and accelerations are retained at every step.")

    def on_source_change(self, event=None):
        if self.mode_var.get() == "Known orbit":
            self.use_orbit_defaults()
        else:
            self.dt_var.set("1")
            self.duration_var.set("10000")
            self.particle_count_var.set("5")
        self.sync_fields()

    def use_orbit_defaults(self, event=None):
        index = self.orbit_combo.current()
        if index < 0:
            return
        orbit = STABLE_ORBITS[index]
        self.dt_var.set(f"{orbit['dt']:.16g}")
        self.duration_var.set(f"{orbit['dt'] * orbit['tot_time']:.16g}")
        if orbit.get("recommended_method"):
            self.method_combo.current(int(orbit["recommended_method"]) - 1)
        self.sync_fields()
        if orbit.get("note"):
            self.hint_var.set(self.hint_var.get() + "  Orbit note: " + orbit["note"])

    def read_config(self):
        mode = {"Random seed": "random", "Specified seed": "manual", "Known orbit": "orbit"}[self.mode_var.get()]
        # Generate at the start of each random run, and retain the actual seed on screen.
        seed = secrets.randbits(32) if mode == "random" else self.seed_var.get()
        if mode == "orbit":
            seed = STABLE_ORBITS[self.orbit_combo.current()]["seed"]
        return validate_config(dict(
            initial_mode=mode, seed=seed, orbit_index=self.orbit_combo.current(),
            dt=self.dt_var.get(), total_time=self.duration_var.get(),
            method="1" if self.method_combo.current() == 0 else "2",
            adaptive_enabled=self.adaptive_var.get(),
            adaptive_algorithm=("timescale" if self.adaptive_algorithm_combo.current() == 0
                                else "acceleration_ratio"),
            adaptive_eta=(self.eta_var.get() if self.adaptive_var.get()
                          and self.adaptive_algorithm_combo.current() == 0 else .03),
            acceleration_trigger_ratio=(self.trigger_ratio_var.get() if self.adaptive_var.get()
                                        and self.adaptive_algorithm_combo.current() == 1 else 2.0),
            acceleration_max_power=(self.max_power_var.get() if self.adaptive_var.get()
                                    and self.adaptive_algorithm_combo.current() == 1 else 3),
            sample_every=self.sample_var.get(),
            compute_backend="gpu" if self.backend_combo.current() == 1 else "cpu",
            particle_count=(STABLE_ORBITS[self.orbit_combo.current()]["N"]
                            if mode == "orbit" else self.particle_count_var.get())))

    def set_running(self, running):
        self.running = running
        if running:
            self.timeline.begin_computation()
        else:
            self.timeline.attach(self.viewer)
        for widget in self.form_widgets:
            widget.configure(state="disabled" if running else
                             "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        self.batch_button.configure(state="disabled" if running else "normal")
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if not running:
            self.sync_fields()

    def start_run(self, config=None):
        if self.running:
            return
        try:
            config = self.read_config() if config is None else validate_config(config)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=self.root)
            return
        if config["initial_mode"] != "orbit":
            self.seed_var.set(str(config["seed"]))
        if self.viewer:
            self.viewer.pause()
        self.cancel_event = threading.Event()
        self.set_running(True)
        self.progress_var.set(0)
        self.progress_text.set(
            f"Starting {config['compute_backend'].upper()} computation... "
            "The first GPU run may compile its kernel. Existing plots still show the previous result.")

        # Record the Toronto start time and UTC offset; saving later does not replace this timestamp.
        run_started_at = datetime.now(gettz("America/Toronto")).isoformat(timespec="microseconds")
        run_id = secrets.token_hex(16)
        record_enabled = bool(self.batch_jobs) or self.record_run_var.get()

        def worker():
            try:
                result = run_simulation(config, progress=lambda *data: self.messages.put(("progress", data)),
                                        cancel=self.cancel_event)
                result["metadata"].update(run_started_at=run_started_at, run_id=run_id, record_enabled=record_enabled)
                if self.batch_jobs:
                    result["metadata"].update(batch_file=self.batch_path.name, batch_index=self.batch_index + 1,
                                              batch_label=self.batch_jobs[self.batch_index]["label"])
                self.messages.put(("complete", result))
            except SimulationCancelled:
                self.messages.put(("cancelled", None))
            except Exception as exc:
                self.messages.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker = threading.Thread(target=worker, name="NBodySimulation", daemon=True)
        self.worker.start()

    def cancel_run(self):
        if self.running:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.progress_text.set("Cancelling; keeping the previous successful result.")

    def poll_worker(self):
        if self.closed:
            return
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "progress":
                    fraction, steps, h, elapsed = data
                    self.progress_var.set(100 * fraction)
                    if not self.cancel_event.is_set():
                        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else 0
                        self.progress_text.set(
                            (f"Batch {self.batch_index + 1}/{len(self.batch_jobs)} · " if self.batch_jobs else "") +
                            f"Progress {fraction:.1%} · {steps:,} steps · current dt = {h:.6g} · "
                            f"Elapsed {elapsed:.1f} s · ETA {eta:.1f} s")
                elif kind == "complete":
                    # A cancel arriving just after the final step should still preserve the old view.
                    if self.cancel_event.is_set():
                        self.set_running(False)
                        self.progress_text.set("Run cancelled; the previous successful result is retained.")
                        self.batch_finished_run(False, "Cancelled.")
                        continue
                    record_enabled = data["metadata"].get("record_enabled", True)
                    if record_enabled:
                        record = build_run_history_record(data["metadata"], data["diagnostics"])
                        record["result"] = data
                        record["save_full_data"] = bool(self.batch_jobs)
                        self.pending_history.append(record)
                    self.progress_text.set("Computation complete. Drawing results...")
                    self.root.update_idletasks()
                    container = ttk.Frame(self.result_host)
                    try:
                        new_viewer = NBodyViewer(root=container, timeline=self.timeline, **data)
                    except Exception as exc:
                        container.destroy()
                        self.set_running(False)
                        self.progress_text.set(f"Plotting failed: {exc}; the previous result is retained.")
                        if record_enabled:
                            self.history_status.set("The run awaits plotting and archiving. Click Retry to try again.")
                            self.retry_history_button.configure(state="normal")
                            self.retry_history_button.grid()
                        self.batch_finished_run(False, f"Plotting failed: {exc}")
                        messagebox.showerror("Plotting failed", str(exc), parent=self.root)
                        continue
                    saved = True
                    if record_enabled:
                        self.progress_text.set("Saving the run summary and conservation figure...")
                        saved = self.save_pending_history(viewer=new_viewer)
                    if self.viewer:
                        self.viewer.close()
                    self.placeholder.grid_remove()
                    container.grid(row=0, column=0, sticky="nsew")
                    self.viewer = new_viewer
                    self.set_running(False)
                    self.toggle_settings(False)
                    if not record_enabled:
                        self.progress_text.set("Simulation complete. Recording disabled; no history or image saved.")
                    self.batch_finished_run(saved, "Archive failed; use Retry saving run history before closing." if not saved else "")
                elif kind == "cancelled":
                    self.set_running(False)
                    self.batch_finished_run(False, "Cancelled.")
                    self.progress_text.set("Run cancelled; the previous successful result is retained.")
                elif kind == "error":
                    self.set_running(False)
                    self.progress_text.set("Run failed; the previous successful result is retained.")
                    self.batch_finished_run(False, data)
                    messagebox.showerror("Computation failed", data, parent=self.root)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(100, self.poll_worker)

    def close(self):
        self.closed = True
        self.cancel_event.set()
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        if self.viewer:
            self.viewer.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    NBodyApp().run()
