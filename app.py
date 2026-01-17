# app.py
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config / styling
# =========================
st.set_page_config(
    page_title="MentePC Compressor Health Analyser",
    layout="wide",
)

st.title("MentePC Compressor Health Analyser (WCMC / WCMD / WCME)")
st.caption(
    "Upload MentePC CSV logs (WCMC=ESA30E-25, WCMD=ESA30EH-25, WCME=ESA30EH2-25). "
    "Produces Red/Amber/Green indicators and trends."
)


# =========================
# Helpers
# =========================
WC_TYPES = ["AUTO", "WCMC", "WCMD", "WCME"]

DISPLAY_MAP = {
    "discharge_temp_c": "Discharge Temperature (°C)",
    "high_pressure_mpa": "High Pressure (MPa)",
    "low_pressure_mpa": "Low Pressure (MPa)",
    "pressure_ratio": "Pressure Ratio (HP/LP)",
    "current_a": "Current Draw (A)",
    "compressor_hz": "Compressor Speed (Hz)",
    "suction_superheat_c": "Suction Superheat (°C)",
    "compression_dT_c": "Compression Temp Rise ΔT (°C)",
    "error_code": "Error Code",
}

# Column patterns by file type.
# These are intentionally flexible; AUTO mode will pick best-matching.
MAPPING_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "WCMD": {
        "error_code": [r"^Error Code$"],
        "discharge_temp_c": [r"^ThoD1$"],
        "high_pressure_mpa": [r"^HP1$"],
        "low_pressure_mpa": [r"^LP1$"],
        "current_a": [r"^CT1$"],
        "compressor_hz": [r"^INV1 actual Hz$", r"INV1.*Hz", r"Actual.*Hz"],
        "suction_superheat_c": [r"^Comp 1 suction superheat$", r"Suction superheat", r"superheat"],
        # optional temperatures for ΔT
        "suction_temp_c": [r"^ThoS1$", r"Suct.*Temp", r"ThoS"],
    },
    "WCMC": {
        "error_code": [r"^Error Code$", r"Error\s*Code", r"Error"],
        "discharge_temp_c": [r"ThoD1", r"Disch.*Temp", r"Discharge.*Temp"],
        "high_pressure_mpa": [r"^HP1$", r"High.*Press", r"HP"],
        "low_pressure_mpa": [r"^LP1$", r"Low.*Press", r"LP"],
        "current_a": [r"^CT1$", r"Current", r"CT"],
        "compressor_hz": [r"INV1.*Hz", r"Actual.*Hz", r"Comp.*Hz", r"INV.*Hz"],
        "suction_superheat_c": [r"superheat", r"Suct.*SH"],
        "suction_temp_c": [r"ThoS1", r"Suct.*Temp", r"ThoS"],
    },
    "WCME": {
        "error_code": [r"^Error Code$", r"Error\s*Code", r"Error"],
        "discharge_temp_c": [r"ThoD1", r"Disch.*Temp", r"Discharge.*Temp"],
        "high_pressure_mpa": [r"^HP1$", r"High.*Press", r"HP"],
        "low_pressure_mpa": [r"^LP1$", r"Low.*Press", r"LP"],
        "current_a": [r"^CT1$", r"Current", r"CT"],
        "compressor_hz": [r"INV1.*Hz", r"Actual.*Hz", r"Comp.*Hz", r"INV.*Hz"],
        "suction_superheat_c": [r"superheat", r"Suct.*SH"],
        "suction_temp_c": [r"ThoS1", r"Suct.*Temp", r"ThoS"],
    },
    "AUTO": {
        # broad matching
        "error_code": [r"^Error Code$", r"Error\s*Code", r"^Error$"],
        "discharge_temp_c": [r"^ThoD1$", r"ThoD", r"Disch.*Temp", r"Discharge.*Temp"],
        "high_pressure_mpa": [r"^HP1$", r"High.*Press", r"\bHP\b"],
        "low_pressure_mpa": [r"^LP1$", r"Low.*Press", r"\bLP\b"],
        "current_a": [r"^CT1$", r"\bCT\b", r"Current"],
        "compressor_hz": [r"Hz", r"INV.*Hz", r"Actual.*Hz", r"Comp.*Hz"],
        "suction_superheat_c": [r"superheat", r"Suct.*SH"],
        "suction_temp_c": [r"ThoS1", r"Suct.*Temp", r"ThoS"],
    },
}


def detect_wc_type_from_filename(name: str) -> str:
    n = (name or "").upper()
    if "WCME" in n:
        return "WCME"
    if "WCMD" in n:
        return "WCMD"
    if "WCMC" in n:
        return "WCMC"
    return "AUTO"


def sniff_header_row(lines: List[str]) -> int:
    """
    MentePC CSV has metadata rows then a real header row.
    We find a likely header row by scoring candidate rows.
    """
    best_i = 0
    best_score = -1

    # Look only in the top section.
    for i, line in enumerate(lines[:150]):
        l = line.strip()
        if not l:
            continue

        # Heuristics: header rows tend to contain these tokens.
        score = 0
        if "Error Code" in l:
            score += 5
        if re.search(r"\bHP1\b", l):
            score += 3
        if re.search(r"\bLP1\b", l):
            score += 3
        if "ThoD" in l:
            score += 2
        if "CT" in l:
            score += 1
        if "Hz" in l:
            score += 1

        # also reward comma-separated with many fields
        comma_count = l.count(",")
        if comma_count >= 10:
            score += 2
        if comma_count >= 30:
            score += 3

        # Penalize very short lines
        if comma_count <= 1:
            score -= 3

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def read_mente_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Robust read: detect header row and parse.
    Handles odd encodings and metadata sections.
    """
    # Decode safely
    raw_text = file_bytes.decode("utf-8", errors="ignore")
    lines = raw_text.splitlines()

    header_idx = sniff_header_row(lines)

    # Read from detected header row
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        skiprows=header_idx,
        header=0,
        engine="python",
    )

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Drop rows that are entirely empty
    df = df.dropna(axis=0, how="all")

    return df


def find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for pat in patterns:
        rpat = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rpat.search(str(c).strip()):
                return c
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to float robustly:
    - handles comma decimals "12,3"
    - strips non-numeric characters
    """
    s = series.astype(str).str.strip()

    # Replace comma decimal with dot when it looks like decimal comma.
    # Example: "12,34" -> "12.34"
    s = s.str.replace(r"(?<=\d),(?=\d)", ".", regex=True)

    # Remove units/extra characters except digits, sign, dot, exponent
    s = s.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)

    out = pd.to_numeric(s, errors="coerce")
    return out


def detect_time_axis(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Returns df with a guaranteed 'Time' column and a label for axis type.
    If timestamp columns exist, parse; otherwise use sample index.
    """
    # Common column candidates in MentePC logs
    candidates = [c for c in df.columns if re.search(r"time|date", str(c), re.IGNORECASE)]

    # Try known patterns: "Date" + "Time"
    date_col = None
    time_col = None
    for c in candidates:
        if re.fullmatch(r"date", str(c).strip(), flags=re.IGNORECASE):
            date_col = c
        if re.fullmatch(r"time", str(c).strip(), flags=re.IGNORECASE):
            time_col = c

    df2 = df.copy()

    if date_col and time_col:
        dt = (df2[date_col].astype(str).str.strip() + " " + df2[time_col].astype(str).str.strip())
        df2["Time"] = pd.to_datetime(dt, errors="coerce", dayfirst=False)
        if df2["Time"].notna().sum() >= max(5, int(0.2 * len(df2))):
            return df2, "timestamp(Date+Time)"

    # Single combined timestamp column
    combined = None
    for c in candidates:
        if re.search(r"stamp|timestamp|date\s*time|time\s*stamp", str(c), re.IGNORECASE):
            combined = c
            break
    if combined:
        df2["Time"] = pd.to_datetime(df2[combined], errors="coerce", dayfirst=False)
        if df2["Time"].notna().sum() >= max(5, int(0.2 * len(df2))):
            return df2, f"timestamp({combined})"

    # Otherwise sample index
    df2["Time"] = np.arange(len(df2), dtype=float)
    return df2, "sample_index"


@dataclass
class SignalPack:
    df: pd.DataFrame
    wc_type: str
    time_axis_kind: str
    cols: Dict[str, Optional[str]]  # canonical -> source col


def build_signal_pack(df_raw: pd.DataFrame, wc_type: str) -> SignalPack:
    # Ensure Time column exists
    df, time_kind = detect_time_axis(df_raw)

    patterns = MAPPING_PATTERNS.get(wc_type, MAPPING_PATTERNS["AUTO"])

    cols = {}
    for canon, pats in patterns.items():
        cols[canon] = find_column(df, pats)

    # If AUTO, attempt to improve by scoring which known type fits best
    if wc_type == "AUTO":
        best_type = "AUTO"
        best_hits = -1
        for t in ["WCMC", "WCMD", "WCME"]:
            pats_t = MAPPING_PATTERNS[t]
            hits = 0
            for canon, pats in pats_t.items():
                if canon in ("suction_temp_c",):
                    continue
                if find_column(df, pats) is not None:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_type = t
        # Rebuild mapping with best_type if it’s materially better
        if best_type != "AUTO" and best_hits >= 4:
            wc_type = best_type
            patterns = MAPPING_PATTERNS[wc_type]
            cols = {}
            for canon, pats in patterns.items():
                cols[canon] = find_column(df, pats)

    return SignalPack(df=df, wc_type=wc_type, time_axis_kind=time_kind, cols=cols)


def extract_signals(pack: SignalPack) -> pd.DataFrame:
    """
    Return a tidy dataframe with canonical numeric columns + Time + running flag.
    """
    df = pack.df.copy()

    out = pd.DataFrame()
    out["Time"] = df["Time"]

    # error code (keep as string)
    if pack.cols.get("error_code"):
        out["error_code"] = df[pack.cols["error_code"]].astype(str).str.strip()
    else:
        out["error_code"] = ""

    # numeric signals
    def add_num(canon: str):
        src = pack.cols.get(canon)
        if src and src in df.columns:
            out[canon] = coerce_numeric(df[src])
        else:
            out[canon] = np.nan

    add_num("discharge_temp_c")
    add_num("high_pressure_mpa")
    add_num("low_pressure_mpa")
    add_num("current_a")
    add_num("compressor_hz")
    add_num("suction_superheat_c")

    # Optional suction temp for ΔT (if available)
    suction_temp = np.nan
    if pack.cols.get("suction_temp_c") and pack.cols["suction_temp_c"] in df.columns:
        suction_temp = coerce_numeric(df[pack.cols["suction_temp_c"]])
    # Compression temperature rise approximation
    out["compression_dT_c"] = out["discharge_temp_c"] - suction_temp

    # Derived: pressure ratio
    out["pressure_ratio"] = out["high_pressure_mpa"] / out["low_pressure_mpa"]

    # Running flag: compressor considered running when Hz is available and > ~1 Hz
    # fallback: if Hz missing, use current draw
    running = pd.Series(False, index=out.index)
    if out["compressor_hz"].notna().sum() > 0:
        running = out["compressor_hz"] > 1.0
    elif out["current_a"].notna().sum() > 0:
        running = out["current_a"] > 1.0
    out["running"] = running

    return out


def basic_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    return {
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std(ddof=0)),
    }


def pct(x: float) -> str:
    if np.isnan(x):
        return "n/a"
    return f"{x*100:.1f}%"


def safe_in_range(val: float, lo: float, hi: float) -> bool:
    return (not np.isnan(val)) and (lo <= val <= hi)


@dataclass
class StatusResult:
    status: str  # GREEN/AMBER/RED/UNKNOWN
    details: str
    stats: Dict[str, float]
    time_in_red: float


def rag_from_thresholds(
    key: str,
    series: pd.Series,
    running: pd.Series,
    baseline_series: Optional[pd.Series] = None,
) -> StatusResult:
    """
    Default rules inspired by your failure report approach:
    - pressure ratio, discharge temp, superheat absolute bands
    - baseline deviation and variability collapse if baseline provided
    - time-in-red estimated
    """
    s = series.copy()

    # Evaluate primarily while running (where applicable)
    if key in ("pressure_ratio", "discharge_temp_c", "suction_superheat_c"):
        s_eval = s[running.values]
    else:
        s_eval = s

    stt = basic_stats(s_eval)
    mean = stt["mean"]
    std = stt["std"]

    # default
    status = "UNKNOWN"
    red_mask = pd.Series(False, index=s_eval.index)

    def set_status(new: str):
        nonlocal status
        order = {"UNKNOWN": 0, "GREEN": 1, "AMBER": 2, "RED": 3}
        if order[new] > order[status]:
            status = new

    # Absolute thresholds
    if key == "pressure_ratio":
        if np.isnan(mean):
            status = "UNKNOWN"
        else:
            if mean < 1.5 or (s_eval.dropna() < 1.5).any():
                set_status("RED")
            elif mean < 1.8 or mean > 3.5:
                set_status("AMBER")
            else:
                set_status("GREEN")
        red_mask = s_eval < 1.5

        details = "HP/LP < 1.5 is critical; < 1.0 often indicates no effective compression or sensor/control anomaly."

    elif key == "discharge_temp_c":
        if np.isnan(mean):
            status = "UNKNOWN"
        else:
            # while running
            if (s_eval.dropna() < 80).any() or (s_eval.dropna() > 120).any():
                set_status("RED")
            elif mean < 85 or mean > 115:
                set_status("AMBER")
            else:
                set_status("GREEN")
        red_mask = (s_eval < 80) | (s_eval > 120)
        details = "Evaluated primarily while compressor is running. Low discharge temp while running can indicate limitation or sensor issues."

    elif key == "suction_superheat_c":
        if np.isnan(mean):
            status = "UNKNOWN"
        else:
            if (s_eval.dropna() < 4).any() or (s_eval.dropna() > 18).any():
                set_status("RED")
            elif mean < 5 or mean > 15:
                set_status("AMBER")
            else:
                set_status("GREEN")
        red_mask = (s_eval < 4) | (s_eval > 18)
        details = "<4°C floodback risk; >18°C indicates feeding/control instability."

    elif key == "current_a":
        # Baseline-driven if available; otherwise show GREEN unless clearly abnormal
        details = "Prefer baseline comparison. Large deviation or near-zero current while 'running' indicates limitation/fault."
        if s_eval.dropna().empty:
            status = "UNKNOWN"
        else:
            set_status("GREEN")
            # hard red: near zero while running
            if (s[running.values].dropna() < 1.0).any():
                set_status("RED")
            red_mask = (s[running.values] < 1.0).reindex(s_eval.index, fill_value=False)

    elif key == "compressor_hz":
        details = "Context metric. Near-zero Hz while demand exists or while system should run may indicate control limitation/fault."
        if s_eval.dropna().empty:
            status = "UNKNOWN"
        else:
            set_status("GREEN")
            if (s.dropna() < 1.0).mean() > 0.3:
                set_status("AMBER")
            if (s.dropna() < 1.0).mean() > 0.7:
                set_status("RED")
        red_mask = s_eval < 1.0

    elif key in ("high_pressure_mpa", "low_pressure_mpa"):
        details = "Prefer baseline comparison for pressure deviation. Also check for 'frozen' values (variability collapse)."
        if s_eval.dropna().empty:
            status = "UNKNOWN"
        else:
            set_status("GREEN")
        red_mask = pd.Series(False, index=s_eval.index)

    else:
        details = "Computed metric."
        status = "UNKNOWN"

    # Baseline deviation checks
    if baseline_series is not None:
        b = baseline_series.copy()
        # align sizes loosely by dropping NaNs and using mean/std only
        b_stats = basic_stats(b)
        b_mean = b_stats["mean"]
        b_std = b_stats["std"]

        # Deviation from baseline mean
        if not np.isnan(mean) and not np.isnan(b_mean) and b_mean != 0:
            dev = abs(mean - b_mean) / abs(b_mean)
            if dev > 0.30:
                set_status("RED")
            elif dev > 0.20:
                set_status("AMBER")

        # Variability collapse check (report-style): std drops massively vs baseline std
        if not np.isnan(std) and not np.isnan(b_std) and b_std > 0:
            ratio = std / b_std
            # collapse threshold bands
            if ratio < 0.10:
                set_status("RED")
            elif ratio < 0.35:
                set_status("AMBER")

    # Time in red
    if s_eval.dropna().empty:
        time_in_red = np.nan
    else:
        time_in_red = float(red_mask.dropna().mean())

    return StatusResult(status=status, details=details, stats=stt, time_in_red=time_in_red)


def overall_from_individual(results: Dict[str, StatusResult]) -> str:
    order = {"UNKNOWN": 0, "GREEN": 1, "AMBER": 2, "RED": 3}
    best = "UNKNOWN"
    for r in results.values():
        if order[r.status] > order[best]:
            best = r.status
    # Conservative: if many UNKNOWNs but no RED/AMBER, show GREEN
    if best == "UNKNOWN":
        return "GREEN"
    return best


def badge(label: str, status: str):
    color = {
        "GREEN": "#1f7a1f",
        "AMBER": "#a67c00",
        "RED": "#b00020",
        "UNKNOWN": "#666666",
    }.get(status, "#666666")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:6px 14px;border-radius:999px;
                    background:{color};color:white;font-weight:700;font-size:13px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UI: file upload
# =========================
st.sidebar.header("Upload MentePC CSV files")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more MentePC CSV logs",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload CSV files to begin (WCMC / WCMD / WCME).")
    st.stop()

# Build a name->bytes map
file_map = {f.name: f.getvalue() for f in uploaded_files}

baseline_name = st.sidebar.selectbox(
    "Select baseline file (healthy reference)",
    options=["(none)"] + list(file_map.keys()),
    index=0,
)

target_name = st.sidebar.selectbox(
    "Select file to analyse now",
    options=list(file_map.keys()),
    index=0,
)

override_type = st.sidebar.selectbox(
    "Override mapping type for target (if detection wrong)",
    options=WC_TYPES,
    index=0,
)

override_base_type = st.sidebar.selectbox(
    "Override mapping type for baseline (optional)",
    options=WC_TYPES,
    index=0,
)

st.sidebar.caption(
    "Tip: If a file is mis-detected, override mapping type. "
    "WCMC=ESA30E-25, WCMD=ESA30EH-25, WCME=ESA30EH2-25."
)


# =========================
# Read / parse
# =========================
@st.cache_data(show_spinner=False)
def load_pack(name: str, raw: bytes, forced_type: str) -> Tuple[SignalPack, pd.DataFrame]:
    df_raw = read_mente_csv(raw)
    wc_type = forced_type if forced_type != "AUTO" else detect_wc_type_from_filename(name)
    pack = build_signal_pack(df_raw, wc_type)
    signals = extract_signals(pack)
    return pack, signals


with st.spinner("Parsing and analysing…"):
    target_pack, target_sig = load_pack(target_name, file_map[target_name], override_type)

    baseline_pack = None
    baseline_sig = None
    if baseline_name != "(none)":
        baseline_pack, baseline_sig = load_pack(baseline_name, file_map[baseline_name], override_base_type)


# =========================
# Analyse
# =========================
KEYS_TO_SCORE = [
    "pressure_ratio",
    "discharge_temp_c",
    "suction_superheat_c",
    "high_pressure_mpa",
    "low_pressure_mpa",
    "current_a",
    "compressor_hz",
]

results: Dict[str, StatusResult] = {}

for key in KEYS_TO_SCORE:
    base_series = None
    if baseline_sig is not None and key in baseline_sig.columns:
        base_series = baseline_sig[key]
    res = rag_from_thresholds(
        key=key,
        series=target_sig.get(key, pd.Series(dtype=float)),
        running=target_sig["running"],
        baseline_series=base_series,
    )
    results[key] = res

overall = overall_from_individual(results)

# =========================
# Header summary
# =========================
st.markdown(
    f"## Overall Compressor Risk (Target: {target_name} | Mapping: {target_pack.wc_type} | Time axis: {target_pack.time_axis_kind})"
)
badge(overall, overall)
st.write("")

# =========================
# Parameter cards
# =========================
st.markdown("## Parameter Status (Red/Amber/Green)")

card_cols = st.columns(3)
card_keys = ["pressure_ratio", "discharge_temp_c", "suction_superheat_c"]

for i, k in enumerate(card_keys):
    r = results[k]
    with card_cols[i]:
        badge(r.status, r.status)
        st.subheader(DISPLAY_MAP.get(k, k))
        st.caption(
            f"mean {r.stats['mean']:.3g} | std {r.stats['std']:.3g} | min {r.stats['min']:.3g} | max {r.stats['max']:.3g}"
            if not np.isnan(r.stats["mean"])
            else "No data"
        )
        st.write(r.details)
        if not np.isnan(r.time_in_red):
            st.caption(f"Time in RED (approx): {pct(r.time_in_red)}")

st.write("---")

grid_cols = st.columns(3)
grid_keys = ["high_pressure_mpa", "low_pressure_mpa", "current_a", "compressor_hz"]

for idx, k in enumerate(grid_keys):
    r = results[k]
    with grid_cols[idx % 3]:
        st.subheader(f"{DISPLAY_MAP.get(k, k)} vs Baseline")
        badge(r.status, r.status)
        if not np.isnan(r.stats["mean"]):
            st.caption(f"mean {r.stats['mean']:.3g} | std {r.stats['std']:.3g}")
        else:
            st.caption("No data")
        if baseline_sig is None:
            st.caption("Select baseline to enable deviation/variability checks.")
        if not np.isnan(r.time_in_red) and r.time_in_red > 0:
            st.caption(f"Time in RED (approx): {pct(r.time_in_red)}")

st.write("---")

# =========================
# Diagnostics / mapping view
# =========================
with st.expander("Mapping diagnostics (what columns were detected)"):
    st.write("**Target mapping**")
    st.json(target_pack.cols)
    if baseline_pack is not None:
        st.write("**Baseline mapping**")
        st.json(baseline_pack.cols)

    st.write("**Detected columns in target file**")
    st.write(list(target_pack.df.columns)[:60])
    if len(target_pack.df.columns) > 60:
        st.caption(f"(showing first 60 of {len(target_pack.df.columns)})")


# =========================
# Trends (safe plotting)
# =========================
st.markdown("## Trends")

# Build a plot dataframe with canonical keys that exist
plot_cols = [c for c in [
    "discharge_temp_c",
    "high_pressure_mpa",
    "low_pressure_mpa",
    "pressure_ratio",
    "current_a",
    "compressor_hz",
    "suction_superheat_c",
] if c in target_sig.columns]

plot_df = target_sig[["Time"] + plot_cols].copy()

# Convert Time to something streamlit accepts consistently:
# - keep datetime if already datetime
# - else numeric
if pd.api.types.is_datetime64_any_dtype(plot_df["Time"]):
    pass
else:
    plot_df["Time"] = pd.to_numeric(plot_df["Time"], errors="coerce")

if plot_df.empty or len(plot_cols) == 0:
    st.info("No plot-capable signals found with the current mapping.")
else:
    # One chart per parameter, with Altair-safe field name "value"
    for key in plot_cols:
        label = DISPLAY_MAP.get(key, key)
        df_plot = plot_df[["Time", key]].copy().dropna()
        if df_plot.empty:
            continue

        df_plot = df_plot.rename(columns={key: "value"})

        st.markdown(f"**{label}**")
        # Explicit x/y avoids shorthand parsing failures.
        st.line_chart(df_plot, x="Time", y="value", height=170)


# =========================
# Export / raw view
# =========================
st.markdown("## Export / Raw")

c1, c2 = st.columns([1, 1])

with c1:
    st.download_button(
        "Download analysed signals (CSV)",
        data=target_sig.to_csv(index=False).encode("utf-8"),
        file_name=f"{target_name}_analysed.csv",
        mime="text/csv",
    )

with c2:
    # Minimal summary export
    summary_lines = []
    summary_lines.append(f"Target: {target_name}")
    summary_lines.append(f"Mapping: {target_pack.wc_type}")
    summary_lines.append(f"Overall: {overall}")
    summary_lines.append("")
    for k, r in results.items():
        summary_lines.append(f"{DISPLAY_MAP.get(k, k)}: {r.status} | mean={r.stats['mean']:.4g} std={r.stats['std']:.4g} red%={r.time_in_red*100 if not np.isnan(r.time_in_red) else np.nan:.3g}")
    summary_txt = "\n".join(summary_lines)

    st.download_button(
        "Download summary (TXT)",
        data=summary_txt.encode("utf-8"),
        file_name=f"{target_name}_summary.txt",
        mime="text/plain",
    )

with st.expander("Raw analysed table (preview)"):
    st.dataframe(target_sig.head(200), use_container_width=True)
