# app.py
# Streamlit web app: Mente PC Compressor Risk Monitor (WCMC / WCMD / WCME)
# Adds: timestamp support (e.g., "Unnamed: 0" datetime column)

import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Defaults: column mappings by file type
# -----------------------------
DEFAULT_MAPPINGS_BY_TYPE: Dict[str, Dict[str, List[str]]] = {
    "WCMD": {
        "timestamp": [r"^Unnamed:\s*0$", r"^Time$", r"^Timestamp$", r"^Date$", r"^Datetime$"],
        "discharge_temp_c": [r"^ThoD1$"],
        "high_pressure_mpa": [r"^HP1$"],
        "low_pressure_mpa": [r"^LP1$"],
        "current_a": [r"^CT1$"],
        "compressor_hz": [r"^INV1 actual Hz$", r"^INV1.*Hz$"],
        "suction_superheat_c": [r"^Comp 1 suction superheat$", r"suction superheat"],
        "error_code": [r"^Error Code$", r"error.*code"],
    },
    "WCMC": {
        "timestamp": [r"^Unnamed:\s*0$", r"^Time$", r"^Timestamp$", r"^Date$", r"^Datetime$"],
        "discharge_temp_c": [r"^ThoD1$", r"discharge.*temp", r"comp.*disch.*t"],
        "high_pressure_mpa": [r"^HP1$", r"high.*press", r"\bHP\b"],
        "low_pressure_mpa": [r"^LP1$", r"low.*press", r"\bLP\b"],
        "current_a": [r"^CT1$", r"current", r"\bCT\b", r"amp"],
        "compressor_hz": [r"^INV1 actual Hz$", r"\bHz\b", r"compressor.*speed", r"inv.*freq"],
        "suction_superheat_c": [r"^Comp 1 suction superheat$", r"superheat", r"\bSH\b"],
        "error_code": [r"^Error Code$", r"error.*code"],
    },
    "WCME": {
        "timestamp": [r"^Unnamed:\s*0$", r"^Time$", r"^Timestamp$", r"^Date$", r"^Datetime$"],
        "discharge_temp_c": [r"^ThoD1$", r"discharge.*temp", r"comp.*disch.*t"],
        "high_pressure_mpa": [r"^HP1$", r"high.*press", r"\bHP\b", r"gas.*cooler.*press"],
        "low_pressure_mpa": [r"^LP1$", r"low.*press", r"\bLP\b", r"suction.*press"],
        "current_a": [r"^CT1$", r"current", r"\bCT\b", r"amp"],
        "compressor_hz": [r"^INV1 actual Hz$", r"\bHz\b", r"compressor.*speed", r"inv.*freq"],
        "suction_superheat_c": [r"^Comp 1 suction superheat$", r"superheat", r"\bSH\b"],
        "error_code": [r"^Error Code$", r"error.*code"],
    },
    "AUTO": {
        "timestamp": [r"^Unnamed:\s*0$", r"^Time$", r"^Timestamp$", r"^Date$", r"^Datetime$"],
        "discharge_temp_c": [r"^ThoD1$", r"discharge.*temp", r"tho.*d", r"tdis"],
        "high_pressure_mpa": [r"^HP1$", r"high.*press", r"\bHP\b", r"pdis"],
        "low_pressure_mpa": [r"^LP1$", r"low.*press", r"\bLP\b", r"psuc"],
        "current_a": [r"^CT1$", r"current", r"\bCT\b", r"amp"],
        "compressor_hz": [r"INV.*Hz", r"\bHz\b", r"compressor.*speed", r"inv.*freq"],
        "suction_superheat_c": [r"superheat", r"\bSH\b"],
        "error_code": [r"^Error Code$", r"error.*code"],
    },
}


# -----------------------------
# Thresholds (RAG scoring)
# -----------------------------
DEFAULT_THRESHOLDS = {
    "pressure_ratio": {
        "green": [1.8, 3.5],
        "amber": [[1.5, 1.8], [3.5, 4.0]],
        "red": [[-1e9, 1.5], [4.0, 1e9]],
        "notes": "HP/LP. <1.5 is critical; <1.0 often indicates no effective compression or sensor/control anomaly."
    },
    "discharge_temp_c": {
        "green": [85, 115],
        "amber": [[80, 85], [115, 120]],
        "red": [[-1e9, 80], [120, 1e9]],
        "notes": "Evaluated primarily while compressor is running. Low discharge temp while running can indicate limitation or sensor issues."
    },
    "suction_superheat_c": {
        "green": [5, 15],
        "amber": [[4, 5], [15, 18]],
        "red": [[-1e9, 4], [18, 1e9]],
        "notes": "<4°C floodback risk; >18°C indicates feeding/control instability."
    },
    "compressor_hz": {
        "running_hz": 10,
        "notes": "Used as a gating/context signal for 'compressor running'."
    },
    "current_a": {"green_dev_pct": 10, "amber_dev_pct": 20, "red_dev_pct": 30, "notes": "Deviation vs baseline mean (if baseline provided)."},
    "high_pressure_mpa": {"green_dev_pct": 10, "amber_dev_pct": 20, "red_dev_pct": 30, "notes": "Deviation vs baseline mean (if baseline provided)."},
    "low_pressure_mpa": {"green_dev_pct": 10, "amber_dev_pct": 20, "red_dev_pct": 30, "notes": "Deviation vs baseline mean (if baseline provided)."},
    "compressor_hz_dev": {"green_dev_pct": 10, "amber_dev_pct": 20, "red_dev_pct": 30, "notes": "Deviation vs baseline mean (optional)."},
    "pressure_ratio_dev": {"green_dev_pct": 10, "amber_dev_pct": 20, "red_dev_pct": 30, "notes": "Deviation vs baseline mean (optional)."},
    "variability_collapse": {
        "amber_std_drop_pct": 80,
        "red_std_drop_pct": 90,
        "notes": "If std-dev collapses vs baseline while compressor is running, flag possible sensor freeze or control limiting."
    }
}


@dataclass
class MetricResult:
    name: str
    status: str  # GREEN / AMBER / RED / UNKNOWN
    value_summary: str
    why: str
    time_in_red_pct: float
    extra: Optional[str] = None


def status_color(status: str) -> str:
    return {"GREEN": "#1a7f37", "AMBER": "#b58100", "RED": "#c1121f", "UNKNOWN": "#6c757d"}.get(status, "#6c757d")


def render_badge(text: str, status: str) -> None:
    st.markdown(
        f"""
        <div style="display:inline-block;padding:6px 10px;border-radius:999px;
                    background:{status_color(status)};color:white;font-weight:700;font-size:13px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def decode_lines(file_bytes: bytes) -> List[str]:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    for enc in encodings:
        try:
            return file_bytes.decode(enc, errors="strict").splitlines()
        except Exception:
            pass
    return file_bytes.decode("utf-8", errors="ignore").splitlines()


def sniff_delimiter(header_line: str) -> str:
    if header_line.count(";") > header_line.count(","):
        return ";"
    return ","


def find_header_row(lines: List[str]) -> int:
    signal_tokens = ["HP", "LP", "CT", "Hz", "Tho"]
    for i, line in enumerate(lines[:400]):
        if "Error Code" in line and any(tok in line for tok in signal_tokens):
            return i
    common_candidates = ["HP1", "LP1", "CT1", "INV", "Hz", "ThoD1", "superheat"]
    for i, line in enumerate(lines[:400]):
        hits = sum(1 for tok in common_candidates if tok in line)
        if hits >= 3:
            return i
    return 0


def read_mente_pc_any_wc(file_bytes: bytes) -> pd.DataFrame:
    lines = decode_lines(file_bytes)
    header_idx = find_header_row(lines)
    delimiter = sniff_delimiter(lines[header_idx]) if header_idx < len(lines) else ","
    df = pd.read_csv(io.BytesIO(file_bytes), skiprows=header_idx, header=0, sep=delimiter, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_wc_type(filename: str, df: pd.DataFrame) -> str:
    fn = filename.upper()
    if "WCMC" in fn:
        return "WCMC"
    if "WCMD" in fn:
        return "WCMD"
    if "WCME" in fn:
        return "WCME"
    cols = " | ".join(df.columns).upper()
    if "WCME" in cols:
        return "WCME"
    if "WCMC" in cols:
        return "WCMC"
    if "WCMD" in cols:
        return "WCMD"
    return "AUTO"


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in df.columns:
            if rx.search(str(c).strip()):
                return c
    return None


def extract_canonical(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> Tuple[Dict[str, pd.Series], Optional[pd.Series]]:
    out: Dict[str, pd.Series] = {}
    ts_col = find_column(df, mapping.get("timestamp", []))
    ts_series = None
    if ts_col:
        ts_series = pd.to_datetime(df[ts_col], errors="coerce")
    for canon, pats in mapping.items():
        if canon == "timestamp":
            continue
        col = find_column(df, pats)
        if col is None:
            continue
        if canon == "error_code":
            out[canon] = df[col].astype(str)
        else:
            out[canon] = coerce_numeric(df[col])
    return out, ts_series


def classify_band(value: float, green_range: List[float], amber_ranges: List[List[float]], red_ranges: List[List[float]]) -> str:
    if value is None or np.isnan(value):
        return "UNKNOWN"
    if green_range[0] <= value <= green_range[1]:
        return "GREEN"
    for a in amber_ranges:
        if a[0] <= value <= a[1]:
            return "AMBER"
    for r in red_ranges:
        if r[0] <= value <= r[1]:
            return "RED"
    return "AMBER"


def pct_dev(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b) or np.isnan(a):
        return np.nan
    return abs(a - b) / abs(b) * 100.0


def compute_baseline_stats(series: Dict[str, pd.Series], thresholds: Dict) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    hz = series.get("compressor_hz")
    running_hz = thresholds["compressor_hz"]["running_hz"]
    running_mask = (hz.fillna(0) >= running_hz) if hz is not None else None

    for key in ["current_a", "high_pressure_mpa", "low_pressure_mpa", "compressor_hz", "discharge_temp_c", "suction_superheat_c"]:
        s = series.get(key)
        if s is None:
            continue
        s_use = s[running_mask] if running_mask is not None else s
        stats[key] = {"mean": float(np.nanmean(s_use)), "std": float(np.nanstd(s_use))}

    hp = series.get("high_pressure_mpa")
    lp = series.get("low_pressure_mpa")
    if hp is not None and lp is not None:
        pr = hp / lp.replace(0, np.nan)
        s_use = pr[running_mask] if running_mask is not None else pr
        stats["pressure_ratio"] = {"mean": float(np.nanmean(s_use)), "std": float(np.nanstd(s_use))}
    return stats


def compute_results(series: Dict[str, pd.Series], thresholds: Dict, baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
                   ) -> Tuple[List[MetricResult], str, List[str], pd.DataFrame]:
    results: List[MetricResult] = []
    reasons: List[str] = []

    hz = series.get("compressor_hz")
    running_hz = thresholds["compressor_hz"]["running_hz"]
    running_mask = (hz.fillna(0) >= running_hz) if hz is not None else None
    running_pct = float(running_mask.mean() * 100.0) if running_mask is not None else np.nan

    hp = series.get("high_pressure_mpa")
    lp = series.get("low_pressure_mpa")
    pr = None
    if hp is not None and lp is not None:
        pr = hp / lp.replace(0, np.nan)
        series["pressure_ratio"] = pr

    plot_df = pd.DataFrame()
    for k in ["discharge_temp_c", "high_pressure_mpa", "low_pressure_mpa", "pressure_ratio", "current_a", "compressor_hz", "suction_superheat_c"]:
        if k in series:
            plot_df[k] = series[k].reset_index(drop=True)

    def time_in_red(status_per_sample: pd.Series) -> float:
        if status_per_sample is None or len(status_per_sample) == 0:
            return 0.0
        return float((status_per_sample == "RED").mean() * 100.0)

    # Pressure ratio absolute
    if pr is not None:
        pr_use = pr[running_mask] if running_mask is not None else pr
        mean_val = float(np.nanmean(pr_use))
        band = classify_band(mean_val, thresholds["pressure_ratio"]["green"], thresholds["pressure_ratio"]["amber"], thresholds["pressure_ratio"]["red"])
        per_sample = pr_use.apply(lambda v: classify_band(v, thresholds["pressure_ratio"]["green"], thresholds["pressure_ratio"]["amber"], thresholds["pressure_ratio"]["red"]))
        red_pct = time_in_red(per_sample)
        if band != "GREEN":
            reasons.append(f"Pressure ratio abnormal: mean={mean_val:.2f} ({band}).")
        results.append(MetricResult("Pressure Ratio (HP/LP)", band, f"mean {mean_val:.2f} | running {running_pct:.0f}%", thresholds["pressure_ratio"]["notes"], red_pct))

    # Discharge temp
    dt = series.get("discharge_temp_c")
    if dt is not None:
        dt_use = dt[running_mask] if running_mask is not None else dt
        mean_val = float(np.nanmean(dt_use))
        band = classify_band(mean_val, thresholds["discharge_temp_c"]["green"], thresholds["discharge_temp_c"]["amber"], thresholds["discharge_temp_c"]["red"])
        per_sample = dt_use.apply(lambda v: classify_band(v, thresholds["discharge_temp_c"]["green"], thresholds["discharge_temp_c"]["amber"], thresholds["discharge_temp_c"]["red"]))
        red_pct = time_in_red(per_sample)
        if band != "GREEN":
            reasons.append(f"Discharge temp abnormal: mean={mean_val:.1f}°C ({band}).")
        results.append(MetricResult("Discharge Temperature (°C)", band, f"mean {mean_val:.1f} | min {np.nanmin(dt_use):.1f} | max {np.nanmax(dt_use):.1f}",
                                    thresholds["discharge_temp_c"]["notes"], red_pct))

    # Suction superheat
    sh = series.get("suction_superheat_c")
    if sh is not None:
        sh_use = sh[running_mask] if running_mask is not None else sh
        mean_val = float(np.nanmean(sh_use))
        band = classify_band(mean_val, thresholds["suction_superheat_c"]["green"], thresholds["suction_superheat_c"]["amber"], thresholds["suction_superheat_c"]["red"])
        per_sample = sh_use.apply(lambda v: classify_band(v, thresholds["suction_superheat_c"]["green"], thresholds["suction_superheat_c"]["amber"], thresholds["suction_superheat_c"]["red"]))
        red_pct = time_in_red(per_sample)
        if band != "GREEN":
            reasons.append(f"Suction superheat abnormal: mean={mean_val:.1f}°C ({band}).")
        results.append(MetricResult("Suction Superheat (°C)", band, f"mean {mean_val:.1f} | std {np.nanstd(sh_use):.2f} | min {np.nanmin(sh_use):.1f} | max {np.nanmax(sh_use):.1f}",
                                    thresholds["suction_superheat_c"]["notes"], red_pct))

    # Baseline deviations (HP/LP/CT/Hz) + variability collapse
    def baseline_metric(key: str, label: str, cfg_key: str) -> None:
        s = series.get(key)
        if s is None:
            return
        s_use = s[running_mask] if running_mask is not None else s
        mean_now = float(np.nanmean(s_use))
        std_now = float(np.nanstd(s_use))

        if not baseline_stats or key not in baseline_stats:
            results.append(MetricResult(label, "UNKNOWN", f"mean {mean_now:.3f} | std {std_now:.3f} (no baseline)", "Select baseline to enable deviation checks.", 0.0))
            return

        base_mean = baseline_stats[key]["mean"]
        base_std = baseline_stats[key]["std"]
        dev = pct_dev(mean_now, base_mean)
        cfg = thresholds[cfg_key]

        if np.isnan(dev):
            band = "UNKNOWN"
        elif dev <= cfg["green_dev_pct"]:
            band = "GREEN"
        elif dev <= cfg["amber_dev_pct"]:
            band = "AMBER"
        else:
            band = "RED" if dev >= cfg["red_dev_pct"] else "AMBER"

        if base_std and base_std > 0:
            std_drop_pct = (1 - (std_now / base_std)) * 100.0
            if std_drop_pct >= thresholds["variability_collapse"]["red_std_drop_pct"]:
                band = "RED"
                reasons.append(f"{label} variability collapse ~{std_drop_pct:.0f}% vs baseline.")
            elif std_drop_pct >= thresholds["variability_collapse"]["amber_std_drop_pct"] and band == "GREEN":
                band = "AMBER"
                reasons.append(f"{label} variability drop ~{std_drop_pct:.0f}% vs baseline.")

        if band != "GREEN":
            reasons.append(f"{label} deviation {dev:.0f}% vs baseline.")

        results.append(MetricResult(label, band, f"mean {mean_now:.3f} | base {base_mean:.3f} | dev {dev:.0f}% | std now {std_now:.3f} / base {base_std:.3f}",
                                    cfg["notes"], 0.0, "Requires baseline selection."))

    baseline_metric("high_pressure_mpa", "High Pressure (MPa) vs Baseline", "high_pressure_mpa")
    baseline_metric("low_pressure_mpa", "Low Pressure (MPa) vs Baseline", "low_pressure_mpa")
    baseline_metric("current_a", "Current Draw (A) vs Baseline", "current_a")
    if hz is not None:
        baseline_metric("compressor_hz", "Compressor Speed (Hz) vs Baseline", "compressor_hz_dev")

    statuses = [r.status for r in results if r.status != "UNKNOWN"]
    if "RED" in statuses:
        overall = "RED"
    elif "AMBER" in statuses:
        overall = "AMBER"
    elif len(statuses) == 0:
        overall = "UNKNOWN"
    else:
        overall = "GREEN"

    top_reasons = list(dict.fromkeys(reasons))[:10]
    return results, overall, top_reasons, plot_df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mente PC Compressor Risk (Web)", layout="wide")

st.title("Mente PC Compressor Risk Monitor (Web)")
st.caption("Upload Mente PC CSV logs (WCMC / WCMD / WCME). The tool auto-detects header rows and model type. "
           "Diagnostic aid only; not a guarantee of failure prediction.")

with st.expander("Configuration (thresholds + mappings)", expanded=False):
    thresholds_text = st.text_area("Thresholds (JSON)", value=json.dumps(DEFAULT_THRESHOLDS, indent=2), height=260)
    mappings_text = st.text_area("Mappings by Type (JSON)", value=json.dumps(DEFAULT_MAPPINGS_BY_TYPE, indent=2), height=260)
    THRESHOLDS = json.loads(thresholds_text)
    MAPPINGS_BY_TYPE = json.loads(mappings_text)

uploads = st.file_uploader("Upload one or more Mente PC CSV files", type=["csv"], accept_multiple_files=True)
if not uploads:
    st.info("Upload CSVs to begin.")
    st.stop()

parsed: List[Tuple[str, str, pd.DataFrame]] = []
errors: List[str] = []

for f in uploads:
    try:
        df = read_mente_pc_any_wc(f.getvalue())
        wc_type = detect_wc_type(f.name, df)
        parsed.append((f.name, wc_type, df))
    except Exception as e:
        errors.append(f"{f.name}: {e}")

if errors:
    st.error("Some files could not be parsed:\n- " + "\n- ".join(errors))

if not parsed:
    st.error("No readable Mente PC files detected.")
    st.stop()

st.subheader("Detected files")
st.dataframe(pd.DataFrame([{
    "file": n,
    "detected_type": t,
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "example_cols": ", ".join(df.columns[:8])
} for (n, t, df) in parsed]), use_container_width=True)

names = [n for (n, _, _) in parsed]
baseline_name = st.selectbox("Select baseline file (healthy reference)", options=["(none)"] + names, index=0)
target_name = st.selectbox("Select file to analyse now", options=names, index=0)

target_type = next(t for (n, t, df) in parsed if n == target_name)
target_df = next(df for (n, t, df) in parsed if n == target_name)
target_mapping = MAPPINGS_BY_TYPE.get(target_type, MAPPINGS_BY_TYPE["AUTO"])

override_type = st.selectbox("Override mapping type for target (if detection wrong)", options=["(auto)"] + list(MAPPINGS_BY_TYPE.keys()), index=0)
if override_type != "(auto)":
    target_mapping = MAPPINGS_BY_TYPE[override_type]
    target_type = override_type

baseline_stats = None
baseline_ts = None
if baseline_name != "(none)":
    base_type = next(t for (n, t, df) in parsed if n == baseline_name)
    base_df = next(df for (n, t, df) in parsed if n == baseline_name)
    base_mapping = MAPPINGS_BY_TYPE.get(base_type, MAPPINGS_BY_TYPE["AUTO"])
    base_series, baseline_ts = extract_canonical(base_df, base_mapping)
    baseline_stats = compute_baseline_stats(base_series, THRESHOLDS)

target_series, target_ts = extract_canonical(target_df, target_mapping)

results, overall, reasons, plot_df = compute_results(target_series, THRESHOLDS, baseline_stats)

# Apply timestamp as index for plotting if available
if target_ts is not None and target_ts.notna().sum() > 0:
    plot_df.index = target_ts.reset_index(drop=True)
    plot_df = plot_df.sort_index()

st.subheader(f"Overall Compressor Risk (Target: {target_name} | Mapping: {target_type})")
render_badge(overall, overall)

if reasons:
    st.markdown("**Top contributing reasons:**")
    for r in reasons:
        st.write(f"- {r}")

st.divider()
st.subheader("Parameter Status (Red/Amber/Green)")
cols = st.columns(3)
for i, r in enumerate(results):
    with cols[i % 3]:
        st.markdown(f"### {r.name}")
        render_badge(r.status, r.status)
        st.write(r.value_summary)
        st.caption(r.why)
        if r.extra:
            st.caption(r.extra)
        if r.time_in_red_pct and r.time_in_red_pct > 0:
            st.write(f"Time in RED (approx): {r.time_in_red_pct:.1f}%")

st.divider()
st.subheader("Trends")
if plot_df.empty:
    st.info("No plot-capable signals found with the current mapping.")
else:
    display_map = {
        "discharge_temp_c": "Discharge Temp (°C)",
        "high_pressure_mpa": "High Pressure (MPa)",
        "low_pressure_mpa": "Low Pressure (MPa)",
        "pressure_ratio": "Pressure Ratio (HP/LP)",
        "current_a": "Current (A)",
        "compressor_hz": "Compressor Speed (Hz)",
        "suction_superheat_c": "Suction Superheat (°C)",
    }
    for key in plot_df.columns:
        st.line_chart(plot_df[key].rename(display_map.get(key, key)), height=170)

st.divider()
st.subheader("Export")
summary_df = pd.DataFrame([{
    "parameter": r.name,
    "status": r.status,
    "summary": r.value_summary,
    "why": r.why,
    "time_in_red_pct": r.time_in_red_pct,
} for r in results])

st.download_button(
    "Download analysis summary (CSV)",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="mente_pc_compressor_risk_summary.csv",
    mime="text/csv"
)

st.caption(
    "Disclaimer: This tool provides an operational risk indication based on logged parameters and thresholds. "
    "It is not a guarantee of failure prediction and should be used alongside engineering judgement and site checks."
)
