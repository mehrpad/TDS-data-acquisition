import re
from pathlib import Path

import numpy as np
import pandas as pd


_TEMPERATURE_NAMES = {
    "t",
    "tc",
    "tdegc",
    "temp",
    "tempc",
    "tempdegc",
    "temperature",
    "temperaturec",
    "temperaturedegc",
    "temperaturecelsius",
}
_RESISTANCE_NAMES = {
    "r",
    "rho",
    "rohm",
    "resistance",
    "resistanceohm",
    "resistivity",
    "resistivityohm",
    "resistivityohmm",
    "calculatedresistance",
    "calculatedresistivity",
    "correctedresistance",
    "correctedresistanceohm",
}


def _normalize_curve_header(value):
    text = str(value).strip().lower()
    text = (
        text.replace("ρ", "rho")
        .replace("Ω", "ohm")
        .replace("ω", "ohm")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("°", "deg")
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def _curve_column_role(value):
    normalized = _normalize_curve_header(value)
    if normalized in _TEMPERATURE_NAMES:
        return "temperature"
    if normalized in _RESISTANCE_NAMES:
        return "resistance"
    if normalized.startswith("electricalresistivity"):
        return "resistance"
    return None


def _extract_curve_from_frames(frames, max_header_rows=12):
    """Find a labeled R/T table in one or more headerless DataFrames."""
    for sheet_name, raw_frame in frames.items():
        if raw_frame is None or raw_frame.empty:
            continue
        rows_to_scan = min(max(int(max_header_rows), 1), len(raw_frame.index))
        for header_row in range(rows_to_scan):
            resistance_column = None
            temperature_column = None
            for column_index, header_value in enumerate(raw_frame.iloc[header_row].tolist()):
                role = _curve_column_role(header_value)
                if role == "resistance" and resistance_column is None:
                    resistance_column = column_index
                elif role == "temperature" and temperature_column is None:
                    temperature_column = column_index

            if resistance_column is None or temperature_column is None:
                continue

            resistance_label = str(raw_frame.iat[header_row, resistance_column]).strip()
            temperature_label = str(raw_frame.iat[header_row, temperature_column]).strip()
            curve_frame = pd.DataFrame(
                {
                    "resistance": pd.to_numeric(
                        raw_frame.iloc[header_row + 1 :, resistance_column], errors="coerce"
                    ),
                    "temperature": pd.to_numeric(
                        raw_frame.iloc[header_row + 1 :, temperature_column], errors="coerce"
                    ),
                }
            ).dropna()
            curve_frame = (
                curve_frame.drop_duplicates(subset=["temperature"], keep="last")
                .sort_values("temperature")
                .reset_index(drop=True)
            )
            if len(curve_frame) < 2:
                continue

            curve = np.vstack(
                (
                    curve_frame["resistance"].to_numpy(dtype=float),
                    curve_frame["temperature"].to_numpy(dtype=float),
                )
            )
            source = {
                "sheet": str(sheet_name),
                "header_row": int(header_row + 1),
                "resistance_column": resistance_label,
                "temperature_column": temperature_label,
                "points": int(curve.shape[1]),
            }
            return curve, source

    raise ValueError(
        "No usable resistance/resistivity-versus-temperature table was found. "
        "Recognized temperature headers include T, Temperature, and Temperature [C]; "
        "recognized resistance headers include R, R_ohm, Resistance, Resistivity, and "
        "Corrected Resistance (Ohm). Headers may be within the first 12 rows."
    )


def load_resistance_temperature_file(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frames = {"CSV": pd.read_csv(path, header=None)}
    elif suffix == ".xlsx":
        frames = pd.read_excel(path, sheet_name=None, header=None)
    else:
        raise ValueError("R vs. T file must be a .csv or .xlsx file.")
    return _extract_curve_from_frames(frames)
