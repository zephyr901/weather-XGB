"""Input adapters and calendar-aware aggregation. Never modifies source files."""
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import VariableSpec


def read_table(source, preview=False):
    path = Path(source.path)
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    if source.header < 1:
        raise ValueError("Header row numbers start at 1.")
    kwargs = dict(header=source.header - 1, dtype=object, keep_default_na=False)
    if preview:
        kwargs["nrows"] = 100
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="gb18030", **kwargs)
    elif path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        frame = pd.read_excel(path, sheet_name=source.sheet or 0, **kwargs)
    else:
        raise ValueError("Supported formats are CSV, XLSX, XLSM and XLS.")
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns) or any(re.search(r"\.\d+$", c) for c in frame.columns):
        raise ValueError("Duplicate or ambiguous column headers were found. Review the source table.")
    return frame.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)


def fingerprint(source):
    digest = hashlib.sha256()
    with open(source.path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(Path(source.path).resolve()), sha256=digest.hexdigest(), sheet=source.sheet, header=source.header)


def station_ids(frame, source):
    if source.station and source.station in frame:
        def normalize(value):
            if pd.isna(value):
                return None
            if isinstance(value, (int, float)) and float(value).is_integer():
                return str(int(value))
            return str(value).strip()
        result = frame[source.station].map(normalize)
    elif source.single_station.strip():
        result = pd.Series(source.single_station.strip(), index=frame.index)
    else:
        raise ValueError(f"Station ID column {source.station} was not found. For single-station data, specify a station ID override.")
    return result.map(lambda x: str(source.id_map.get(x, x)).strip() if x is not None else None)


def parse_time(values):
    valid = values.dropna()
    if len(valid) and valid.map(lambda x: isinstance(x, (int, float))).all():
        return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit="D", origin="1899-12-30", errors="coerce")
    result = pd.to_datetime(values, format="mixed", errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(result) or getattr(result.dt, "tz", None) is not None:
        raise ValueError("Use consistent station-local times; do not mix time zones.")
    return result


def load_variable(v: VariableSpec, require_target=True):
    f = read_table(v.source)
    f["__station"] = station_ids(f, v.source)
    f["__row"] = np.arange(len(f)) + v.source.header + 1
    if v.source.layout == "wide":
        dates = [c for c in f if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T].*)?", c)]
        if not dates:
            raise ValueError("No date columns were found in the wide table.")
        f = f.melt(id_vars=["__station", "__row"], value_vars=dates, var_name="__date", value_name="__value")
    else:
        if not require_target and v.column not in f:
            f[v.column] = np.nan
        for col in (v.source.time, v.column):
            if col not in f:
                raise ValueError(f"{v.name} is missing column: {col}")
        f = f.assign(__date=f[v.source.time], __value=f[v.column])
    d = pd.DataFrame(dict(station_id=f.__station, date=parse_time(f.__date), y=pd.to_numeric(f.__value, errors="coerce"), source_row=f.__row))
    reason = pd.Series("", index=d.index)
    reason.loc[d.station_id.isna() | d.station_id.eq("")] += "Missing station ID;"
    reason.loc[d.date.isna()] += "Invalid time;"
    invalid = ~np.isfinite(d.y)
    if v.minimum is not None:
        invalid |= d.y < v.minimum
    if v.maximum is not None:
        invalid |= d.y > v.maximum
    reason.loc[invalid] += "Target missing, nonnumeric or out of range;"
    excluded = d[reason != ""].assign(reason=reason[reason != ""])
    valid_keys = d.station_id.notna() & d.date.notna()
    if d[valid_keys].duplicated(["station_id", "date"]).any():
        raise ValueError(f"{v.name}: duplicate station/time records must be resolved explicitly in a copy of the source data.")
    # Retain missing observations on the time grid to prevent shifts over gaps.
    d.loc[invalid, "y"] = np.nan
    d = d[valid_keys].sort_values(["station_id", "date"]).reset_index(drop=True)
    if d.empty or (require_target and not d.y.notna().any()):
        raise ValueError(f"{v.name} has no valid observations.")
    if v.frequency == "h":
        aligned = d.date.eq(d.date.dt.floor("h"))
    elif v.frequency == "D":
        aligned = d.date.eq(d.date.dt.normalize())
    else:
        aligned = d.date.eq(d.date.dt.to_period("M").dt.to_timestamp())
    if not aligned.all():
        raise ValueError(f"{v.name} timestamps do not match declared frequency {v.frequency}. Hourly records must be on the hour, daily records at midnight and monthly records on the first day.")
    return d, excluded


def load_shapes(shapes):
    merged = None
    mapping = {}
    categorical = []
    for spec in shapes:
        f = read_table(spec.source)
        station = station_ids(f, spec.source)
        if station.isna().any() or station.eq("").any() or station.duplicated().any():
            raise ValueError(f"{spec.scale}: morphology station IDs are missing or duplicated.")
        result = pd.DataFrame({"station_id": station})
        names = []
        for col in spec.numeric + spec.categorical:
            if col not in f:
                raise ValueError(f"{spec.scale}: morphology table is missing {col}")
            name = f"m::{spec.scale}::{col}"
            if col in spec.numeric:
                values = pd.to_numeric(f[col], errors="coerce")
                if (f[col].notna() & ~np.isfinite(values)).any():
                    raise ValueError(f"{spec.scale}/{col} contains nonnumeric or infinite values.")
                result[name] = values
            else:
                result[name] = f[col].map(lambda x: str(x) if pd.notna(x) else None)
                categorical.append(name)
            names.append(name)
        mapping[spec.scale] = names
        merged = result if merged is None else merged.merge(result, on="station_id", how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError("The selected morphology scales have no common stations.")
    return merged, mapping, categorical


GRAINS = {"hourly": "h", "daily": "D", "monthly": "MS", "seasonal": "QS", "balanced": "MS", "daynight": "D"}


def period_bounds(t, preset):
    grain = preset.grain
    if grain == "hourly":
        start = t.floor("h")
        return start, start + pd.Timedelta(hours=1), "all"
    if grain == "daily":
        start = t.normalize()
        return start, start + pd.Timedelta(days=1), "all"
    if grain in ("monthly", "balanced"):
        start = t.to_period("M").start_time
        return start, start + pd.offsets.MonthBegin(1), "all"
    if grain == "seasonal":
        start_month = ((t.month - preset.season_start) % 12 // 3 * 3 + preset.season_start - 1) % 12 + 1
        year = t.year - int(start_month > t.month)
        start = pd.Timestamp(year, start_month, 1)
        return start, start + pd.DateOffset(months=3), "all"
    start_day, end_day = preset.day_start, preset.day_end
    hour = t.hour
    is_day = start_day <= hour < end_day if start_day < end_day else hour >= start_day or hour < end_day
    start_hour, end_hour = (start_day, end_day) if is_day else (end_day, start_day)
    start = t.normalize() + pd.Timedelta(hours=start_hour)
    if start > t:
        start -= pd.Timedelta(days=1)
    end = start.normalize() + pd.Timedelta(hours=end_hour)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end, "day" if is_day else "night"


def aggregate(data, variable, preset):
    rank = {"h": 0, "D": 1, "MS": 2}
    target = GRAINS[preset.grain]
    if preset.grain == "daynight" and variable.frequency != "h":
        raise ValueError("Day/night experiments require hourly observations.")
    if target != "QS" and rank[variable.frequency] > rank[target]:
        raise ValueError("Lower-frequency observations cannot be upsampled to a higher frequency.")
    dates = data.date.drop_duplicates()
    bounds = {t: period_bounds(t, preset) for t in dates}
    f = data.copy()
    f["bucket"] = f.date.map(lambda t: bounds[t][0])
    f["end"] = f.date.map(lambda t: bounds[t][1])
    f["segment"] = f.date.map(lambda t: bounds[t][2])
    if f.empty:
        raise ValueError("Aggregation input is empty.")
    grouped = f.groupby(["station_id", "bucket", "segment"], sort=True)
    method = variable.aggregation
    values = grouped.y.sum(min_count=1) if method == "sum" else grouped.y.agg(method)
    result = grouped.agg(valid=("y", "count"), end=("end", "first")).join(values.rename("y")).reset_index().rename(columns={"bucket": "date"})
    if variable.frequency == "MS":
        result["expected"] = (result.end.dt.year - result.date.dt.year) * 12 + result.end.dt.month - result.date.dt.month
    else:
        unit = pd.Timedelta(hours=1) if variable.frequency == "h" else pd.Timedelta(days=1)
        result["expected"] = (result.end - result.date) / unit
    result["coverage"] = result.valid / result.expected
    result["included"] = (result.coverage >= preset.coverage) & result.y.notna()
    audit = result.drop(columns="y").copy()
    result.loc[~result.included, "y"] = np.nan
    if preset.grain == "balanced":
        all_months = pd.date_range(result.date.min(), result.date.max(), freq="MS")
        counts = result[result.included].groupby("station_id").date.nunique()
        eligible = counts[counts == len(all_months)].index
        audit["station_included"] = audit.station_id.isin(eligible)
        result = result[result.station_id.isin(eligible)].groupby("station_id", as_index=False).agg(y=("y", "mean"), date=("date", "min"), end=("end", "max"))
        result["segment"] = "all"
    return result[["station_id", "date", "end", "segment", "y"]], audit


def calendar_features(data):
    d = data.copy()
    hour = d.date.dt.hour
    day = d.date.dt.dayofyear
    month = d.date.dt.month
    for name, values, cycle in (("hour", hour, 24), ("year", day, 365.25), ("month", month, 12)):
        d[f"t::{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        d[f"t::{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    return d


def forecast_features(data, grain, lags=None):
    frequency = GRAINS[grain]
    frames = []
    for station, group in data.groupby("station_id"):
        g = group.set_index("date").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq=frequency)
        g = g.reindex(grid).rename_axis("date").reset_index()
        g["station_id"] = station
        frames.append(g)
    d = pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"])
    candidates = {"hourly": [1, 24, 168], "daily": [1, 7, 14], "monthly": [1, 3, 12]}[grain]
    # Only use period counts, never target values, to choose the historical windows.
    if lags is None:
        length = d.groupby("station_id").size().max()
        lags = [lag for lag in candidates if length >= lag + 12]
        if 1 not in lags:
            raise ValueError("One-step forecasting requires at least 13 time periods and sufficient data for inner and outer time validation.")
    for lag in lags:
        d[f"h::lag_{lag}"] = d.groupby("station_id").y.shift(lag)
    window = max(2, max(lags))
    d["h::rolling_mean"] = d.groupby("station_id").y.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    d["h::rolling_std"] = d.groupby("station_id").y.transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
    if "background" in d:
        d["background"] = d.groupby("station_id").background.shift(1)
    return calendar_features(d), lags
