"""Versioned, JSON-serializable project and model contracts."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

VERSION = 1


@dataclass
class TableSource:
    path: str = ""
    sheet: str = ""
    header: int = 1
    station: str = "station_id"
    time: str = "datetime"
    single_station: str = ""
    layout: str = "long"
    id_map: dict = field(default_factory=dict)


@dataclass
class VariableSpec:
    name: str = "Air temperature"
    unit: str = "°C"
    column: str = "temperature"
    frequency: str = "h"
    aggregation: str = "mean"
    source: TableSource = field(default_factory=TableSource)
    minimum: float | None = None
    maximum: float | None = None
    version: int = VERSION


@dataclass
class ShapeSource:
    scale: str = "100"
    source: TableSource = field(default_factory=TableSource)
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)


@dataclass
class ExperimentPreset:
    name: str = "Hourly"
    grain: str = "hourly"
    task: str = "association"
    features: str = "calendar"
    coverage: float = .8
    day_start: int = 6
    day_end: int = 18
    season_start: int = 12
    reference_ids: list = field(default_factory=list)
    reference_min: int = 1
    version: int = VERSION


@dataclass
class ProjectConfig:
    name: str = "Weather experiment"
    variables: list = field(default_factory=lambda: [VariableSpec(), VariableSpec(name="Wind speed", unit="m/s", column="wind", minimum=0)])
    shapes: list = field(default_factory=list)
    presets: list = field(default_factory=lambda: [ExperimentPreset()])
    backgrounds: list = field(default_factory=list)
    groups: dict = field(default_factory=lambda: {"All features": []})
    output_dir: str = field(default_factory=lambda: str(Path.home() / "Documents" / "WeatherXGB"))
    outer_folds: int = 5
    inner_folds: int = 3
    time_outer: int = 3
    time_inner: int = 2
    seed: int = 42
    n_jobs: int = 2
    grid: list = field(default_factory=lambda: [
        dict(n_estimators=120, max_depth=2, learning_rate=.04, min_child_weight=5, reg_lambda=10),
        dict(n_estimators=250, max_depth=3, learning_rate=.03, min_child_weight=10, reg_lambda=20)])
    version: int = VERSION


@dataclass
class ModelManifest:
    variable: dict
    preset: dict
    features: list
    numeric: list
    categories: dict
    lags: list
    scale: str
    project: dict
    parameters: dict
    segment: str = "all"
    version: int = VERSION


def dump(value, path):
    value = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def variable_from_dict(d):
    d = dict(d)
    d["source"] = TableSource(**d.get("source", {}))
    return VariableSpec(**d)


def from_dict(d):
    d = dict(d)
    if d.get("version", 1) != VERSION:
        raise ValueError("Unsupported configuration version.")
    d["variables"] = [variable_from_dict(v) for v in d.get("variables", [])]
    d["backgrounds"] = [variable_from_dict(v) for v in d.get("backgrounds", [])]
    d["shapes"] = [ShapeSource(scale=s["scale"], source=TableSource(**s["source"]), numeric=s.get("numeric", []), categorical=s.get("categorical", [])) for s in d.get("shapes", [])]
    d["presets"] = [ExperimentPreset(**p) for p in d.get("presets", [])]
    return ProjectConfig(**d)


def validate(c):
    if not isinstance(c.groups, dict) or any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in c.groups.values()):
        raise ValueError("Feature groups must map names to lists of column names.")
    if not c.output_dir.strip():
        raise ValueError("Select an output directory.")
    if len(c.variables) != 2 or not c.shapes or not c.presets:
        raise ValueError("Configure two weather variables, at least one morphology table and one experiment preset.")
    if len({v.name for v in c.variables}) != 2:
        raise ValueError("The two target names must be different.")
    if len({s.scale for s in c.shapes}) != len(c.shapes):
        raise ValueError("Morphology scale names must be unique.")
    for v in c.variables + c.backgrounds:
        if any(token in (v.name + " " + v.column).casefold() for token in ("风向", "wind_direction", "wind direction")):
            raise ValueError("Wind direction is a circular variable and is not supported as an ordinary regression target.")
        if not v.name.strip() or v.frequency not in ("h", "D", "MS") or v.aggregation not in ("mean", "sum", "min", "max"):
            raise ValueError("Invalid variable name, observation frequency or aggregation.")
        if v.minimum is not None and v.maximum is not None and v.minimum > v.maximum:
            raise ValueError("The valid minimum cannot exceed the valid maximum.")
    for s in c.shapes:
        if not s.numeric and not s.categorical:
            raise ValueError(f"{s.scale}: select at least one morphology feature.")
        if set(s.numeric) & set(s.categorical):
            raise ValueError("A feature cannot be both numeric and categorical.")
    for p in c.presets:
        if p.version != VERSION or not 0 < p.coverage <= 1 or p.reference_min < 1:
            raise ValueError("Invalid preset version, coverage or reference station threshold.")
        if p.grain not in ("hourly", "daily", "monthly", "seasonal", "daynight", "balanced") or p.task not in ("association", "forecast") or p.features not in ("morphology", "calendar", "background"):
            raise ValueError("Unknown experiment preset.")
        if p.task == "forecast" and p.grain not in ("hourly", "daily", "monthly"):
            raise ValueError("One-step forecasting supports hourly, daily and monthly grains only.")
        if not 0 <= p.day_start < 24 or not 0 <= p.day_end < 24 or p.day_start == p.day_end or not 1 <= p.season_start <= 12:
            raise ValueError("Invalid day/night boundaries or first season start month.")
    if any(type(getattr(c, k)) is not int or getattr(c, k) < 2 for k in ("outer_folds", "inner_folds", "time_outer", "time_inner")) or type(c.n_jobs) is not int or c.n_jobs < 1:
        raise ValueError("Validation folds must be at least 2; threads must be at least 1.")
    if type(c.seed) is not int or c.seed < 0 or not c.grid or not c.groups:
        raise ValueError("Invalid random seed, parameter candidates or feature groups.")
    required = {"n_estimators", "max_depth", "learning_rate", "min_child_weight", "reg_lambda"}
    for row in c.grid:
        if set(row) != required or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row.values()):
            raise ValueError("Parameter candidates have missing fields or invalid numbers.")
        if any(type(row[k]) is not int or row[k] < 1 for k in ("n_estimators", "max_depth")) or not 0 < row["learning_rate"] <= 1 or min(row["min_child_weight"], row["reg_lambda"]) < 0:
            raise ValueError("XGBoost parameters are outside their valid ranges.")


def library():
    """Built-in presets are data definitions, not executable plugins."""
    return [ExperimentPreset(name=n, grain=g) for n, g in [("Hourly", "hourly"), ("Daily", "daily"), ("Monthly", "monthly"), ("Seasonal", "seasonal"), ("Day/night groups", "daynight"), ("Month-balanced station mean", "balanced")]] + [ExperimentPreset(name="Independent background stations", features="background")]


def migrate_legacy(old):
    """Return a reviewable conversion without silently changing target semantics."""
    c = ProjectConfig(name="Imported legacy project", output_dir=old.get("output_dir", ProjectConfig().output_dir))
    for v in c.variables:
        v.source.path = old.get("weather_file", "")
    c.shapes = [ShapeSource(scale=str(r), source=TableSource(path=f)) for r, f in old.get("shape_files", {}).items() if f]
    c.presets = [ExperimentPreset(name=p.name, grain=p.grain, features=p.features, coverage=old.get("coverage", .8)) for p in library() if p.grain in old.get("schemes", []) or (p.features == "background" and "background" in old.get("schemes", []))]
    c.grid = old.get("xgb_grid", c.grid)
    for k in ("outer_folds", "inner_folds", "n_jobs", "seed"):
        setattr(c, k, old.get(k, getattr(c, k)))
    return c, "Review the weather columns, frequency, units and morphology features. The midpoint of legacy maximum/minimum temperatures is not assumed to be an observed daily mean. Select independent reference stations explicitly. Each target is modelled separately."


def load_project_file(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if "weather_file" in raw:
        config, message = migrate_legacy(raw)
    else:
        config, message = from_dict(raw), ""
    for spec in config.variables + config.backgrounds + config.shapes:
        if spec.source.path and not Path(spec.source.path).is_absolute():
            spec.source.path = str(path.parent / spec.source.path)
    if not Path(config.output_dir).is_absolute():
        config.output_dir = str(path.parent / config.output_dir)
    return config, message
