"""Generate clearly labelled synthetic examples, never scientific observations."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ProjectConfig, VariableSpec, TableSource, ShapeSource, ExperimentPreset, dump


def generate(folder):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    times = pd.date_range("2025-01-01", periods=24 * 21, freq="h")
    rows = []
    for i in range(10):
        for t in times:
            weather = np.sin(t.dayofyear / 5) * 2
            rows.append(dict(station_id=f"S{i:02}", datetime=str(t), temperature=18 + i * .4 + 3 * np.sin(t.hour / 24 * 2 * np.pi) + weather + rng.normal(0,.3), wind=2 + i * .1 + .5 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0,.1)))
    pd.DataFrame(rows).to_csv(folder / "synthetic_weather.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dict(station_id=[f"S{i:02}" for i in range(8)], GCR=np.linspace(.1,.8,8), BCR=np.linspace(.8,.1,8), LCZ=["2", "5", "6", "9"] * 2)).to_csv(folder / "synthetic_shapes.csv", index=False, encoding="utf-8-sig")
    source = TableSource(path=str(folder / "synthetic_weather.csv"), station="station_id", time="datetime")
    c = ProjectConfig(name="Synthetic example · Not observed data", variables=[VariableSpec(name="Air temperature", unit="°C", column="temperature", source=source), VariableSpec(name="Wind speed", unit="m/s", column="wind", minimum=0, source=source)],
                      shapes=[ShapeSource(scale="250", source=TableSource(path=str(folder / "synthetic_shapes.csv"), station="station_id"), numeric=["GCR", "BCR"], categorical=["LCZ"])],
                      presets=[ExperimentPreset(name="Daily morphology", grain="daily"), ExperimentPreset(name="Daily one-step forecast", grain="daily", task="forecast"), ExperimentPreset(name="Independent background", grain="daily", features="background", reference_ids=["S08","S09"])],
                      outer_folds=3, inner_folds=2, grid=[dict(n_estimators=20,max_depth=2,learning_rate=.08,min_child_weight=2,reg_lambda=5)], output_dir=str(folder.parent / "runs"))
    dump(c, folder / "example_project.json")
    return c


if __name__ == "__main__":
    generate(Path(__file__).parent / "examples")
