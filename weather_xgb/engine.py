"""Independent preparation, nested validation, native model export and inference."""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import threading
import uuid

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from .config import ModelManifest, dump, from_dict, validate
from .data import (GRAINS, aggregate, calendar_features, fingerprint, forecast_features,
                   load_shapes, load_variable, station_ids, read_table)


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def scores(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    sst = np.square(y - y.mean()).sum()
    return {"r2": float(1 - np.square(y - pred).sum() / sst) if len(y) > 1 and sst > 0 else None,
            "rmse": float(np.sqrt(np.square(y - pred).mean())), "mae": float(np.abs(y - pred).mean())}


def safe_name(value):
    return re.sub(r"[^\w.-]+", "_", str(value))[:90]


@dataclass
class Prepared:
    data: pd.DataFrame
    variable_index: int
    preset_index: int
    scale: str
    group: str
    features: list
    categorical: list
    lags: list
    coverage: pd.DataFrame
    excluded: pd.DataFrame


def splits(data, config, task, inner=False):
    if task == "association":
        n = config.inner_folds if inner else config.outer_folds
        if data.station_id.nunique() < n:
            raise ValueError(f"Only {data.station_id.nunique()} valid stations are available for {n} {'inner' if inner else 'outer'} folds.")
        return list(GroupKFold(n).split(data, groups=data.station_id))
    n = config.time_inner if inner else config.time_outer
    times = np.sort(data.date.unique())
    if len(times) < n + 2:
        raise ValueError(f"Only {len(times)} valid time points are available for {n} {'inner' if inner else 'outer'} time splits.")
    result = []
    for a, b in TimeSeriesSplit(n_splits=n).split(times):
        train = np.flatnonzero(data.date.isin(times[a]))
        test = np.flatnonzero(data.date.isin(times[b]))
        if min(len(train), len(test)) < 2:
            raise ValueError("Each time validation split requires at least two valid samples.")
        result.append((train, test))
    return result


def check_splits(data, config, task):
    outer = splits(data, config, task)
    for a, _ in outer:
        splits(data.iloc[a], config, task, inner=True)
    return outer


def prepare_project(config, cancel=None):
    validate(config)
    cancel = cancel or threading.Event()
    shapes, mapping, categorical = load_shapes(config.shapes)
    shape_union = set()
    for spec in config.shapes:
        shape_union.update(station_ids(read_table(spec.source), spec.source))
    reports, jobs, sources, matching = [], [], [], []
    for spec in config.shapes:
        sources.append(fingerprint(spec.source))
    for vi, variable in enumerate(config.variables):
        if cancel.is_set():
            raise InterruptedError("Cancelled.")
        raw, excluded = load_variable(variable)
        sources.append(fingerprint(variable.source))
        for sid in sorted(set(raw.station_id) | set(shapes.station_id)):
            matching.append(dict(target=variable.name, station_id=sid, weather=sid in set(raw.station_id), morphology=sid in set(shapes.station_id)))
        unmatched = raw[~raw.station_id.isin(shapes.station_id)].assign(reason="Not matched across all selected morphology scales")
        excluded = pd.concat([excluded, unmatched], ignore_index=True)
        for pi, preset in enumerate(config.presets):
            report = dict(target=variable.name, preset=preset.name, task=preset.task, status="ready", rows=0, stations=0, models=0, fits=0, reason="")
            try:
                ag, coverage = aggregate(raw[raw.station_id.isin(shapes.station_id)], variable, preset)
                if preset.features == "background":
                    refs = next((v for v in config.backgrounds if v.name == variable.name), None)
                    if refs is not None:
                        if refs.unit != variable.unit or refs.aggregation != variable.aggregation:
                            raise ValueError("Background target units and aggregation must match the study target.")
                        reference, _ = load_variable(refs)
                        sources.append(fingerprint(refs.source))
                        if set(reference.station_id) & shape_union:
                            raise ValueError("The independent background file contains study stations. Remove overlapping station IDs.")
                    else:
                        ids = set(preset.reference_ids)
                        if not ids:
                            raise ValueError("Specify independent reference station IDs or import a background file for this variable.")
                        if ids & shape_union:
                            raise ValueError("Background station IDs must not include study stations from any morphology table.")
                        reference = raw[raw.station_id.isin(ids)]
                        refs = variable
                    if reference.empty:
                        raise ValueError("No independent background stations were matched.")
                    bg, _ = aggregate(reference, refs, preset)
                    bg = bg.dropna(subset=["y"]).groupby(["date", "segment"]).y.agg(["mean", "count"]).reset_index()
                    bg.loc[bg["count"] < preset.reference_min, "mean"] = np.nan
                    ag = ag.merge(bg.rename(columns={"mean": "background", "count": "reference_count"}), on=["date", "segment"], how="left", validate="many_to_one")
                lags = []
                if preset.task == "forecast":
                    ag, lags = forecast_features(ag, preset.grain)
                    ag = ag.dropna(subset=["y", "h::lag_1"])
                else:
                    ag = calendar_features(ag).dropna(subset=["y"])
                if preset.features == "background":
                    ag = ag.dropna(subset=["background"])
                ag = ag.merge(shapes, on="station_id", validate="many_to_one")
                if ag.empty:
                    raise ValueError("No valid samples remain after coverage, missing-value or background filtering.")
                if preset.task == "association" and ag.station_id.nunique() < 2:
                    raise ValueError("A single station has no between-station morphology variation. Select one-step forecasting.")
                segments = list(ag.segment.dropna().unique()) if preset.grain == "daynight" else ["all"]
                pending = []
                for segment in segments:
                    part = ag[ag.segment == segment].copy() if preset.grain == "daynight" else ag.copy()
                    part = part.sort_values(["station_id", "date"]).reset_index(drop=True)
                    outer = check_splits(part, config, preset.task)
                    for scale, cols in mapping.items():
                        for group, selected in config.groups.items():
                            def selected_metric(col):
                                original = col.split("::")[-1]
                                normalized = re.sub(r"[\s_]", "", original).upper()
                                return not selected or original in selected or any(normalized in (str(x).upper(), f"{scale}M{str(x).upper()}", f"{str(x).upper()}{scale}M") for x in selected)
                            morph = [col for col in cols if selected_metric(col)]
                            if not morph:
                                raise ValueError(f"{scale}/{group} has no usable features.")
                            features = morph.copy()
                            if preset.features == "calendar" or preset.task == "forecast":
                                features += [c for c in part if c.startswith("t::")]
                            if preset.task == "forecast":
                                features += [c for c in part if c.startswith("h::")]
                            if preset.features == "background":
                                features += ["background"]
                            label = group + (" / " + segment if preset.grain == "daynight" else "")
                            pending.append(Prepared(part, vi, pi, scale, label, features, [c for c in categorical if c in features], lags, coverage, excluded))
                            folds_inner = config.inner_folds if preset.task == "association" else config.time_inner
                            baseline_fits = len(outer) if preset.features == "background" else 0
                            report["fits"] += (len(outer) + 1) * (folds_inner * len(config.grid) + 1) + baseline_fits
                jobs.extend(pending)
                report.update(rows=len(ag), stations=int(ag.station_id.nunique()), models=len(pending))
                constants = [col for columns in mapping.values() for col in columns if ag[col].nunique(dropna=True) <= 1]
                if constants:
                    report["reason"] = "Constant or entirely missing features: " + ", ".join(constants)
            except ValueError as exc:
                report.update(status="unavailable", reason=str(exc), fits=0)
            reports.append(report)
    return jobs, dict(reports=reports, sources=sources, matching=matching)


class Encoder:
    def __init__(self, features, categorical, categories=None):
        self.features = list(features)
        self.categorical = list(categorical)
        self.categories = categories

    def fit(self, frame):
        self.categories = {c: sorted(frame[c].dropna().astype(str).unique().tolist()) for c in self.categorical}
        return self

    def transform(self, frame):
        out = pd.DataFrame(index=frame.index)
        for i, col in enumerate(self.features):
            if col not in frame:
                raise ValueError(f"Prediction data is missing model column: {col}")
            if col in self.categorical:
                mapping = {v: k for k, v in enumerate(self.categories[col])}
                out[f"f{i}"] = frame[col].map(lambda v: mapping.get(str(v), np.nan) if pd.notna(v) else np.nan)
            else:
                out[f"f{i}"] = pd.to_numeric(frame[col], errors="coerce")
        return out.astype(float)


def estimator(params, config):
    return xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=config.n_jobs, random_state=config.seed, subsample=.85, reg_alpha=.1, **params)


def fit(train, features, categorical, params, config):
    encoder = Encoder(features, categorical).fit(train)
    model = estimator(params, config)
    model.fit(encoder.transform(train), train.y)
    return model, encoder


def checkpoint(cancel):
    if cancel.is_set():
        raise InterruptedError("Cancelled. Completed folds and models are retained.")


def tune(data, job, config, cancel):
    trials = []
    task = config.presets[job.preset_index].task
    inner = splits(data, config, task, inner=True)
    for params in config.grid:
        total, n = 0., 0
        for a, b in inner:
            checkpoint(cancel)
            model, enc = fit(data.iloc[a], job.features, job.categorical, params, config)
            pred = model.predict(enc.transform(data.iloc[b]))
            total += np.square(data.iloc[b].y.to_numpy() - pred).sum()
            n += len(b)
        trials.append(dict(params=params, mse=float(total / n)))
    return min(trials, key=lambda t: t["mse"])["params"], trials


def save_model(model, enc, folder, job, config, params):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(folder / "model.ubj")
    manifest = ModelManifest(asdict(config.variables[job.variable_index]), asdict(config.presets[job.preset_index]), job.features,
                             [c for c in job.features if c not in job.categorical], enc.categories,
                             job.lags, job.scale, asdict(config), params,
                             segment=str(job.data.segment.iloc[0]) if config.presets[job.preset_index].grain == "daynight" else "all")
    dump(manifest, folder / "manifest.json")


def run_project(config, emit=lambda *args: None, cancel=None):
    from .plots import export_plots
    cancel = cancel or threading.Event()
    validate(config)
    out = Path(config.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    dump(config, out / "project.json")
    dump({"status": "running"}, out / "status.json")
    emit("output", str(out))
    summaries = []
    try:
        checkpoint(cancel)
        jobs, audit = prepare_project(config, cancel)
        dump(audit, out / "audit.json")
        csv(pd.DataFrame(audit["matching"]), out / "station_matching.csv")
        if not jobs:
            raise ValueError("No experiments are ready to run: " + "；".join(r["reason"] for r in audit["reports"]))
        total_steps = sum((config.outer_folds if config.presets[j.preset_index].task == "association" else config.time_outer) + 1 for j in jobs)
        step = 0
        for index, job in enumerate(jobs):
            checkpoint(cancel)
            variable, preset = config.variables[job.variable_index], config.presets[job.preset_index]
            name = f"{index+1:03}_{safe_name(variable.name)}_{safe_name(preset.name)}_{safe_name(job.scale)}_{safe_name(job.group)}"
            folder = out / name
            folder.mkdir()
            csv(job.coverage, folder / "coverage.csv")
            csv(job.excluded, folder / "excluded.csv")
            csv(job.data, folder / "prepared.csv")
            preds, metrics, selections = [], [], []
            shap_rows, importance = [], []
            for fold, (a, b) in enumerate(splits(job.data, config, preset.task), 1):
                checkpoint(cancel)
                emit("log", f"{name} · Outer fold {fold}")
                train, test = job.data.iloc[a], job.data.iloc[b]
                params, trials = tune(train, job, config, cancel)
                model, enc = fit(train, job.features, job.categorical, params, config)
                pred = model.predict(enc.transform(test))
                record = test[["station_id", "date", "y"]].copy()
                record["prediction"] = pred
                record["baseline"] = float(train.y.mean())
                record["fold"] = fold
                if preset.task == "forecast":
                    record["persistence"] = test["h::lag_1"].to_numpy()
                if preset.features == "background":
                    bgmodel, bgenc = fit(train, ["background"], [], params, config)
                    record["background_only"] = bgmodel.predict(bgenc.transform(test))
                preds.append(record)
                row = dict(fold=fold, train_rows=len(train), test_rows=len(test), **scores(test.y, pred))
                row.update({"train_" + k: v for k, v in scores(train.y, model.predict(enc.transform(train))).items()})
                for baseline in ("baseline", "persistence", "background_only"):
                    if baseline in record:
                        row.update({baseline + "_" + k: v for k, v in scores(record.y, record[baseline]).items()})
                metrics.append(row)
                selections.append(dict(fold=fold, selected=params, trials=trials))
                save_model(model, enc, folder / f"fold_{fold}", job, config, params)
                # Native exact TreeSHAP uses held-out samples and the fitted fold encoder.
                sample = test.sample(min(300, len(test)), random_state=config.seed)
                matrix = enc.transform(sample)
                contrib = model.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)
                shap_frame = pd.DataFrame(contrib[:, :-1], columns=job.features)
                shap_frame.insert(0, "fold", fold)
                shap_rows.append(shap_frame)
                importance.extend(dict(fold=fold, feature=f, gain=float(g)) for f, g in zip(job.features, model.feature_importances_))
                csv(pd.concat(preds, ignore_index=True), folder / "predictions.csv")
                csv(pd.DataFrame(metrics), folder / "metrics.csv")
                dump(selections, folder / "selection.json")
                step += 1
                emit("progress", [step, total_steps])
            checkpoint(cancel)
            params, trials = tune(job.data, job, config, cancel)
            final_model, final_enc = fit(job.data, job.features, job.categorical, params, config)
            save_model(final_model, final_enc, folder / "final", job, config, params)
            dump(dict(selected=params, trials=trials), folder / "final_selection.json")
            pm = pd.concat(preds, ignore_index=True)
            sm = pd.concat(shap_rows, ignore_index=True)
            im = pd.DataFrame(importance)
            csv(sm, folder / "treeshap.csv")
            csv(im, folder / "importance.csv")
            export_plots(pm, im, sm, folder, variable.unit)
            row = dict(experiment=name, target=variable.name, unit=variable.unit, preset=preset.name, grain=preset.grain, task=preset.task,
                       scale=job.scale, group=job.group, rows=len(pm), stations=int(pm.station_id.nunique()), **scores(pm.y, pm.prediction))
            for baseline in ("baseline", "persistence", "background_only"):
                if baseline in pm:
                    row.update({baseline + "_" + k: v for k, v in scores(pm.y, pm[baseline]).items()})
            row["fold_rmse_sd"] = float(pd.DataFrame(metrics).rmse.std())
            summaries.append(row)
            csv(pd.DataFrame(summaries), out / "summary.csv")
            step += 1
            emit("progress", [step, total_steps])
        skipped = [r for r in audit["reports"] if r["status"] != "ready"]
        notes = ["# Weather XGBoost experiment report", "", "Evaluation uses outer out-of-fold predictions. The final model is tuned and fitted again on all valid training samples; its training score does not measure generalization.",
                 "Station validation evaluates transfer to held-out stations. Time validation evaluates rolling one-step forecasts using actual historical observations available at each prediction time.",
                 "Reference data comes only from the specified independent stations. Morphology associations and TreeSHAP do not establish causality.",
                 "Interpret each target, unit and time grain separately. Different tasks are not ranked together. Negative R-squared values are retained; R-squared is undefined for constant targets.",
                 "TreeSHAP uses at most 300 observations per outer validation fold. Gain importance measures tree split importance.",
                 "", "## Experiment results", *[f"- {r['target']} / {r['preset']} / {r['scale']} / {r['group']}: R²={r['r2']}, RMSE={r['rmse']:.5g} {r['unit']}, MAE={r['mae']:.5g}" for r in summaries],
                 "", "## Skipped presets", *[f"- {r['target']} / {r['preset']}: {r['reason']}" for r in skipped]]
        (out / "review.md").write_text("\n\n".join(notes), encoding="utf-8")
        dump(dict(status="completed_with_skips" if skipped else "completed", experiments=len(summaries), skipped=len(skipped)), out / "status.json")
        emit("result", str(out))
        return out
    except Exception as exc:
        state = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        dump(dict(status=state, error=str(exc), completed_experiments=len(summaries)), out / "status.json")
        (out / "review.md").write_text(f"# Experiment incomplete\n\nStatus: {state}\n\n{exc}\n\nCompleted folds are saved and must not be compared as a complete experiment.", encoding="utf-8")
        raise


def load_bundle(folder):
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if m.get("version") != 1:
        raise ValueError("Unsupported model version.")
    model = xgb.XGBRegressor()
    model.load_model(folder / "model.ubj")
    return model, Encoder(m["features"], list(m["categories"]), m["categories"]), m


def predict_frame(folder, frame):
    model, enc, _ = load_bundle(folder)
    return model.predict(enc.transform(frame))


def predict_project(folder, incoming):
    """Recreate training transforms using new mapped data; forecast one next period per station."""
    model, enc, manifest = load_bundle(folder)
    trained = from_dict(manifest["project"])
    vi = next(i for i, v in enumerate(trained.variables) if v.name == manifest["variable"]["name"])
    variable = next((v for v in incoming.variables if v.name == manifest["variable"]["name"]), None)
    if variable is None:
        raise ValueError("The prediction project does not contain the target name required by the model.")
    original = trained.variables[vi]
    if (variable.unit, variable.frequency, variable.aggregation) != (original.unit, original.frequency, original.aggregation):
        raise ValueError("Prediction data units, observation frequency and aggregation must match the model.")
    from .config import ExperimentPreset
    preset = ExperimentPreset(**manifest["preset"])
    raw, _ = load_variable(variable, require_target=preset.task == "forecast")
    ag, _ = aggregate(raw, variable, preset)
    shapes, _, _ = load_shapes(incoming.shapes)
    if preset.features == "background":
        bgv = next((v for v in incoming.backgrounds if v.name == variable.name), None)
        reference = None
        if bgv:
            if (bgv.unit, bgv.aggregation) != (variable.unit, variable.aggregation):
                raise ValueError("Background units or aggregation do not match.")
            reference, _ = load_variable(bgv)
        else:
            reference = raw[raw.station_id.isin(preset.reference_ids)]
            bgv = variable
        if reference.empty or set(reference.station_id) & set(shapes.station_id):
            raise ValueError("Required independent background stations are missing or overlap the study stations.")
        bg, _ = aggregate(reference, bgv, preset)
        bg = bg.dropna(subset=["y"]).groupby(["date", "segment"]).y.agg(["mean", "count"]).reset_index()
        bg.loc[bg["count"] < preset.reference_min, "mean"] = np.nan
        ag = ag.merge(bg.rename(columns={"mean": "background"}), on=["date", "segment"], how="left", validate="many_to_one")
    ag = ag[ag.station_id.isin(shapes.station_id)]
    if preset.task == "forecast":
        future = []
        offset = pd.tseries.frequencies.to_offset(GRAINS[preset.grain])
        for sid, group in ag.groupby("station_id"):
            future.append(dict(station_id=sid, date=group.date.max() + offset, y=np.nan, segment="all"))
        wanted = pd.DataFrame(future)
        if wanted.empty:
            raise ValueError("No stations were matched.")
        ag = pd.concat([ag, wanted], ignore_index=True)
        ag, _ = forecast_features(ag, preset.grain, manifest["lags"])
        ag = ag.merge(wanted[["station_id", "date"]], on=["station_id", "date"], validate="one_to_one")
        needed = [f"h::lag_{lag}" for lag in manifest["lags"]]
        if ag[needed].isna().any().any():
            raise ValueError("Insufficient history or missing values at required lag times: " + ", ".join(needed))
    else:
        ag = calendar_features(ag)
        if preset.grain == "daynight":
            ag = ag[ag.segment == manifest.get("segment", "all")]
    ag = ag.merge(shapes, on="station_id", validate="many_to_one")
    if ag.empty or (preset.features == "background" and ag.background.isna().any()):
        raise ValueError("No prediction samples are available or background coverage is insufficient.")
    result = ag[["station_id", "date"]].copy()
    result["prediction"] = model.predict(enc.transform(ag))
    result["target"] = original.name
    result["unit"] = original.unit
    return result


def worker(config_dict, events, cancel, check_only=False):
    try:
        config = from_dict(config_dict)
        if isinstance(check_only, str):
            result = predict_project(check_only, config)
            events.put(("predicted", result))
        elif check_only:
            _, audit = prepare_project(config, cancel)
            events.put(("checked", audit))
        else:
            run_project(config, lambda kind, value: events.put((kind, value)), cancel)
    except Exception as exc:
        events.put(("error", str(exc)))
    finally:
        events.put(("done", None))
