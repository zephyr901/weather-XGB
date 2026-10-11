"""Headless publication exports, also viewable in the desktop results page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c5cbd0", "text.color": "#28343c", "axes.labelcolor": "#28343c"})


def export_plots(pred, importance, shap, folder, unit):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / (name + ".png"), dpi=180, facecolor="white")
        fig.savefig(folder / (name + ".svg"), facecolor="white")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(pred.y, pred.prediction, s=13, alpha=.55, c="#258b99", edgecolors="none")
    lo, hi = min(pred.y.min(), pred.prediction.min()), max(pred.y.max(), pred.prediction.max())
    ax.plot([lo, hi], [lo, hi], color="#b29a66", lw=1)
    ax.set(xlabel=f"Observed ({unit})", ylabel=f"Out-of-fold prediction ({unit})", title="Observed / Predicted")
    save(fig, "observed_predicted")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sid = pred.station_id.iloc[0]
    sample = pred[pred.station_id == sid].sort_values("date")
    ax.plot(sample.date, sample.y, color="#45545f", label="Observed", lw=1.3)
    ax.plot(sample.date, sample.prediction, color="#258b99", label="Prediction", lw=1.3)
    ax.legend(frameon=False)
    ax.set(title=f"Station {sid}", ylabel=unit)
    fig.autofmt_xdate()
    save(fig, "time_series")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(pred.prediction, pred.y - pred.prediction, c="#258b99", s=13, alpha=.5, edgecolors="none")
    ax.axhline(0, color="#b29a66", lw=1)
    ax.set(xlabel=f"Prediction ({unit})", ylabel=f"Residual ({unit})", title="Out-of-fold residuals")
    save(fig, "residuals")
    fig, ax = plt.subplots(figsize=(8, 5))
    values = importance.groupby("feature").gain.mean().nlargest(20).sort_values()
    ax.barh(values.index, values, color="#72969e")
    ax.set(xlabel="Mean gain importance", title="Feature importance")
    save(fig, "importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    values = shap.drop(columns="fold").abs().mean().nlargest(20).sort_values()
    ax.barh(values.index, values, color="#b4a077")
    ax.set(xlabel=f"Mean |TreeSHAP| ({unit})", title="Held-out TreeSHAP")
    save(fig, "treeshap")
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    for i, feature in enumerate(values.index):
        vals = shap[feature].to_numpy()
        ax.scatter(vals, i + rng.uniform(-.25, .25, len(vals)), s=6, alpha=.45, c="#258b99", edgecolors="none")
    ax.set_yticks(np.arange(len(values)), values.index)
    ax.axvline(0, color="#b29a66", lw=1)
    ax.set(xlabel=f"TreeSHAP ({unit})", title="Held-out contribution distribution")
    save(fig, "treeshap_distribution")
