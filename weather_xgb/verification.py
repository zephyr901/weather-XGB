"""Synthetic-only UI and frozen-runtime verification. Never reads research files."""
from dataclasses import asdict
from pathlib import Path
import argparse
import multiprocessing as mp
import queue
import re
import time

from .config import dump


def ui_check(folder):
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QComboBox
    from .examples import generate
    from .ui import MainWindow, ShapeDialog

    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    config = generate(folder / 'input')
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    window = MainWindow(config)
    report = []
    for width, height in ((1280, 800), (1920, 1080)):
        window.resize(width, height); window.show(); app.processEvents()
        for page in range(5):
            window.nav.setCurrentRow(page); app.processEvents()
            window.grab().save(str(folder / f'ui_{width}_{height}_{page}.png'))
            report.append(dict(width=width, height=height, page=page, actual_width=window.width(), actual_height=window.height(), dpr=window.devicePixelRatioF()))
        window.nav.setCurrentRow(2); window.advanced_toggle.setChecked(True); app.processEvents()
        window.grab().save(str(folder / f'advanced_{width}_{height}.png'))
        window.advanced_toggle.setChecked(False)
    assert asdict(window.config_value()) == asdict(config), 'UI configuration roundtrip differs'
    texts = [w.text() for cls in (QLabel, QPushButton) for w in window.findChildren(cls)]
    texts += [window.nav.item(i).text() for i in range(window.nav.count())]
    texts += [box.itemText(i) for box in window.findChildren(QComboBox) for i in range(box.count())]
    assert not any(re.search(r'[\u4e00-\u9fff]', text) for text in texts), 'A built-in UI label is not English'
    dialog = ShapeDialog(config.shapes[0], window); dialog.show(); app.processEvents()
    dialog.grab().save(str(folder / 'morphology_dialog.png')); dialog.close()
    window.close()
    dump(dict(status='passed', pages=report, english_label_count=len(texts)), folder / 'layout.json')


def frozen_check(folder):
    from .examples import generate
    from .engine import worker, predict_project
    from .ui import MainWindow
    from PySide6.QtWidgets import QApplication
    import pandas as pd

    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    config = generate(folder / 'input')
    config.presets = config.presets[:1]
    config.grid = [dict(n_estimators=4, max_depth=2, learning_rate=.1, min_child_weight=1, reg_lambda=1)]
    ctx = mp.get_context('spawn'); events = ctx.Queue(); cancel = ctx.Event()
    process = ctx.Process(target=worker, args=(asdict(config), events, cancel, False)); process.start()
    messages = []; output = None; deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            kind, value = events.get(timeout=.25); messages.append([kind, value])
            if kind == 'result': output = Path(value)
            if kind == 'done': break
        except queue.Empty:
            if not process.is_alive(): break
    process.join(timeout=5)
    if process.is_alive(): process.terminate(); process.join()
    events.close()
    errors = [m for m in messages if m[0] == 'error']
    if errors or output is None or process.exitcode != 0:
        raise RuntimeError(f'Runtime worker failed: {errors}; exit={process.exitcode}')
    summary = pd.read_csv(output / 'summary.csv')
    predictions = predict_project(output / summary.experiment.iloc[0] / 'final', config)
    predictions.to_csv(folder / 'reloaded_predictions.csv', index=False)
    assert not predictions.empty, 'Reloaded model produced no predictions'
    assert not re.search(r'[\u4e00-\u9fff]', (output / 'review.md').read_text(encoding='utf-8'))
    assert all((output / row.experiment / 'treeshap.svg').is_file() for row in summary.itertuples())
    app = QApplication.instance() or QApplication([]); app.setStyle('Fusion')
    window = MainWindow(config); window.resize(1280, 800); window.show(); app.processEvents()
    window.load_results(output); window.nav.setCurrentRow(3); app.processEvents()
    window.grab().save(str(folder / 'runtime_results.png')); window.close()
    report = dict(status='passed', child_exit=process.exitcode, experiments=len(summary), predictions=len(predictions), charts_exported=True)
    dump(report, folder / 'verification.json')
    print(report)
    return report


if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('ui', 'runtime'))
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    (ui_check if args.mode == 'ui' else frozen_check)(args.folder)
