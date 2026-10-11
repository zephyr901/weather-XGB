"""Silver desktop workbench. All expensive research work runs outside the UI process."""
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys

import pandas as pd
from PySide6.QtCore import Qt, QTimer, QUrl, QLocale
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QStackedWidget, QTabWidget, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QProgressBar, QSplitter, QStyle, QSizePolicy)
from .config import (ProjectConfig, ShapeSource, TableSource, VariableSpec, ExperimentPreset,
                     dump, from_dict, library, migrate_legacy, validate, load_project_file)
from .data import read_table
from .engine import worker, predict_project

STYLE = """
QWidget { background:#f4f6f7; color:#29353d; font-family:'Segoe UI'; font-size:13px; }
QMainWindow { background:#f4f6f7; }
QLabel#brand { font-size:22px; font-weight:600; color:#26353c; }
QLabel#title { font-size:23px; font-weight:600; }
QLabel#eyebrow { color:#538b96; font-size:11px; }
QLabel#muted { color:#76848c; }
QListWidget#nav { background:#e9eef0; border:0; padding:12px 6px; }
QListWidget#nav::item { padding:16px 12px; margin:3px 0; border-radius:5px; }
QListWidget#nav::item:selected { background:#fff; color:#247e8b; border-left:3px solid #a9956e; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background:#fff; border:1px solid #d4dcdf; border-radius:4px; padding:6px; min-height:20px; selection-background-color:#d5eaee; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border:1px solid #5799a4; }
QPushButton { background:#fff; border:1px solid #cdd7db; border-radius:4px; padding:8px 13px; }
QPushButton:hover { border-color:#6399a2; background:#eef7f8; }
QPushButton:disabled { color:#9da9ae; background:#edf0f1; }
QPushButton#primary { background:#267e8b; border-color:#267e8b; color:white; }
QPushButton#primary:hover { background:#216e79; }
QTableWidget { background:white; alternate-background-color:#f5f8f9; border:1px solid #d9e0e3; gridline-color:#edf0f1; }
QTableWidget::item:selected { background:#d7ebef; color:#20434c; }
QHeaderView::section { background:#eaf0f2; border:0; border-bottom:1px solid #d2dce0; padding:8px; color:#50616a; }
QTabWidget::pane { border:0; border-top:1px solid #d2dce0; }
QTabBar::tab { padding:10px 17px; background:#eaf0f2; margin-right:3px; }
QTabBar::tab:selected { background:white; color:#237d8a; border-top:2px solid #b09b70; }
QProgressBar { border:0; background:#dee7ea; height:5px; text-align:center; }
QProgressBar::chunk { background:#4799a7; }
QScrollArea { border:0; }
QSplitter::handle { background:#dce5e8; }
QCheckBox { spacing:7px; }
QStatusBar { background:#e9eff1; color:#65808b; }
"""


def combo(items, current=None):
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    if current is not None:
        box.setCurrentIndex(max(0, box.findData(current)))
    return box


def number(value, minimum=1, maximum=999):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


def action(parent, label, callback, icon=None, primary=False):
    button = QPushButton(label)
    if icon:
        button.setIcon(parent.style().standardIcon(icon))
    button.setToolTip(label)
    button.clicked.connect(callback)
    if primary:
        button.setObjectName("primary")
    return button


def table(frame=None):
    view = QTableWidget()
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QTableWidget.NoEditTriggers)
    view.setSelectionBehavior(QTableWidget.SelectRows)
    view.verticalHeader().hide()
    view.horizontalHeader().setStretchLastSection(True)
    if frame is not None:
        populate(view, frame)
    return view


def populate(view, frame):
    view.clear()
    view.setColumnCount(len(frame.columns))
    view.setRowCount(len(frame))
    view.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for i, row in enumerate(frame.itertuples(index=False, name=None)):
        for j, value in enumerate(row):
            text = f"{value:.5g}" if isinstance(value, float) else str(value)
            view.setItem(i, j, QTableWidgetItem(text))
    view.resizeColumnsToContents()
    for col in range(view.columnCount()):
        view.setColumnWidth(col, min(300, max(100, view.columnWidth(col))))


class SourceEditor(QWidget):
    def __init__(self, source, weather=True):
        super().__init__()
        self.weather = weather
        self.mapping = dict(source.id_map)
        form = QFormLayout(self)
        pathrow = QWidget(); row = QHBoxLayout(pathrow); row.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(source.path)
        row.addWidget(self.path)
        row.addWidget(action(self, "Browse...", self.choose, QStyle.SP_DialogOpenButton))
        form.addRow("Data file", pathrow)
        self.sheet = QLineEdit(source.sheet)
        self.sheet.setPlaceholderText("First worksheet")
        self.header = number(source.header, 1, 1000)
        self.station = QLineEdit(source.station)
        self.single = QLineEdit(source.single_station)
        self.time = QLineEdit(source.time)
        self.layout_box = combo([("Long table", "long"), ("Stations in rows / dates in columns", "wide")], source.layout)
        form.addRow("Worksheet", self.sheet)
        form.addRow("Header row", self.header)
        form.addRow("Station ID column", self.station)
        form.addRow("Single-station ID override", self.single)
        if weather:
            form.addRow("Time column", self.time)
            form.addRow("Table layout", self.layout_box)
        form.addRow("ID mapping", action(self, "Import mapping JSON", self.import_mapping, QStyle.SP_DialogOpenButton))

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data", "", "Data (*.xlsx *.xlsm *.xls *.csv)", options=QFileDialog.DontUseNativeDialog)
        if path:
            self.path.setText(path)

    def import_mapping(self):
        path, _ = QFileDialog.getOpenFileName(self, "Station ID mapping", "", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
        if path:
            try:
                value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if not isinstance(value, dict) or any(not isinstance(v, (str, int)) for v in value.values()):
                    raise ValueError("The mapping must be a JSON object from source IDs to canonical station IDs.")
                self.mapping = {str(k): str(v) for k, v in value.items()}
                QMessageBox.information(self, "Imported", f"{len(value)} station ID mappings")
            except Exception as exc:
                QMessageBox.warning(self, "Mapping error", str(exc))

    def value(self):
        return TableSource(path=self.path.text().strip(), sheet=self.sheet.text().strip(), header=self.header.value(),
                           station=self.station.text().strip(), time=self.time.text().strip(), single_station=self.single.text().strip(),
                           layout=self.layout_box.currentData(), id_map=self.mapping)


class VariableEditor(QWidget):
    def __init__(self, spec):
        super().__init__()
        layout = QVBoxLayout(self)
        self.source = SourceEditor(spec.source)
        columns = QHBoxLayout()
        columns.addWidget(self.source, 1)
        detail = QWidget()
        columns.addWidget(detail, 1)
        layout.addLayout(columns)
        form = QFormLayout(detail)
        self.name = QLineEdit(spec.name); self.unit = QLineEdit(spec.unit); self.column = QComboBox(); self.column.setEditable(True); self.column.addItem(spec.column)
        self.freq = combo([("Hourly", "h"), ("Daily", "D"), ("Monthly", "MS")], spec.frequency)
        self.agg = combo([("Mean", "mean"), ("Sum of interval totals", "sum"), ("Minimum", "min"), ("Maximum", "max")], spec.aggregation)
        self.low = QLineEdit("" if spec.minimum is None else str(spec.minimum))
        self.high = QLineEdit("" if spec.maximum is None else str(spec.maximum))
        self.name.editingFinished.connect(self.suggest_aggregation)
        for label, widget in [("Variable name", self.name), ("Unit", self.unit), ("Observation column", self.column), ("Input frequency", self.freq), ("Aggregation", self.agg), ("Valid minimum", self.low), ("Valid maximum", self.high)]:
            form.addRow(label, widget)
        layout.addWidget(action(self, "Preview and detect columns", self.preview, QStyle.SP_FileDialogContentsView))
        self.preview_table = table(); self.preview_table.setMinimumHeight(160)
        layout.addWidget(self.preview_table)

    def preview(self):
        try:
            frame = read_table(self.source.value(), preview=True)
            current = self.column.currentText()
            self.column.clear(); self.column.addItems(frame.columns)
            if current in frame:
                self.column.setCurrentText(current)
            for widget, candidates in [(self.source.station, ["站号", "station_id", "站点编号", "原站号"]), (self.source.time, ["时间", "date", "datetime", "日期"])]:
                if widget.text() not in frame:
                    found = [c for c in candidates if c in frame]
                    if len(found) == 1:
                        widget.setText(found[0])
            populate(self.preview_table, frame.head(30))
        except Exception as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))

    def value(self):
        return VariableSpec(name=self.name.text().strip(), unit=self.unit.text().strip(), column=self.column.currentText().strip(),
                            frequency=self.freq.currentData(), aggregation=self.agg.currentData(), source=self.source.value(),
                            minimum=float(self.low.text()) if self.low.text().strip() else None,
                            maximum=float(self.high.text()) if self.high.text().strip() else None)

    def suggest_aggregation(self):
        if any(term in self.name.text().lower() for term in ("降水量", "precipitation", "rainfall")):
            self.agg.setCurrentIndex(self.agg.findData("sum"))
            if not self.low.text(): self.low.setText("0")


class ShapeDialog(QDialog):
    def __init__(self, spec=None, parent=None):
        super().__init__(parent)
        spec = spec or ShapeSource()
        self.setWindowTitle("Morphology data and scale"); self.resize(700, 690)
        layout = QVBoxLayout(self)
        form = QFormLayout(); self.scale = QComboBox(); self.scale.setEditable(True); self.scale.addItems(["100", "250", "500", "750", "1000"]); self.scale.setCurrentText(spec.scale)
        form.addRow("Research scale / identifier", self.scale); layout.addLayout(form)
        self.source = SourceEditor(spec.source, weather=False); layout.addWidget(self.source)
        layout.addWidget(action(self, "Read features", self.read, QStyle.SP_FileDialogContentsView))
        self.columns = QTableWidget(0, 2); self.columns.setHorizontalHeaderLabels(["Feature column", "Type"]); self.columns.horizontalHeader().setStretchLastSection(True); layout.addWidget(self.columns)
        for col in spec.numeric + spec.categorical:
            self.add_column(col, "numeric" if col in spec.numeric else "categorical")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def add_column(self, name, kind):
        row = self.columns.rowCount(); self.columns.insertRow(row)
        item = QTableWidgetItem(name); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.columns.setItem(row, 0, item)
        self.columns.setCellWidget(row, 1, combo([("Ignore", "ignore"), ("Numeric", "numeric"), ("Categorical", "categorical")], kind))

    def read(self):
        try:
            frame = read_table(self.source.value(), preview=True)
            self.columns.setRowCount(0)
            for col in frame:
                if col != self.source.station.text():
                    numeric = pd.to_numeric(frame[col], errors="coerce")
                    kind = "numeric" if numeric.notna().any() and numeric.notna().sum() == frame[col].notna().sum() else "ignore"
                    self.add_column(col, kind)
        except Exception as exc:
            QMessageBox.warning(self, "Read failed", str(exc))

    def value(self):
        columns = {"numeric": [], "categorical": []}
        for row in range(self.columns.rowCount()):
            kind = self.columns.cellWidget(row, 1).currentData()
            if kind in columns:
                columns[kind].append(self.columns.item(row, 0).text())
        return ShapeSource(self.scale.currentText().strip(), self.source.value(), **columns)


class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.setWindowTitle("Weather XGB · Meteorological Workbench"); self.resize(1280, 860); self.setMinimumSize(980, 650)
        self.cfg = config or ProjectConfig(); self.process = None; self.events = None; self.cancel_event = None; self.last_output = None; self.close_pending = False
        self.setStyleSheet(STYLE)
        root = QWidget(); self.setCentralWidget(root); outer = QVBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        head = QWidget(); hl = QHBoxLayout(head); hl.setContentsMargins(24, 18, 24, 18)
        brand = QLabel("WEATHER XGB"); brand.setObjectName("brand"); hl.addWidget(brand)
        badge = QLabel("METEOROLOGICAL RESEARCH / ENGLISH"); badge.setObjectName("eyebrow"); hl.addWidget(badge); hl.addStretch()
        hl.addWidget(action(self, "Open project", self.load_project, QStyle.SP_DialogOpenButton)); hl.addWidget(action(self, "Save project", self.save_project, QStyle.SP_DialogSaveButton)); outer.addWidget(head)
        body = QHBoxLayout(); body.setSpacing(0)
        self.nav = QListWidget(); self.nav.setObjectName("nav"); self.nav.setFixedWidth(215)
        self.nav.addItems(["Project and data", "Experiment presets", "Training", "Results", "Prediction"])
        body.addWidget(self.nav); self.stack = QStackedWidget(); body.addWidget(self.stack, 1); outer.addLayout(body, 1)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.currentRowChanged.connect(lambda _: QTimer.singleShot(0, self.show_chart))
        self.build_data(); self.build_presets(); self.build_training(); self.build_results(); self.build_prediction()
        self.nav.setCurrentRow(0)
        self.statusBar().showMessage("Ready · Local computation")
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(150)

    def page(self, title):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(26, 22, 26, 18); layout.setSpacing(15)
        heading = QLabel(title); heading.setObjectName("title"); layout.addWidget(heading); self.stack.addWidget(page)
        return page, layout

    def build_data(self):
        _, layout = self.page("Project and data")
        form = QFormLayout(); self.project_name = QLineEdit(self.cfg.name); form.addRow("Project name", self.project_name); layout.addLayout(form)
        self.data_tabs = QTabWidget(); self.variable_editors = []
        for i, spec in enumerate(self.cfg.variables):
            editor = VariableEditor(spec); self.variable_editors.append(editor)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(editor); self.data_tabs.addTab(scroll, f"Weather variable {i+1}")
        shape_page = QWidget(); shape_layout = QVBoxLayout(shape_page)
        self.shape_table = table(); shape_layout.addWidget(self.shape_table)
        buttons = QHBoxLayout()
        for text, fn, icon in [("Add scale", self.add_shape, QStyle.SP_FileIcon), ("Edit scale", self.edit_shape, QStyle.SP_FileDialogDetailedView), ("Remove scale", self.remove_shape, QStyle.SP_TrashIcon)]:
            buttons.addWidget(action(self, text, fn, icon))
        buttons.addStretch(); shape_layout.addLayout(buttons); self.data_tabs.addTab(shape_page, "Morphology features")
        bg = QWidget(); bl = QVBoxLayout(bg); self.background_tabs = QTabWidget(); self.background_editors = []; self.background_checks = []
        for i, variable in enumerate(self.cfg.variables):
            spec = next((v for v in self.cfg.backgrounds if v.name == variable.name), VariableSpec(name=variable.name, unit=variable.unit, column=variable.column, frequency=variable.frequency, aggregation=variable.aggregation))
            panel = QWidget(); pl = QVBoxLayout(panel); enabled = QCheckBox("Use an independent background file"); enabled.setChecked(any(v.name == variable.name for v in self.cfg.backgrounds)); pl.addWidget(enabled)
            editor = VariableEditor(spec); pl.addWidget(editor); self.background_editors.append(editor); self.background_checks.append(enabled)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); self.background_tabs.addTab(scroll, f"Variable {i+1}")
        bl.addWidget(self.background_tabs); self.data_tabs.addTab(bg, "Independent background")
        layout.addWidget(self.data_tabs); self.refresh_shapes()

    def refresh_shapes(self):
        populate(self.shape_table, pd.DataFrame([{"Scale": s.scale, "File": Path(s.source.path).name, "Numeric features": ", ".join(s.numeric), "Categorical features": ", ".join(s.categorical)} for s in self.cfg.shapes], columns=["Scale", "File", "Numeric features", "Categorical features"]))

    def add_shape(self):
        d = ShapeDialog(parent=self)
        if d.exec():
            self.cfg.shapes.append(d.value()); self.refresh_shapes()

    def edit_shape(self):
        row = self.shape_table.currentRow()
        if row >= 0:
            d = ShapeDialog(self.cfg.shapes[row], self)
            if d.exec():
                self.cfg.shapes[row] = d.value(); self.refresh_shapes()

    def remove_shape(self):
        row = self.shape_table.currentRow()
        if row >= 0:
            self.cfg.shapes.pop(row); self.refresh_shapes()

    def build_presets(self):
        _, layout = self.page("Experiment presets")
        toolbar = QHBoxLayout(); self.library_box = combo([(p.name, i) for i, p in enumerate(library())]); toolbar.addWidget(self.library_box)
        toolbar.addWidget(action(self, "Add preset", self.add_preset, QStyle.SP_FileIcon)); toolbar.addWidget(action(self, "Import", self.import_preset, QStyle.SP_DialogOpenButton)); toolbar.addWidget(action(self, "Export", self.export_preset, QStyle.SP_DialogSaveButton)); toolbar.addStretch(); layout.addLayout(toolbar)
        self.preset_tabs = QTabWidget(); self.preset_tabs.setTabsClosable(True); self.preset_tabs.tabCloseRequested.connect(self.remove_preset); layout.addWidget(self.preset_tabs)
        self.preset_editors = []
        for preset in self.cfg.presets:
            self.insert_preset(preset)
        self.group_json = QPlainTextEdit(json.dumps(self.cfg.groups, ensure_ascii=False, indent=2)); self.group_json.setMaximumHeight(110)
        layout.addWidget(QLabel("Feature groups")); layout.addWidget(self.group_json)
        shortcuts = QHBoxLayout()
        for text, cols in [("All", []), ("Green space", ["GCR", "GED", "GSI", "GAI"]), ("Buildings", ["BCR", "FAI", "BMH", "BHD", "BSI"])]:
            shortcuts.addWidget(action(self, text, lambda checked=False, t=text, c=cols: self.add_group(t, c)))
        shortcuts.addStretch(); layout.addLayout(shortcuts)

    def add_group(self, name, cols):
        try:
            groups = json.loads(self.group_json.toPlainText()); groups[name] = cols
            self.group_json.setPlainText(json.dumps(groups, ensure_ascii=False, indent=2))
        except Exception as exc:
            QMessageBox.warning(self, "Feature group format", str(exc))

    def insert_preset(self, p):
        page = QWidget(); form = QFormLayout(page); fields = {}
        fields["name"] = QLineEdit(p.name)
        fields["grain"] = combo([(n, g) for n, g in [("Hourly", "hourly"), ("Daily", "daily"), ("Monthly", "monthly"), ("Seasonal", "seasonal"), ("Day/night", "daynight"), ("Month-balanced", "balanced")]], p.grain)
        fields["task"] = combo([("Morphology association", "association"), ("One-step-ahead forecast", "forecast")], p.task)
        fields["features"] = combo([("Morphology features", "morphology"), ("Calendar + morphology", "calendar"), ("Background + morphology", "background")], p.features)
        fields["coverage"] = QDoubleSpinBox(); fields["coverage"].setRange(.01, 1); fields["coverage"].setSingleStep(.05); fields["coverage"].setValue(p.coverage)
        fields["day_start"] = number(p.day_start, 0, 23); fields["day_end"] = number(p.day_end, 0, 23); fields["season_start"] = number(p.season_start, 1, 12)
        fields["reference_ids"] = QLineEdit(",".join(p.reference_ids)); fields["reference_min"] = number(p.reference_min)
        labels = ["Name", "Time grain", "Research task", "Feature set", "Minimum coverage", "Daytime start hour", "Daytime end hour", "First season start month", "Reference station IDs (comma-separated)", "Minimum valid reference stations"]
        for label, widget in zip(labels, fields.values()):
            form.addRow(label, widget)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self.preset_tabs.addTab(scroll, p.name); self.preset_editors.append(fields)

    def preset_value(self, fields):
        values = {}
        for key, widget in fields.items():
            if isinstance(widget, QComboBox): values[key] = widget.currentData()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)): values[key] = widget.value()
            else: values[key] = widget.text().strip()
        values["reference_ids"] = [v.strip() for v in values["reference_ids"].replace("，", ",").split(",") if v.strip()]
        return ExperimentPreset(**values)

    def add_preset(self):
        self.insert_preset(library()[self.library_box.currentData()]); self.preset_tabs.setCurrentIndex(len(self.preset_editors)-1)

    def remove_preset(self, row):
        self.preset_editors.pop(row); self.preset_tabs.removeTab(row)

    def import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import experiment preset", "", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
        if path:
            try: self.insert_preset(ExperimentPreset(**json.loads(Path(path).read_text(encoding="utf-8-sig"))))
            except Exception as exc: QMessageBox.warning(self, "Invalid preset", str(exc))

    def export_preset(self):
        row = self.preset_tabs.currentIndex()
        if row >= 0:
            path, _ = QFileDialog.getSaveFileName(self, "Export experiment preset", "preset.json", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
            if path: dump(self.preset_value(self.preset_editors[row]), path)

    def build_training(self):
        _, outer_layout = self.page("Training")
        training_scroll = QScrollArea(); training_scroll.setWidgetResizable(True)
        training_body = QWidget(); layout = QVBoxLayout(training_body)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(14)
        training_scroll.setWidget(training_body); outer_layout.addWidget(training_scroll)
        form = QFormLayout(); self.output = QLineEdit(self.cfg.output_dir); outputrow = QWidget(); ol = QHBoxLayout(outputrow); ol.setContentsMargins(0,0,0,0); ol.addWidget(self.output); ol.addWidget(action(self,"Browse...", self.choose_output, QStyle.SP_DirIcon)); form.addRow("Output directory", outputrow)
        self.fold_widgets = {}
        for key, label in [("outer_folds", "Station outer folds"), ("inner_folds", "Station inner folds"), ("time_outer", "Time outer splits"), ("time_inner", "Time inner splits"), ("n_jobs", "Threads"), ("seed", "Random seed")]:
            widget = number(getattr(self.cfg, key), 0 if key == "seed" else 1, 2147483647 if key == "seed" else 64); self.fold_widgets[key] = widget; form.addRow(label, widget)
        advanced = QWidget(); advanced.setLayout(form)
        advanced.setMinimumHeight(310)
        self.advanced_toggle = QCheckBox("Advanced training settings"); self.advanced_toggle.setChecked(False); layout.addWidget(self.advanced_toggle); layout.addWidget(advanced); advanced.hide(); self.advanced_toggle.toggled.connect(advanced.setVisible)
        self.parameter_keys = ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 'reg_lambda']
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(['Trees', 'Max depth', 'Learning rate', 'Min child weight', 'L2 regularization'])
        self.parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parameter_table.setFixedHeight(160)
        self.parameter_table.verticalHeader().setDefaultSectionSize(42)
        parameter_panel = QWidget(); parameter_layout = QVBoxLayout(parameter_panel); parameter_layout.setContentsMargins(0,0,0,0)
        parameter_layout.addWidget(self.parameter_table)
        parameter_actions = QHBoxLayout()
        parameter_actions.addWidget(action(self, 'Add candidate', self.add_parameters, QStyle.SP_FileIcon))
        parameter_actions.addWidget(action(self, 'Remove candidate', self.remove_parameters, QStyle.SP_TrashIcon))
        parameter_actions.addStretch(); parameter_layout.addLayout(parameter_actions)
        for params in self.cfg.grid: self.add_parameters(params)
        layout.addWidget(parameter_panel); parameter_panel.hide(); self.advanced_toggle.toggled.connect(parameter_panel.setVisible)
        toolbar = QHBoxLayout(); self.check_button = action(self, "Check and build experiment plan", lambda: self.start(True), QStyle.SP_DialogApplyButton); self.run_button = action(self, "Run experiments", lambda: self.start(False), QStyle.SP_MediaPlay, True); self.stop_button = action(self, "Stop", self.stop, QStyle.SP_MediaStop); self.stop_button.setEnabled(False)
        for b in [self.check_button, self.run_button, self.stop_button]: toolbar.addWidget(b)
        toolbar.addStretch(); layout.addLayout(toolbar)
        self.plan_table = table(); self.plan_table.setMinimumHeight(160); layout.addWidget(self.plan_table, 1)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); layout.addWidget(self.progress)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(190); layout.addWidget(self.log)
        self.log.setMinimumHeight(100)
        self.advanced_toggle.toggled.connect(lambda expanded: training_body.setMinimumHeight(1020 if expanded else 450))

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Results directory", options=QFileDialog.DontUseNativeDialog)
        if path: self.output.setText(path)

    def add_parameters(self, params=None):
        if not isinstance(params, dict): params = ProjectConfig().grid[0]
        row = self.parameter_table.rowCount(); self.parameter_table.insertRow(row)
        for col, key in enumerate(self.parameter_keys):
            if key in ('n_estimators', 'max_depth'):
                widget = number(params[key], 1, 10000 if key == 'n_estimators' else 32)
            else:
                widget = QDoubleSpinBox(); widget.setDecimals(5); widget.setRange(.00001 if key == 'learning_rate' else 0, 1 if key == 'learning_rate' else 100000); widget.setValue(params[key])
            self.parameter_table.setCellWidget(row, col, widget)

    def remove_parameters(self):
        row = self.parameter_table.currentRow()
        if row >= 0: self.parameter_table.removeRow(row)

    def parameter_values(self):
        return [{key: self.parameter_table.cellWidget(row,col).value() for col,key in enumerate(self.parameter_keys)} for row in range(self.parameter_table.rowCount())]

    def config_value(self):
        c = ProjectConfig(name=self.project_name.text().strip(), variables=[v.value() for v in self.variable_editors], shapes=self.cfg.shapes,
                          presets=[self.preset_value(p) for p in self.preset_editors], backgrounds=[v.value() for v, check in zip(self.background_editors, self.background_checks) if check.isChecked()],
                          groups=json.loads(self.group_json.toPlainText()), grid=self.parameter_values(), output_dir=self.output.text().strip(),
                          **{k: w.value() for k, w in self.fold_widgets.items()})
        if not isinstance(c.groups, dict) or any(not isinstance(v, list) for v in c.groups.values()):
            raise ValueError("Feature groups must be a JSON object mapping names to column lists.")
        return c

    def start(self, check_only):
        if self.process and self.process.is_alive(): return
        try:
            c = self.config_value(); validate(c)
        except Exception as exc:
            QMessageBox.warning(self, "Project settings", str(exc)); return
        if not check_only and getattr(self, "checked_config", None) != asdict(c):
            self.statusBar().showMessage("Check the current configuration before running experiments.")
            self.start(True); return
        if not check_only and not getattr(self, "runnable", False):
            QMessageBox.warning(self, "Cannot run", "No presets are ready to run. Review the experiment plan."); return
        self.nav.setCurrentRow(2); self.log.clear(); self.progress.setRange(0, 0)
        ctx = mp.get_context("spawn"); self.events = ctx.Queue(); self.cancel_event = ctx.Event()
        self.process = ctx.Process(target=worker, args=(asdict(c), self.events, self.cancel_event, check_only)); self.process.start()
        self.running_config = asdict(c); self.run_button.setEnabled(False); self.check_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Checking" if check_only else "Experiments running")
        self.worker_done = False

    def stop(self):
        if self.cancel_event:
            self.cancel_event.set(); self.statusBar().showMessage("Stopping after the current fit finishes")

    def poll(self):
        if not self.events: return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log": self.log.appendPlainText(value)
                elif kind == "output": self.last_output = Path(value); self.log.appendPlainText(value)
                elif kind == "progress": self.progress.setRange(0, value[1]); self.progress.setValue(value[0])
                elif kind == "checked":
                    self.checked_config = self.running_config
                    self.runnable = any(r["status"] == "ready" for r in value["reports"])
                    populate(self.plan_table, pd.DataFrame(value["reports"]).rename(columns={"target":"Target", "preset":"Preset", "task":"Task", "status":"Status", "rows":"Samples", "stations":"Stations", "models":"Models", "fits":"Fits", "reason":"Check details"}))
                    fits = sum(r["fits"] for r in value["reports"]); self.log.appendPlainText(f"Check complete · {fits} estimated fits")
                    self.statusBar().showMessage("Check complete · Ready to run" if self.runnable else "Check complete · No runnable experiments")
                elif kind == "result": self.load_results(Path(value)); self.nav.setCurrentRow(3); self.statusBar().showMessage("Experiments complete")
                elif kind == "predicted":
                    self.prediction_frame = value; populate(self.prediction_table, value); self.nav.setCurrentRow(4); self.statusBar().showMessage("Prediction complete")
                elif kind == "error": self.log.appendPlainText(value); self.statusBar().showMessage("Cancelled" if self.cancel_event.is_set() else "Incomplete"); QMessageBox.warning(self, "Experiment incomplete", value)
                elif kind == "done": self.worker_done = True
        except queue.Empty: pass
        if self.process and not self.process.is_alive():
            self.process.join(timeout=.1)
            if not self.worker_done:
                self.log.appendPlainText(f"Worker exited with code {self.process.exitcode}. Completed folds remain in the output directory.")
            self.process = None; self.events.close(); self.events = None
            self.run_button.setEnabled(True); self.check_button.setEnabled(True); self.stop_button.setEnabled(False)
            if self.progress.maximum() == 0: self.progress.setRange(0, 100); self.progress.setValue(0)
            if self.close_pending: self.close()

    def build_results(self):
        _, layout = self.page("Results")
        toolbar = QHBoxLayout(); toolbar.addWidget(action(self, "Load results", self.choose_results, QStyle.SP_DialogOpenButton)); toolbar.addWidget(action(self, "Open directory", self.open_output, QStyle.SP_DirIcon)); toolbar.addStretch(); layout.addLayout(toolbar)
        self.summary = table(); self.summary.setMaximumHeight(220); self.summary.itemSelectionChanged.connect(self.select_experiment); layout.addWidget(self.summary)
        self.result_tabs = QTabWidget(); layout.addWidget(self.result_tabs, 1)
        self.image = QLabel("No experiment results yet"); self.image.setAlignment(Qt.AlignCenter); self.image.setMinimumSize(200, 160); self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image_page = QWidget(); il = QVBoxLayout(image_page); self.chart = combo([("Observed vs predicted", "observed_predicted"), ("Time series", "time_series"), ("Residuals", "residuals"), ("Importance", "importance"), ("TreeSHAP", "treeshap"), ("Contribution distribution", "treeshap_distribution")]); self.chart.currentIndexChanged.connect(self.show_chart); il.addWidget(self.chart); il.addWidget(self.image, 1); self.result_tabs.addTab(image_page, "Charts")
        self.metrics_table = table(); self.result_tabs.addTab(self.metrics_table, "Fold metrics")
        self.review = QPlainTextEdit(); self.review.setReadOnly(True); self.result_tabs.addTab(self.review, "Research notes")
        self.result_frame = pd.DataFrame(); self.result_folder = None
        self.image.installEventFilter(self)

    def choose_results(self):
        path = QFileDialog.getExistingDirectory(self, "Load current or legacy results", options=QFileDialog.DontUseNativeDialog)
        if path:
            try: self.load_results(Path(path))
            except Exception as exc: QMessageBox.warning(self, "Failed to load results", str(exc))

    def load_results(self, folder):
        self.last_output = folder
        candidates = [folder / "summary.csv", folder / "cv_summary_by_radius.csv", folder / "cv_summary.csv"]
        source = next((p for p in candidates if p.exists()), None)
        if source is None: raise ValueError("No results summary was found in this directory.")
        self.result_frame = pd.read_csv(source)
        front = [c for c in ['target','preset','scale','group','r2','rmse','mae','unit','rows','stations','baseline_rmse','persistence_rmse','background_only_rmse'] if c in self.result_frame]
        shown = self.result_frame[front + [c for c in self.result_frame if c not in front]].rename(columns={'target':'Target','preset':'Preset','scale':'Scale','group':'Feature groups','r2':'R²','rmse':'RMSE','mae':'MAE','unit':'Unit','rows':'Samples','stations':'Stations','baseline_rmse':'Mean baseline RMSE','persistence_rmse':'Persistence RMSE','background_only_rmse':'Background baseline RMSE'})
        populate(self.summary, shown)
        self.review.setPlainText((folder / "review.md").read_text(encoding="utf-8") if (folder / "review.md").exists() else "Legacy results (read-only)")
        if len(self.result_frame): self.summary.selectRow(0)

    def select_experiment(self):
        row = self.summary.currentRow()
        if row < 0 or self.last_output is None: return
        self.result_folder = self.last_output / str(self.result_frame.iloc[row].get("experiment", ""))
        metrics = self.result_folder / "metrics.csv"
        populate(self.metrics_table, pd.read_csv(metrics) if metrics.exists() else pd.DataFrame())
        self.show_chart()
        model = self.result_folder / "final"
        if (model / "manifest.json").exists(): self.model_path.setText(str(model))

    def show_chart(self):
        if self.result_folder:
            path = self.result_folder / (self.chart.currentData() + ".png")
            if path.exists():
                self.image.setPixmap(QPixmap(str(path)).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); return
        self.image.setText("No chart is available for this result")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "image"): QTimer.singleShot(50, self.show_chart)

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if watched is getattr(self, 'image', None) and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self.show_chart)
        return super().eventFilter(watched, event)

    def open_output(self):
        if self.last_output: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def build_prediction(self):
        _, layout = self.page("Prediction")
        form = QFormLayout(); self.model_path = QLineEdit(); row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.addWidget(self.model_path); rl.addWidget(action(self, "Select model", self.choose_model, QStyle.SP_DialogOpenButton)); form.addRow("Model directory", row); layout.addLayout(form)
        toolbar = QHBoxLayout(); toolbar.addWidget(action(self, "Load prediction project", self.choose_prediction_project, QStyle.SP_DialogOpenButton)); toolbar.addWidget(action(self, "Predict", self.predict, QStyle.SP_MediaPlay, True)); toolbar.addWidget(action(self, "Export CSV", self.export_prediction, QStyle.SP_DialogSaveButton)); toolbar.addStretch(); layout.addLayout(toolbar)
        self.prediction_label = QLabel("Prediction data: current project"); self.prediction_label.setObjectName("muted"); layout.addWidget(self.prediction_label)
        self.prediction_table = table(); layout.addWidget(self.prediction_table); self.prediction_config = None; self.prediction_frame = None

    def choose_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select a model directory containing manifest.json", options=QFileDialog.DontUseNativeDialog)
        if path: self.model_path.setText(path)

    def choose_prediction_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Prediction data mapping project", "", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
        if path:
            try:
                self.prediction_config, message = load_project_file(path)
                if message: raise ValueError("Prediction requires a current-format project configuration.")
                self.prediction_label.setText("Prediction data: " + Path(path).name)
            except Exception as exc: QMessageBox.warning(self, "Invalid project", str(exc))

    def predict(self):
        if self.process:
            QMessageBox.warning(self, "Task running", "Wait for the current task to finish."); return
        try:
            folder = self.model_path.text().strip()
            if not (Path(folder) / "manifest.json").exists(): raise ValueError("Select a valid model directory.")
            c = self.prediction_config or self.config_value()
            ctx = mp.get_context("spawn"); self.events = ctx.Queue(); self.cancel_event = ctx.Event()
            self.process = ctx.Process(target=worker, args=(asdict(c), self.events, self.cancel_event, folder)); self.process.start()
            self.worker_done = False; self.run_button.setEnabled(False); self.check_button.setEnabled(False); self.stop_button.setEnabled(True)
            self.statusBar().showMessage("Predicting")
        except Exception as exc: QMessageBox.warning(self, "Prediction failed", str(exc))

    def export_prediction(self):
        if self.prediction_frame is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Prediction results", "predictions.csv", "CSV (*.csv)", options=QFileDialog.DontUseNativeDialog)
            if path: self.prediction_frame.to_csv(path, index=False, encoding="utf-8-sig")

    def save_project(self):
        try:
            c = self.config_value()
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "project.json", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
            if path: dump(c, path)
        except Exception as exc: QMessageBox.warning(self, "Save failed", str(exc))

    def load_project(self):
        if self.process:
            QMessageBox.warning(self, "Task running", "Wait for the current task to finish or stop it before loading a project."); return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "JSON (*.json)", options=QFileDialog.DontUseNativeDialog)
        if not path: return
        try:
            c, message = load_project_file(path)
            if message:
                QMessageBox.information(self, "Legacy configuration converted", message)
                for s in c.shapes:
                    f = read_table(s.source, preview=True)
                    s.numeric = [col for col in f if col not in ("站号", "站名") and pd.to_numeric(f[col], errors="coerce").notna().any()]
            self.cfg = c
            self.variable_editors = []
            while self.stack.count():
                page = self.stack.widget(0); self.stack.removeWidget(page); page.deleteLater()
            self.build_data(); self.build_presets(); self.build_training(); self.build_results(); self.build_prediction(); self.nav.setCurrentRow(0); self.stack.setCurrentIndex(0)
            self.checked_config = None
        except Exception as exc: QMessageBox.warning(self, "Load failed", str(exc))

    def closeEvent(self, event):
        if self.process and self.process.is_alive():
            self.close_pending = True; self.stop(); event.ignore()
        else: event.accept()


def main():
    import argparse
    from . import __version__

    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Weather XGB meteorological research workbench (English)")
    parser.add_argument("--project", type=Path, help="Open a project JSON at startup")
    parser.add_argument("--version", action="version", version=f"Weather XGB {__version__} (English)")
    args = parser.parse_args()
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app = QApplication(sys.argv)
    app.setApplicationName("Weather XGB English")
    app.setStyle("Fusion")
    config = None
    if args.project:
        try:
            config, message = load_project_file(args.project)
            if message:
                QMessageBox.information(None, "Legacy configuration converted", message)
        except Exception as exc:
            QMessageBox.warning(None, "Failed to open project", str(exc))
            sys.exit(1)
    window = MainWindow(config); window.show()
    sys.exit(app.exec())
