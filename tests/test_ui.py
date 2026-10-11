"""Native widget contracts; training uses an actual spawned worker."""
import os
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from weather_xgb.examples import generate
from weather_xgb.ui import MainWindow


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = generate(Path(self.tmp.name)/'input')
        self.cfg.presets = self.cfg.presets[:1]
        self.cfg.grid[0]['n_estimators'] = 3
        self.window = MainWindow(self.cfg)
        self.window.show(); self.app.processEvents()

    def tearDown(self):
        if self.window.process and self.window.process.is_alive():
            self.window.stop()
            self.wait_finished()
        self.window.close(); self.app.processEvents(); self.tmp.cleanup()

    def wait_finished(self):
        deadline = time.monotonic()+120
        while self.window.process and time.monotonic() < deadline:
            self.app.processEvents(); QTest.qWait(20)
        self.assertIsNone(self.window.process, 'Worker did not finish')

    def test_controls_and_configuration(self):
        self.assertEqual(asdict(self.window.config_value()), asdict(self.cfg))
        self.window.add_preset(); self.assertEqual(len(self.window.config_value().presets),2)
        self.window.remove_preset(1)
        self.window.add_parameters();self.assertEqual(len(self.window.parameter_values()),2)
        self.window.parameter_table.selectRow(1);self.window.remove_parameters()
        self.assertEqual(len(self.window.parameter_values()),1)
        self.window.variable_editors[0].preview()
        self.assertEqual(self.window.variable_editors[0].preview_table.rowCount(),30)
        self.window.resize(1280,800);self.app.processEvents()
        self.assertEqual(self.window.width(),1280)

    def test_check_train_results_and_async_prediction(self):
        errors=[]
        with patch.object(QMessageBox,'warning',side_effect=lambda *args:errors.append(args[-1])):
            self.window.nav.setCurrentRow(2);self.app.processEvents()
            QTest.mouseClick(self.window.check_button,Qt.LeftButton)
            self.wait_finished();self.assertTrue(self.window.runnable)
            self.assertEqual(self.window.plan_table.rowCount(),2)
            QTest.mouseClick(self.window.run_button,Qt.LeftButton)
            self.wait_finished()
            self.assertEqual(len(self.window.result_frame),2)
            self.assertTrue((self.window.last_output/'summary.csv').exists())
            self.window.predict();self.wait_finished()
            self.assertGreater(self.window.prediction_table.rowCount(),0)
        self.assertEqual(errors,[])


if __name__=='__main__':unittest.main()
