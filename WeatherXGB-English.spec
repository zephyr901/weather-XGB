# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

root = Path(SPECPATH)
# A venv based on conda keeps these CPython dependencies under Library/bin.
# Collect them explicitly when present; standard python.org builds need no extra files.
conda_bin = Path(sys.base_prefix) / 'Library' / 'bin'
runtime_binaries = [(str(conda_bin / name), '.') for name in ('ffi.dll', 'libmpdec-4.dll', 'sqlite3.dll') if (conda_bin / name).is_file()]
a = Analysis([str(root / 'weather_xgb' / 'launcher.py')], pathex=[str(root)],
             binaries=collect_dynamic_libs('xgboost') + runtime_binaries, datas=collect_data_files('xgboost'),
             hiddenimports=['openpyxl', 'xlrd'] + collect_submodules('scipy._external.array_api_compat') + collect_submodules('scipy._external.array_api_extra'),
             hookspath=[], runtime_hooks=[], excludes=['tkinter', 'PyQt5', 'PyQt6', 'IPython', 'notebook'],
             hooksconfig={'matplotlib': {'backends': ['Agg', 'svg']}}, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='WeatherXGB-English',
          debug=False, strip=False, upx=False, console=bool(os.environ.get('WEATHERXGB_CONSOLE')), disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='WeatherXGB-English')
