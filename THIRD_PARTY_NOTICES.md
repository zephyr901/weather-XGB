# Third-party dependencies

Weather XGB uses NumPy, pandas, scikit-learn, XGBoost, PySide6/Qt, Matplotlib, openpyxl and xlrd. PyInstaller is used to assemble the Windows portable release. Versions of the installed build dependencies are recorded in `requirements-lock.txt` and `dependency_versions.json` in the portable folder.

The portable distribution contains dynamically loaded Qt/PySide6 and other dependency libraries under `_internal`. Copies of available installed dependency license, copying and notice files are collected under `third_party_licenses`; each dependency retains its respective license. The Python/Qt application source and PyInstaller build specification are included in this public repository.

Upstream projects and source locations:

- Python: https://www.python.org/ and https://github.com/python/cpython
- Qt 6.9: https://download.qt.io/archive/qt/6.9/
- PySide6/Qt for Python: https://code.qt.io/pyside/pyside-setup.git/
- NumPy: https://github.com/numpy/numpy
- pandas: https://github.com/pandas-dev/pandas
- scikit-learn: https://github.com/scikit-learn/scikit-learn
- XGBoost: https://github.com/dmlc/xgboost
- Matplotlib: https://github.com/matplotlib/matplotlib
- openpyxl: https://openpyxl.readthedocs.io/
- xlrd: https://github.com/python-excel/xlrd
- PyInstaller: https://github.com/pyinstaller/pyinstaller

No project-wide license has been selected for the original application code in this edition. Third-party dependency licenses are separate from the application's copyright.
