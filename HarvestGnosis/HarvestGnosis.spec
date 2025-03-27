# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('static', 'static'),
        ('models/Experiment2/baseline_cnn.keras', 'models/Experiment2'),
        ('data/preprocessed/class_indices.npy', 'data/preprocessed'),
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'werkzeug.utils',
        'webbrowser',
        'threading',
        'time',
        'cv2',
        'numpy',
        'numpy._core',
        'numpy._core.multiarray',
        'numpy.core._dtype_ctypes',
        'numpy.core._methods',
        'numpy.lib.format',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['nltk', 'scipy.stats'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HarvestGnosis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HarvestGnosis',
)
