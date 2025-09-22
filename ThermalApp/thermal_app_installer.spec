# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ThermalVideoApp executable

import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Application entry script
script = 'ThermalAppWithRoiRecording.py'

# Collect required dynamic libraries and data files
# E.g., pythonnet runtime, IR16Filters assemblies, comtypes
binaries = collect_dynamic_libs('pythonnet') + collect_dynamic_libs('comtypes')
# Add LeptonUVC and ManagedIR16Filters DLLs if necessary
# Example: binaries += [('path/to/LeptonUVC.dll', '.')]
binaries += [
    ('LeptonUVC.dll', '.'), 
    ('IR16Filters.dll', '.')
]

datas = collect_data_files('PIL', includes=['tkinter/*', 'ImageTk/*'])
# Add any additional data/assets (icons, config files) here

# Hidden imports that PyInstaller may miss
hidden_imports = [
    'clr',
    'comtypes',
    'LeptonUVC',
    'IR16Filters',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
]

# Analysis: discover imports and collect resources
a = Analysis(
    [script],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

# Python module archive
pyz = PYZ(a.pure, a.zipped_data,
          cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ThermalVideoApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None  # path to .ico if desired
)

# Bundle everything into a single folder or onefile
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='ThermalVideoApp'
)

