Imported (required) packages
=============================
Core
---- 
 * import os 
 * import warnings 
 * import pathlib
 * import importlib
 * from collections import OrderedDict as odict 
 * import colorsys 
 * import itertools 
 * import copy
 * import time
 * import tkinter
 * import ctypes
 * import platform
 * import subprocess
 * import cProfile
 * import pstats
 * import io


Imported 3e party dependencies :
--------------------------------
 * numpy (automatic install)
 * scipy (stats, optimize, interpolate, ...)
 
Lazily imported 3e party dependencies ():
-----------------------------------------
 * matplotlib.pyplot (any graphic output anywhere)
 * imageio (imread(), imsave())
 * openpyxl (in luxpy.utils: read_excel, write_excel) 
 
3e party dependencies (automatic install on import)
---------------------------------------------------
 * import pyswarms (when importing particleswarms from math)
 * import pymoo (when importing pymoo_nsga_ii from math)
 * import harfang as hg (when importing toolbox.stereoscopicviewer)
 
3e party dependencies (requiring manual install)
------------------------------------------------
To control Ocean Optics spectrometers with spectro toolbox:
 * import seabreeze (conda install -c poehlmann python-seabreeze)
 * pip install pyusb (for use with 'pyseabreeze' backend of python-seabreeze)

