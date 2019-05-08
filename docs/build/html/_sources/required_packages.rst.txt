Imported (required) packages
=============================
Core
---- 
 * import os 
 * import warnings 
 * from collections import OrderedDict as odict 
 * from mpl_toolkits.mplot3d import Axes3D 
 * import colorsys 
 * import itertools 
 * import copy
 * import time
 * import tkinter
 * import ctypes
 * import platform


3e party dependencies (automatic install)
-----------------------------------------
 * import numpy as np 
 * import pandas as pd 
 * import matplotlib.pyplot as plt 
 * import scipy as sp 
 * from scipy import interpolate 
 * from scipy.optimize import minimize 
 * from scipy.spatial import cKDTree 
 * from imageio import imsave
 
 
3e party dependencies (requiring manual install)
------------------------------------------------
To control Ocean Optics spectrometers with spectro toolbox:
 * import seabreeze (conda install -c poehlmann python-seabreeze)
 * pip install pyusb (for use with 'pyseabreeze' backend of python-seabreeze)

