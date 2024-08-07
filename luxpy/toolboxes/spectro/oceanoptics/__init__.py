# -*- coding: utf-8 -*-
########################################################################
# <spectropy.oceanoptics: a Python package for spectral measurement with oceanoptics spectrometers.>
# Copyright (C) <2019>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################
"""
Module for spectral measurements using OceanOptics spectrometers 
================================================================

Installation:
-------------
    1. Install oceanoptics drivers (OMNISPAM or OCEANVIEW / SPECTRASUITE software, etc.)
    2. Install python API for seabreeze: conda install -c conda-forge seabreeze
    3. Ready to go!
    
Functions:
----------
 :dvc_open(): initialize spectrometer
 :get_spd(): measure spectrum
 :create_dark_model(): create a model for dark counts
 :estimate_dark_from_model(): estimate dark counts for specified integration time based on model
 :get_temperature(): Get temperature of ocean optics devive (if tec supported; returns NaN if not).
 :set_temperature(): Set temperature of ocean optics devive (if tec supported).

Default parameters (not exported):
----------------------------------
 :_TINT: default integration time in seconds
 :_TINT_MAX: max integration time, If None: get max supported by device
 :_CORRECT_DARK_COUNTS: bool, automatic dark count correction supported by some spectrometers
 :_CORRECT_NONLINEARITY: bool, automatic non-linearity correction
 :_TARGET_MAX_CNTS_RATIO: float, aim for e.g. 80% (0.8) of max number of counts
 :_IT_RATIO_INCREASE: float, first stage of Tint optimization: increase Tint by this fraction
 :_MAX_NUMBER_OF_RATIO_INCREASES: int, number of times to apply ration increase before estimating using lin. regr.
 :_DARK_MODEL_TINTS: ndarray, e.g. np.linspace(1e-6, 7.5, 5): array with integration times for dark model
 :_SAVGOL_WINDOW: window for smoothing of dark measurements
 :_SAVGOL_ORDER: order of savgol filter
 :_VERBOSITY: (0: nothing, 1: text, 2: text + graphs)
 :_DARK_MODEL_PATH: path and file where default dark_model is stored.
 :_ERROR: error value.
 :_TEMPC: default value of temperature (°C) to cool TEC supporting devices.
    
Notes:
------
    1. More info on: https://github.com/ap--/python-seabreeze
    
    2. Due to the way ocean optics firmware/drivers are implemented, 
    most spectrometers do not support an abort mode of the standard 'free running mode', 
    which causes spectra to be continuously stored in a FIFO array. 
    This first-in-first-out (FIFO) causes a very unpractical behavior of the spectrometers,
    such that, to ensure one gets a spectrum corresponding to the latest integration time 
    sent to the device, one is forced to call the spec.intensities() function twice! 
    This means a simple measurements now takes twice as long, resulting in a sub-optimal efficiency. 
    
    3. Hopefully, Ocean Optics will at some point implement a more simple and 
    more logical operation of their devices: one that just reads a spectrum 
    at the desired integration time the momemt the function is called and which puts the 
    spectrometer in idle mode when no spectrum is requested.
    
    4. When using pyseabreeze backend: change read_eeprom_slot() in eeprom.py in **pyseabreeze** because the 
    ubs output used ',' as decimal separator instead of '.' (probably because
    of the use of a french keyboard, despite having system set to use '.' as separator):  
    line 20 in eeprom.py: "return data.rstrip('\x00')" was changed to
    "return data.rstrip('\x00').replace(',','.')"
    
Last updated for seabreeze v1.3.0 (sep 2020) on March 9, 2021.
    
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .oceanoptics import *
#__all__ = oceanoptics.__all__