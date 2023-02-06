# -*- coding: utf-8 -*-
########################################################################
# <spectropy: a Python package for spectral measurement.>
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
    1. Download and install the seabreeze installer from sourceforge:
    https://sourceforge.net/projects/seabreeze/files/SeaBreeze/installers/
    2. Install python-seabreeze: conda install -c conda-forge seabreeze
    3. Windows: Force the spectrometer to use a libusb driver via Zadig 
    (http://zadig.akeo.ie/)
    4. Install pyusb ("import usb.core", "usb.core.find()" should work before proceeding)
    5. Ready to go!
    
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
    1. Changed read_eeprom_slot() in eeprom.py in **pyseabreeze** because the 
    ubs output used ',' as decimal separator instead of '.' (probably because
    of the use of a french keyboard, despite having system set to use '.' as separator):  
    line 20 in eeprom.py: "return data.rstrip('\x00')" was changed to
    "return data.rstrip('\x00').replace(',','.')"
    
    2. More info on: https://github.com/ap--/python-seabreeze
    
    3. Cooling for supported spectrometers not yet implemented/tested (May 2, 2019).
 
    4. Due to the way ocean optics firmware/drivers are implemented, 
    most spectrometers do not support an abort mode of the standard 'free running mode', 
    which causes spectra to be continuously stored in a FIFO array. 
    This first-in-first-out (FIFO) causes a very unpractical behavior of the spectrometers,
    such that, to ensure one gets a spectrum corresponding to the latest integration time 
    sent to the device, one is forced to call the spec.intensities() function twice! 
    This means a simple measurements now takes twice as long, resulting in a sub-optimal efficiency. 
    
    5. Hopefully, Ocean Optics will at some point implement a more simple and 
    more logical operation of their devices: one that just reads a spectrum 
    at the desired integration time the momemt the function is called and which puts the 
    spectrometer in idle mode when no spectrum is requested.
    
    
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

import time
import os
import tkinter
from tkinter import messagebox
from scipy.signal import savgol_filter

from luxpy import cie_interp, getwlr
from luxpy.utils import np, pd, plt, _EPS

import seabreeze
seabreeze.use("pyseabreeze")
import seabreeze.spectrometers as sb


__all__ = ['dvc_open','dvc_close', 'get_spd','create_dark_model','estimate_dark_from_model','plot_spd', 'get_temperature', 'set_temperature']

# Init default parameters
_TINT = 0.5 # default integration time
_TINT_MAX = None
_CORRECT_DARK_COUNTS = False # automatic dark count correction supported by some spectrometers
_CORRECT_NONLINEARITY = False # automatic non-linearity correction
_TARGET_MAX_CNTS_RATIO = 0.8 # aim for 80% of max number of counts
_IT_RATIO_INCREASE = 1.2 # first stage: increase Tint by this fraction
_MAX_NUMBER_OF_RATIO_INCREASES = 4 # number of times to apply ration increase before estimating using lin. regr.
_DARK_MODEL_TINTS = np.linspace(1e-6, 4, 3) # array with integration times for dark model
_SAVGOL_WINDOW = 1/20.0 # window for smoothing of dark measurements
_SAVGOL_ORDER = 3 # order of savgol filter
_VERBOSITY = 1 # verbosity (0: nothing, 1: text, 2: text + graphs)
_DARK_MODEL_PATH = os.path.join(os.path.dirname(__file__),'data','dark_model.dat')
_ERROR = None # Error value (for some cases, NaN is used anyway!)
_TEMPC = -20.0 # default value of temperature (°C) to cool TEC supporting devices.

def dvc_open(dvc = 0, N = 10, Errors = {}, out = "dvc,Errors", verbosity = _VERBOSITY):
    """
    Open device.
    
    Args:
        :dvc:
            | Device handle or int.
        :N:
            | Maximum number of times to try detecting devices before giving up. 
        :Errors:
            | Dict with error messages.
        :out:
            | "dvc, Errors", optional
            | Requested return.
        :verbosity:
            | 1, optional
            | 0: no printed error message output.
    
    Returns:
        :dvc:
            | Device handle, if succesfull open (_ERROR: failure, nan: closed)
        :Errors:
            | Dict with error messages.
    """
    out = out.replace(' ','')
    try:
        Errors["OpenDevice"] = None
        if isinstance(dvc,int):
    
            # Get list of connected OO devices:
            devices = []
            cntr = 0
            if verbosity > 0:
                print("Trying to detect Ocean Optics devices ...")
            while (devices == []) & (cntr < N): #cnts to avoid infinite loop
                cntr += 1
                devices = sb.list_devices()
                time.sleep(0.5)
            if verbosity > 0:
                print("The following Ocean Optics devices were found: ", devices)
            time.sleep(1)
        
            if devices != []:
                if verbosity > 0:
                    print("Opening device number: {:1.0f}".format(dvc))
                
                # Initialize device:
                dvc = sb.Spectrometer(devices[dvc])
                time.sleep(1)
            
                # Add other info to dvc struct:
                dvc._tint_min = dvc._dev.interface._INTEGRATION_TIME_MIN/1e6
                dvc._tint_max = dvc._dev.interface._INTEGRATION_TIME_MAX/1e6
            
                if dvc._has_nonlinearity_coeffs:
                    if sum(dvc._nc) == 0: # avoid problems with division by zero by messed up coefficients.
                        dvc._nc[0] = 1.0
                   
                # check for tec feature:
                try:
                    dvc.tec_set_enable(True)
                    dvc._has_tec = True
                except:
                    dvc._has_tec = False
                
                # Set global variable _TINT_MAX to device dependent value
                global _TINT_MAX
                if _TINT_MAX is None:
                    _TINT_MAX = dvc._tint_max
                Errors["OpenDevice"] = 0
            else:
                dvc = _ERROR
                Errors["OpenDevice"] = 'dvc_open() could not detect any device, even after {:1.0f} tries. Make sure you have set the usb driver for your spectrometer to libusb using e.g. Zadig!'.format(N)
            
    except:
        Errors["OpenDevice"] = 'dvc_open() fails.'
        dvc = _ERROR 
    finally:
        if out == "dvc,Errors":
            return dvc, Errors
        elif out == "dvc":
            return dvc
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")

def dvc_close(dvc, Errors = {}, close_device = True, out = "dvc,Errors", verbosity = _VERBOSITY):
    """
    Close an open device.
    
    Args:
        :dvc:
            | Device handle or int.
        :Errors:
            | Dict with error messages.
        :close_device:
            | True: try and close device.
            | False: Do nothing.
        :out:
            | "dvc,Errors", optional
            | Requested return.
        :verbosity:
            | 1, optional
            | 0: no printed error message output.
                
    Returns:
        :dvc: 
            | NaN for closed spectrometer (else return dvc handle untouched).
        :Errors:
            | Dict with error messages.
    """
    Errors["CloseDevice"] = None
    out = out.replace(' ','')
    try:
        if (not isinstance(dvc,int)) & close_device:
            if verbosity > 0:
                print("Closing device.")
            dvc.close()
            dvc = np.nan
        Errors["CloseDevice"] = 0
    except:
        Errors["CloseDevice"] = "dvc_close() failed."
        dvc = _ERROR
    finally:
        if out == "dvc,Errors":
            return dvc, Errors
        elif out == "dvc":
            return dvc
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")


def _getOOcounts(dvc, Tint = _TINT, \
                 correct_dark_counts = _CORRECT_DARK_COUNTS, \
                 correct_nonlinearity = _CORRECT_NONLINEARITY,
                 Errors = {}, out = 'cnts,Errors'):
    """
    Get a measurement in counts for the specified integration time.
    
    Args:
        :dvc: 
            | spectrometer handle
        :Tint: 
            | _TINT, optional
            | Integration time in seconds.
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
        :Errors:
            | Dict with error messages.
        :out:
            | "cnts,Errors", optional
            | Requested return.
    
    Returns:
        :cnts:
            | ndarray of counts per pixel column (wavelength).
        :Errors:
            | Dict with error messages.
        
            
    Notes:
        1. Due to the way ocean optics firmware/drivers are implemented, 
        most spectrometers do not support an abort mode of the standard 'free running mode', 
        which causes spectra to be continuously stored in a FIFO array. 
        This first-in-first-out (FIFO) causes a very unpractical behavior of the spectrometers,
        such that, to ensure one gets a spectrum corresponding to the latest integration time 
        sent to the device, one is forced to call the dvc.intensities() function twice! 
        This means a simple measurements now takes twice as long, resulting in a sub-optimal efficiency. 
    
        2. Hopefully, Ocean Optics will at some point implement a more simple and 
        more logical operation of their devices: one that just reads a spectrum 
        at the desired integration time the momemt the function is called and which puts the 
        spectrometer in idle mode when no spectrum is requested.
    """
    out = out.replace(' ','')
    try:
        Errors['MeasureCnts'] = None
        dvc.integration_time_micros(Tint*1e6) # expects micro secs.
        
        # Turn off features some devices do not support.
        if not dvc._has_dark_pixels:
            correct_dark_counts = False
        if not dvc._has_nonlinearity_coeffs:
            correct_nonlinearity = False
            
        cnts = dvc.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
        cnts = dvc.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity) # double call to avoid ending up with wrong buffer values due to poor programming of ocean optics api
        Errors['MeasureCnts'] = 0
    except:
        Errors['MeasureCnts'] = '_getOOcounts () fails'
        cnts = _ERROR
    finally:
        if out == "cnts,Errors":
            return cnts, Errors
        elif out == "cnts":
            return cnts
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")
    

def create_dark_model(dvc, dark_model_Tints = _DARK_MODEL_TINTS, \
                      savgol_window = _SAVGOL_WINDOW, \
                      correct_dark_counts = _CORRECT_DARK_COUNTS, \
                      correct_nonlinearity = _CORRECT_NONLINEARITY, \
                      verbosity = _VERBOSITY, close_device = True, \
                      Errors = {}, out = 'dark_model,Errors', filename = None):
    """
    Create a dark model to account for readout noise and dark light.
    
    Args:
        :dvc: 
            | spectrometer handle or int
        :dark_model_Tints:
            | _DARK_MODEL_TINTS, optional
            | ndarray with increasing integration times at which a 
            | dark measurement is to be performed. 
            | Ideally (to avoid extrapolation) these should span the expected
            | integration times of the light measurements which are to be 
            | corrected.
        :savgol_window: 
            | _SAVGOL_WINDOW, optional
            | int: odd window_length (>0) used in smoothing the measured dark
            |       spectra used to build the model.
            | float: ratio (> 0.0) to calculate the odd window_length as a 
            |       percentage (max. = 1) of the number of wavelengths:
            |       window_length = 2*round(savgol_window*Nwavelengths) + 1
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
        :close_device:
            | True, optional
            | Close spectrometer after measurement.
        :Errors:
            | Dict with error messages.
        :out:
            | "dark_model,Errors", optional
            | Requested return.
        :filename:
            | None or 0 or str, optional
            | None: don't write (default)
            | 0: use default location: _DARK_MODEL_PATH
            | Path + filename to write dark_model to.
            
    Returns:
        :dark_model: 
            | ndarray with dark model
            |   first column (from row 1 onwards): integration times (secs)
            |   second column onwards: dark spectra (cnts, with wavelengths
            |   on row 0). (if error --> np.nan)
        :Errors:
            | Dict with error messages.
        
    """
    Errors["create_dark_model"] = None
    out = out.replace(' ','')
    try:
        
        if isinstance(dvc,int):
            dvc, Errors = dvc_open(dvc = dvc, out='dvc,Errors', Errors = Errors, verbosity = verbosity)
        
        # Ask user response:
        root = tkinter.Tk() #hide tkinter main window for messagebox
        if verbosity > 0:
            print("Close shutter and press Ok in messagebox to continue with measurement.")
        messagebox.showinfo("Dark Model Measurements","Close shutter and press Ok to continue with measurement.")
        
        # Determine odd window_length of savgol filter for smoothing (if 0: no smoothing):
        if savgol_window > 0:
            if isinstance(savgol_window,int):
                savgol_window = (savgol_window % 2==0) + savgol_window # ensure odd window length
            else:
                savgol_window = np.int32(2*np.round(dvc.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
        
        # prepare graphic output:
        if verbosity > 1:
            dark_fig = plt.figure("Dark Model (savgol_window = {:1.1f})". format(savgol_window))    
            ax1 = dark_fig.add_subplot(1, 3, 1) 
            ax2 = dark_fig.add_subplot(1, 3, 2)  
            
        # Measure dark for several integration times:    
        for i,it in enumerate(dark_model_Tints):
            if verbosity > 0:
                print("Measuring dark counts for integration time {:1.0f}/{:1.0f} ({:1.4f}s)".format(i,len(dark_model_Tints),it))
            dark_Tint, dark_cnts, Errors = _find_opt_Tint(dvc, it, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity, Errors = Errors, out = 'Tint,cnts,Errors')
            dark_cnts_s = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
            
             # Graphic output
            if verbosity > 1:
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Counts')
                ax1.set_title('Dark Measurements (raw)')
                ax1.plot(dvc.wavelengths(), dark_cnts,'.')
             
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Counts')
                ax2.set_title('Dark Measurements (smoothed)')
                ax2.plot(dvc.wavelengths(), dark_cnts_s,'.')
                plt.show()
                plt.pause(0.1)
      
            if i == 0:
                dark_cnts_arr = dark_cnts
                dark_cnts_s_arr = dark_cnts_s
                sum_dark_cnts = [dark_cnts.sum()]
                dark_its_arr = [dark_Tint]
            else:
                dark_cnts_arr = np.vstack((dark_cnts_arr,dark_cnts))
                dark_cnts_s_arr = np.vstack((dark_cnts_s_arr,dark_cnts_s))
                sum_dark_cnts.append(dark_cnts.sum())
                dark_its_arr.append(dark_Tint)
        
        # Ask user response:    
        if verbosity > 0:
            print("All dark measurements have finished. Press Ok in messagebox to continue with measurement.")
        messagebox.showinfo("Dark Model Measurements","All dark measurements have finished. Press Ok to continue with measurement.")
        
        dark_its_arr = np.asarray(dark_its_arr)
        sum_dark_cnts = np.asarray(sum_dark_cnts)
        
        # Graphic output:
        if verbosity > 1:
            ax3 =  dark_fig.add_subplot(1, 3, 3) 
            ax3.plot(dark_its_arr,sum_dark_cnts,'bo-')
            ax3.set_xlabel('Integration time (s)')
            ax3.set_ylabel('sum(cnts)')
            ax3.set_title('Integration time vs sum(cnts)')
        
        root.withdraw() # close tkinter main window
        
        # Store integration times and dark counts in nd-array:
        if savgol_window > 0: # use smoothed dark measurements
            dark_model = np.hstack((np.vstack((0,dark_its_arr[:,None])),\
                                np.vstack((dvc.wavelengths(),dark_cnts_s_arr))))
        else: # use non-smoothed dark measurements
            dark_model = np.hstack((np.vstack((0,dark_its_arr[:,None])),\
                                np.vstack((dvc.wavelengths(),dark_cnts_arr))))
            
        dvc, Errors = dvc_close(dvc, Errors = Errors, close_device = close_device, verbosity = verbosity)
        Errors["create_dark_model"] = 0
        try:
            if filename is not None:
                if filename == 0:
                    filename = _DARK_MODEL_PATH
                pd.DataFrame(dark_model).to_csv(filename, index=False, header=False, float_format='%1.5f')
        except:
            print('WARNING: Could not write dark_model to {:s}'.format(filename))
        Errors["create_dark_model"] = 0
    except:
        Errors["create_dark_model"] = 'fails'
        dark_model = np.nan
    finally:
        if out == "dark_model,Errors":
            return dark_model, Errors
        elif out == "dark_model":
            return dark_model
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")

def _find_two_closest(value, values):
    """
    Finds two closest values to value (local helper function).
    
    Args:
        :value: 
            | value to find two closest elements of :values: for.
        :values:
            | list of increasing values.
            
    Returns:
        :p1: 
            | index to 1st closest value (smallest of the two)
        :p2:
            | index to 2nd closest value (largest of the two)
    """
    # find position of two closes values to value:
    if (value > values.min()) & (value < values.max()):
        d = np.abs(values - value)
        p1 = d.argmin()
        d[p1] = d.max() + 1
        p2 = d.argmin()
        p1,p2 = np.sort((p1,p2)) # make sure the smallest (not closest) of values always comes first
    elif value <= values.min():
        p1 = 0
        p2 = 1
    elif value >= values.max():
        p1 = values.shape[0] - 1
        p2 = p1 - 1
    return p1,p2

def estimate_dark_from_model(Tint, dark_model, Errors = {}, out = 'cnts,Errors'):
    """
    Estimate the dark spectrum for a specified integration time given a dark model.
    
    Args:
        :Tint: 
            | integration time in seconds
        :dark_model: 
            | ndarray with dark model
            |   first column (from row 1 onwards): integration times (secs)
            |   second column onwards: dark spectra (cnts, with wavelengths
            |   on row 0). 
        :Errors:
            | Dict with error messages.
        :out:
            | "dark,Errors", optional
            | Requested return.
            
    Returns:
        :cnts:
            | dark spectrum (row 0: wavelengths, row 1: counts)
        :Errors:
            | dictionary with errors.
    """
    Errors["estimate_dark"] = None
    out = out.replace(' ','')
    try:
        if not np.isnan(dark_model).any():
            
            if dark_model.shape[0] > 2: # contains array with dark model
                dark_its_arr = dark_model[1:,0] # integration times
                dark_cnts_arr = dark_model[1:,1:] # dark counts (first axis of dark_model are wavelengths)
                p1,p2 = _find_two_closest(Tint, dark_its_arr)
                dark1 = dark_cnts_arr[p1]
                dark2 = dark_cnts_arr[p2]
                it1 = dark_its_arr[p1]
                it2 = dark_its_arr[p2]
                dark = dark1 + (Tint-it1)*(dark2-dark1)/(it2-it1)
                cnts = np.vstack((dark_model[0,1:],dark)) # add wavelengths and return dark cnts
                Errors["estimate_dark"] = 0
            elif dark_model.shape[0] == 2: # contains array with dark spectrum
                cnts = dark_model 
                Errors["estimate_dark"] = 0
            else:
                Errors["estimate_dark"] = "dark_model does not contain a dark model (.shape[0] > 2) or spectrum (.shape[0] == 2)! Setting dark to 0."
                print('WARNING: ' + Errors["estimate_dark"])
                cnts = np.array([[0]])
        else:
            Errors["estimate_dark"] = "dark_model contained NaN's (no model available), setting dark to 0."
            cnts = np.array([[0]])
    except:
        Errors["estimate_dark"] = "Fails. Setting dark to 0."
        cnts = np.array([0])
    finally:
        if out == "cnts,Errors":
            return cnts, Errors
        elif out == "cnts":
            return cnts
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")
        
            

def _correct_for_dark(dvc, cnts, Tint, method = 'dark_model.dat', \
                      savgol_window = _SAVGOL_WINDOW, \
                      correct_dark_counts = _CORRECT_DARK_COUNTS, \
                      correct_nonlinearity = _CORRECT_NONLINEARITY,\
                      verbosity = _VERBOSITY, Errors = {}, out = 'cnts,Errors'):
    """
    Correct a light spectrum (in counts) with a dark spectrum.
    
    Args:
        :dvc: 
            | spectrometer handle
        :cnts: 
            | light spectrum in counts
        :Tint:
            | integration time of light spectrum measurement
        :method:
            | 'dark_model.dat' or str or ndarray, optional
            | If str: 
            |   - 'none': don't perform dark correction
            |   - 'measure': perform a dark measurement with integration time
            |                specified in :Tint:.
            |   - 'dark_model.dat' or other filename. Read cvs-file with 
            |       model or dark counts.
            | else: method should contain an ndarray with the dark model or dark cnts.
            |        - ndarray-format for model:
            |           first column (from row 1 onwards): integration times (secs)
            |           second column onwards: dark spectra (cnts, with wavelengths
            |           on row 0). 
            |        - ndarray-format for dark spectrum
            |            row 0: wavelengths and row 1: dark spectrum in counts.
        :savgol_window: 
            | _SAVGOL_WINDOW, optional
            | int: odd window_length (>0) used in smoothing the measured dark
            |       spectra used to build the model.
            | float: ratio (> 0.0) to calculate the odd window_length as a 
            |       percentage (max. = 1) of the number of wavelengths:
            |       window_length = 2*round(savgol_window*Nwavelengths) + 1
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
        :Errors:
            | Dict with error messages.
        :out:
            | "cnts,Errors", optional
            | Requested return.
   
    Returns:
        :cnts:
            | ndarray with dark corrected light spectrum in counts.
        :Errors:
            | dictionary with errors.
    """
    Errors["_correct_for_dark"] = None
    out = out.replace(' ','')
    try:
        if method == 'none':
            dark_cnts = 0
        elif method == 'measure':
                # Determine odd window_length of savgol filter for smoothing (if 0: no smoothing):
                if savgol_window > 0:
                    if isinstance(savgol_window,int):
                        savgol_window = (savgol_window % 2==0) + savgol_window # ensure odd window length
                    else:
                        savgol_window = np.int32(2*np.round(dvc.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
            
                # Ask user response:
                root = tkinter.Tk() #hide tkinter main window
                if verbosity > 0:
                    print("Close shutter and press Ok in messagebox to continue with measurement.")
                messagebox.showinfo("Dark Measurement","Close shutter and press Ok to continue with measurement.")
                dark_cnts, Errors = _getOOcounts(dvc, Tint = Tint, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity, Errors = Errors, out = 'cnts,Errors')
                if savgol_window > 0: # apply smoothing
                    try:
                        dark_cnts = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
                    except:
                        pass
                # Ask user response:
                if verbosity > 0:
                    print("Dark measurement completed. Press Ok in messagebox to continue with measurement.")
                messagebox.showinfo("Dark Measurement","Dark measurement completed. Press Ok to continue with measurement.")
                root.withdraw()
        else:
            if isinstance(method,str):
                dark_model = pd.read_csv(method, sep =',', header = None).values
            dark_cnts, Errors = estimate_dark_from_model(Tint, dark_model, Errors=Errors, out = 'cnts,Errors')
            dark_cnts = dark_cnts[1] #take second row (first are wavelengths) 
        cnts = cnts - dark_cnts
        Errors["_correct_for_dark"] = 0
    except:
        Errors["_correct_for_dark"] = 'Fails. Outputting uncorrected counts!'
        cnts = cnts
    finally:
        if out == "cnts,Errors":
            return cnts, Errors
        elif out == "cnts":
            return cnts
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")
            
        
        

def _find_opt_Tint(dvc, Tint, autoTint_max = _TINT_MAX, \
                       correct_dark_counts = False, \
                       correct_nonlinearity = _CORRECT_NONLINEARITY, \
                       verbosity = _VERBOSITY, Errors = {}, out = 'Tint,cnts,Errors'):
    """
    Find optimum integration time and get measured counts.
    
    Args:
        :Tint:
            | 0 or Float, optional
            | Integration time in seconds. (if 0: find best integration time, but < autoTint_max).
        :autoTint_max:
            | Limit Tint to this value when Tint = 0.
        :correct_dark_counts: 
            | False, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
            | Default is False, to ensure correct check of measurement saturation.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction. 
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
        :Errors:
            | Dict with error messages.
        :out:
            | "Tint,cnts,Errors", optional
            | Requested return.
    
    Returns:
        :Tint:
            | 'optimized' integration time (according to specifications described above)
        :cnts:
            | ndarray with final measured ('optimized') spectrum
            | Note that by default this is a non-dark-count corrected spectrum.
        :Errors:
            | dictionary with errors.
    """
    Errors["_find_opt_Tint"] = None
    out = out.replace(' ','')
    try:
        # Get max pixel value and min. integration time:
        max_value = dvc._dev.interface._MAX_PIXEL_VALUE
        Tint_min = dvc._tint_min
                    
        # Limit max integration time:
        Tint_max = autoTint_max
        if Tint_max is not None:
            if Tint_max > dvc._tint_max:
                Tint_max = dvc._tint_max 
            if Tint > Tint_max:
                Tint = Tint_max
        else:
            if Tint > dvc._tint_max:
                Tint = dvc._tint_max
                
        # Setup integration time and limit if necessary:
        if Tint > 0:
            Tint_max = None # None means fixed integration time  
            if Tint > dvc._tint_max:
                Tint = dvc._tint_max
        elif Tint == 0:
            if autoTint_max is not None:
                if autoTint_max > dvc._tint_max:
                    autoTint_max = dvc._tint_max 
                if Tint > autoTint_max:
                    Tint = autoTint_max
                Tint_max = autoTint_max
            else:
                Tint_max = dvc._tint_max
                
        # Ensure Tint is high enough (Tint == 0 is encoded by Tint_max == None)
        if Tint < Tint_min:
            Tint = Tint_min
        
        # Determine integration time for optimum counts:
        getcnts = lambda it, Errors: _getOOcounts(dvc, it, correct_dark_counts = False, correct_nonlinearity = correct_nonlinearity, Errors = Errors, out='cnts,Errors')
        is_sat_bool = lambda cnts: (cnts.max() >= max_value) # check for saturation
        counter = 0
        if Tint_max is not None:
            target_cnts_bool = lambda cnts: (cnts.max() < (max_value*_TARGET_MAX_CNTS_RATIO)) # check for max_counts
            target_it_bool = lambda it: (it <= Tint_max) # check for Tint_max
                    
            it = Tint_min # start at min. Tint
            cnts, Errors = getcnts(it, Errors) # get cnts
            its = [it] # store Tint in array
            max_cnts = [cnts.max()] # store max_cnts in array
            max_number_of_ratio_increases = _MAX_NUMBER_OF_RATIO_INCREASES
            
            if verbosity > 1:
                fig_opt = plt.figure('Integration time optimization')
                ax_opt1 = fig_opt.add_subplot(1,1,1)
                ax_opt1.set_xlabel('Integration time (s)')
                ax_opt1.set_ylabel('Maximum counts')
            extra_increase_factor_for_low_light_levels = 1    
            while (target_cnts_bool(cnts) & target_it_bool(it)) & (not is_sat_bool(cnts)):
                if (len(max_cnts) < (max_number_of_ratio_increases)):
                    it = it * _IT_RATIO_INCREASE * extra_increase_factor_for_low_light_levels
                else:
                    p_max_cnts_vs_its = np.polyfit(max_cnts[-(max_number_of_ratio_increases-1):],its[-(max_number_of_ratio_increases-1):],1) # try and predict a value close to target cnts
                    it = np.polyval(p_max_cnts_vs_its, max_value*_TARGET_MAX_CNTS_RATIO)
                    if not target_it_bool(it):
                        it = Tint_max
                if verbosity > 0:
                    print("Integration time optimization: measuring ... {:1.5f}s".format(it))
                
                # get counts:
                cnts, Errors = getcnts(it, Errors)
    
                
                # Ensure only increasing it-vs-max, 
                # if not: sign of unstable measurement due to to short integration time
                if (cnts.max() > max_cnts[-1]):
                    its.append(it) # keep track of integration times
                    max_cnts.append(cnts.max())  # keep track of max counts
                    counter = 0
                    extra_increase_factor_for_low_light_levels = 1
                elif (len(max_cnts) > max_number_of_ratio_increases):
                    counter += 1 # if max keeps the same, get out of loop
                    if counter > 3:
                        if verbosity > 0:
                            print('Break while loop using counter.')
                        break
                else:
                    extra_increase_factor_for_low_light_levels = extra_increase_factor_for_low_light_levels * 1.5
    
                if verbosity > 1:
                    ax_opt1.plot(its[-1],max_cnts[-1],'o')
                    plt.show()
                    plt.pause(0.1)
                
                # When current fitted Tint or max. cnts differ by less than 10%
                # from previous or when Tint == Tint_max, break while loop 
                # (i.e. sacrifice small gain for increased efficiency):
                if (len(max_cnts) > max_number_of_ratio_increases):
                    if ((np.abs(1.0*cnts.max() - max_cnts[-2])/max_cnts[-2]) < 0.1) | ((np.abs(1.0*it - its[-2])/its[-2]) < 0.1) | (it ==  Tint_max): # if max counts changes by less than 1%: break loop
                        if verbosity > 0:
                            print('Break while loop: less than 10% diff between last two max. or Tint values, or Tint == Tint_max.')
                        break
    
                
            while is_sat_bool(cnts): # if saturated, start reducing Tint again
                it = it / _IT_RATIO_INCREASE
                if verbosity > 0:
                    print('Saturated max count value. Reducing integration time to {:1.2f}s'.format(it))
                its.append(it) # keep track of integration times
                cnts, Errors = getcnts(it, Errors)
                
                max_cnts.append(cnts.max())  # keep track of max counts
                
                if verbosity > 1:
                    ax_opt1.plot(its[-1],max_cnts[-1],'s')
                    plt.show()
                    plt.pause(0.1)
                    
            Tint = it
       
        else:
            # Limit integration time to min-max range:
            if Tint < dvc._tint_min:
                Tint = dvc._tint_min
            if Tint > dvc._tint_max:
                Tint = dvc._tint_max
    
            # get counts:
            cnts, Errors = getcnts(Tint, Errors)
            if is_sat_bool(cnts):
                if verbosity > 0:
                    print('WARNING: Saturated max count value at integration time of {:1.2f}s'.format(Tint))
        Errors["_find_opt_Tint"] = 0 
    except:
        Errors["_find_opt_Tint"] = 'Fails.'
        Tint, cnts = _ERROR, _ERROR
    finally:
        if out == "Tint,cnts,Errors":
            return Tint, cnts, Errors
        elif out == "Tint,cnts":
            return Tint, cnts
        elif out == "Tint,Errors":
            return Tint, Errors
        elif out == "cnts,Errors":
            return cnts, Errors
        elif out == "Tint":
            return Tint
        elif out == "cnts":
            return cnts
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")
        

def cntsps_to_radiom_units(cntsps, REFmeas = None, REFspd = None, RFL = None, \
                           Errors = {}, out = 'spd,Errors'):
    """
    :cntsps:
        | ndarray with measured spectrum. (row 0: wavelengths, row1: cnts/sec)
    :REFmeas:
        | None, optional
        | Used for conversion of measured stimulus spectrum to radiometric units.
        | If not None: ndarray (2,N) with meaured cnts/s (of same device) of reference standard.
    :REFspd:
        | None, optional
        | Used for conversion of measured stimulus spectrum to radiometric units.
        | ndarray (2,N) with spd of reference standard in radiometric units.
    :RFL:
        | None, optional
        | Used to convert illumination to to radiometric units.
        | ndarray (2,N) with spectral reflectance of a tile illuminated by test source.
    :Errors:
        | Dict with error messages.
    :out:
        | "spd,Errors", optional
        | Requested return.

    """
    Errors["cntsps_to_radiom_units"] = None
    out = out.replace(' ','')
    spd = cntsps.copy()
    try:
        if (REFmeas is not None) & (REFspd is not None):
            REFmeas = cie_interp(REFmeas, spd[0,:], kind = 'spd')
            REFspd = cie_interp(REFspd, spd[0,:], kind = 'spd')
            REFspd[1,np.where(REFspd[1,:]==0)] = _EPS # avoid division by zero
            spd[1,:] = spd[1,:]/REFmeas[1,:]*REFspd[1,:]
        if RFL is not None:
            RFL = cie_interp(RFL, spd[0,:], kind = 'rfl')
            RFL[1,np.where(RFL[1,:]==0)] = _EPS # avoid division by zero
            spd[1,:] = spd[1,:]/RFL[1,:]
        Errors["cntsps_to_radiom_units"] = 0
    except:
        Errors["cntsps_to_radiom_units"] = 'Fails.'
        print('WARNING: Could not convert to radiometric units. Output is counts/sec!')
    finally:
        if out == 'spd,Errors':
            return spd, Errors
        elif out == 'spd':
            return spd
        elif out == "Errors":
            return Errors
        else:
            raise Exception("Requested output error.")
 

def set_temperature(dvc=0, tempC = _TEMPC, repeat_get_temp = 2, Errors = {}, out = 'Errors', verbosity = _VERBOSITY):
    """
    Set temperature of ocean optics devive (if tec supported).
    
    Args:
        :dvc:
            | Spectrometer handle or int, optional
            | If int: device will be opened before and closed after getting the temperature.
        :tempC:
            | _TEMPC or float, optional
            | Temperature (°C) to cool device to.
        :repeat_get_temp:
            | 2 or int, optional
            | Number of times to repeat dvc.tec_get_temperature_C() to get temp (min. twice due to firmware error.)
        :Errors:
            | Dict with error messages, optional
        :out:
            | "Errors", optional
            | Requested return.
            
    Returns:
        :Errors:
            | Dict with error messages.
        [:tempC_meas:
            | Temperature in °C.(NaN if no TEC support)]
    """
    out = out.replace(' ','')
    if isinstance(dvc,int):
        was_closed = True
    elif isinstance(dvc,seabreeze.spectrometers.Spectrometer):
        was_closed = False
    else:
        was_closed = True
    try:
        Errors["set_temperature"] = None
        if isinstance(dvc,int):
            dvc,Errors = dvc_open(dvc=dvc,Errors=Errors,verbosity=verbosity)
            was_closed = True
        
        tempC_meas = np.nan 
        if isinstance(dvc,seabreeze.spectrometers.Spectrometer): 
            if dvc._has_tec:
                dvc.tec_set_temperature_C(set_point_C = tempC)
                time.sleep(0.5)
                if ('tempC_meas' in out.split(',')) | (verbosity > 0):
                    tempC_meas, Errors = get_temperature(dvc=dvc, repeat_get_temp = repeat_get_temp, Errors = Errors, out = 'tempC,Errors', verbosity = verbosity)
                if (verbosity > 0):
                    print('Requested device temperature (°C) = {:1.1f}'.format(tempC))                      
        Errors["set_temperature"] = 0
        
    except:
        Errors["set_temperature"] = 'tec_set_temperature_C() fails.'
        tempC_meas = np.nan  
    finally:
        dvc,Errors = dvc_close(dvc = dvc, close_device = was_closed, Errors = Errors, out = 'dvc,Errors')
        if out == 'Errors':
            return Errors
        elif out == 'tempC_meas,Errors':
            return tempC_meas, Errors
        elif out == 'tempC_meas':
            return tempC_meas

           
def get_temperature(dvc=0, repeat_get_temp = 2, Errors = {}, out = 'tempC,Errors', verbosity = _VERBOSITY):
    """
    Get temperature of ocean optics devive (if tec supported).
    
    Args:
        :dvc:
            | Spectrometer handle or int, optional
            | If int: device will be opened before and closed after getting the temperature.
        :repeat_get_temp:
            | 2 or int, optional
            | Number of times to repeat dvc.tec_get_temperature_C() to get temp (min. twice due to firmware error.)
        :Errors:
            | Dict with error messages, optional
        :out:
            | "tempC,Errors", optional
            | Requested return.
            
    Returns:
        :tempC:
            | Temperature in °C (NaN if no TEC support).
        :Errors:
            | Dict with error messages.
    """
    out = out.replace(' ','')
    if isinstance(dvc,int):
        was_closed = True
    elif isinstance(dvc,seabreeze.spectrometers.Spectrometer):
        was_closed = False
    else:
        was_closed = True
    try:
        Errors["get_temperature"] = None
        if isinstance(dvc,int):
            dvc,Errors = dvc_open(dvc=dvc,Errors=Errors,verbosity=verbosity)
        
        if isinstance(dvc,seabreeze.spectrometers.Spectrometer): 
            if dvc._has_tec:
                for i in range(min(2,int(repeat_get_temp))):
                    tempC = dvc.tec_get_temperature_C()# repeat it a few times (due to error in firmware!)
                    time.sleep(0.01)
                if verbosity > 0:
                    print("Device temperature = {:1.1f}°C".format(tempC))     
            else:
                tempC = np.nan
                Errors["get_temperature"] = 0
    
    except:
        Errors["get_temperature"] = 'tec_get_temperature_C() fails.'
        tempC = np.nan
    finally:
        dvc,Errors = dvc_close(dvc = dvc, close_device = was_closed, Errors = Errors, out = 'dvc,Errors')
        if out == 'tempC,Errors':
            return tempC, Errors
        elif out == 'tempC':
            return tempC
        elif out == 'Errors':
            return Errors
        
        


def get_spd(dvc = 0, Tint = _TINT, autoTint_max = _TINT_MAX, \
            correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY, \
            tempC = _TEMPC, repeat_get_temp = 2, \
            dark_cnts = _DARK_MODEL_PATH, savgol_window = _SAVGOL_WINDOW,\
            units = 'cnts/s', verbosity = _VERBOSITY,
            close_device = True, wlstep = None, wlstart = None, wlend = None,\
            REFmeas = None, REFspd = None, RFL = None, \
            out = 'spd,dvc,Errors'):
    """
    Measure a light spectrum.
    
    Args:
        :dvc: 
            | spectrometer handle or int, optional
            | If int: function will try to initialize the spectrometer to 
            |       obtain a handle. The int represents the Ocean Optics device 
            |       number in a list of all connected OO-devices.
        :Tint:
            | 0 or Float, optional
            | Integration time in seconds. (if 0: find best integration time, but < autoTint_max).
        :autoTint_max:
            | Limit Tint to this value when Tint = 0.
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
        :tempC: None, optional
            | Set board temperature on TEC supported spectrometers.
        :repeat_get_temp:
            | 2 or int, optional
            | Number of times to repeat dvc.tec_get_temperature_C() to get temp (min. twice due to firmware error.)
        :dark_cnts:
            | 'dark_model.dat' or str or ndarray, optional
            | If str: 
            |   - 'none': don't perform dark correction
            |   - 'measure': perform a dark measurement with integration time
            |                specified in :Tint:.
            |   - 'dark_model.dat' or other filename. Read cvs-file with 
            |       model or dark counts.
            | else: method should contain an ndarray with the dark model or dark cnts.
            |        - ndarray-format for model:
            |           first column (from row 1 onwards): integration times (secs)
            |           second column onwards: dark spectra (cnts, with wavelengths
            |           on row 0). 
            |        - ndarray-format for dark spectrum
            |           row 0: wavelengths and row 1: dark spectrum in counts.
        :savgol_window: 
            | _SAVGOL_WINDOW, optional
            | int: odd window_length (>0) used in smoothing the measured dark
            |       spectra used to build the model.
            | float: ratio (> 0.0) to calculate the odd window_length as a 
            |       percentage (max. = 1) of the number of wavelengths:
            |       window_length = 2*round(savgol_window*Nwavelengths) + 1
        :units:
            | 'cnts/s' (default) or 'cnts', optional
            | Output spectrum in counts or in counts/s (or radiometric units if callamp and CALspd are not None)
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
        :close_device:
            | True, optional
            | Close spectrometer after measurement.
            | If 'dvc' not in out.split(','): always close!!!
        :REFmeas:
            | None, optional
            | Used for conversion of measured stimulus spectrum to radiometric units.
            | If not None: ndarray (2,N) with meaured cnts/s (of same device) of reference standard.
        :REFspd:
            | None, optional
            | Used for conversion of measured stimulus spectrum to radiometric units.
            | ndarray (2,N) with spd of reference standard in radiometric units.
        :RFL:
            | None, optional
            | Used to convert illumination to to radiometric units.
            | ndarray (2,N) with spectral reflectance of a tile illuminated by test source.
        :out:
            | "spd,dvc,Errors", optional
            | Requested return.
            
    Returns:
        :spd:
            | ndarray with spectrum. (row 0: wavelengths, row1: cnts(/s))
        :dvc:
            | Device handle, if succesfull open (_ERROR: failure, nan: closed)
        :Errors:
            | Dict with error messages.
            
    Notes:
        1. Due to the way ocean optics firmware/drivers are implemented, 
        most spectrometers do not support an abort mode of the standard 
        'free running mode', which causes spectra to be continuously stored 
        in a FIFO array. This first-in-first-out (FIFO) causes a very 
        unpractical behavior of the spectrometers, such that, to ensure one 
        gets a spectrum corresponding to the latest integration time sent to 
        the device, one is forced to call the dvc.intensities() function twice! 
        This means a simple measurements now takes twice as long, 
        resulting in a sub-optimal efficiency. 
        
        2. Hopefully, Ocean Optics will at some point implement a more simple and 
        more logical operation of their devices: one that just reads a spectrum 
        at the desired integration time the momemt the function is called and which puts the 
        spectrometer in idle mode when no spectrum is requested.
    """
    Errors = {} 
    Errors["get_spd"] = None
    out = out.replace(' ','')
    try:
        # Initialize device:
        dvc, Errors = dvc_open(dvc = dvc, Errors = Errors, out = 'dvc,Errors', verbosity = verbosity)
    except:
        raise Exception('Could not open sepctrometer. Try reconnecting (unplug and replug USB)')
    try: 
        # Set temperature (if device has tec support):
        tempC_meas, Errors = set_temperature(dvc=dvc, tempC = tempC, repeat_get_temp = repeat_get_temp, Errors = Errors, out = 'tempC_meas,Errors', verbosity = verbosity)
        
        if verbosity > 0:
            print('Getting spectrum ...')
        
        # Find optimum integration time and get counts (0: unlimited (but < autoTint_max), >0 fixed)
        Tint, cnts,Errors = _find_opt_Tint(dvc, Tint, autoTint_max = autoTint_max, correct_nonlinearity = correct_nonlinearity, verbosity = verbosity, Errors = Errors, out= 'Tint,cnts,Errors')
        
        # Get cnts anew when correct_dark_counts == True (is set to False in _find_opt_Tint):
        if (correct_dark_counts == True) & dvc._has_dark_pixels:
            cnts,Errors = _getOOcounts(dvc, Tint, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity, Errors = Errors, out='cnts,Errors')
    
        # Correct for dark_counts if not supported by device:
        cnts,Errors = _correct_for_dark(dvc, cnts, Tint, method = dark_cnts, savgol_window = savgol_window, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity, verbosity = verbosity, Errors = Errors, out = 'cnts,Errors')
        
        # Reset integration time to min. value for fast new measurement (see notes on crappy ocean optics software):
        dvc.integration_time_micros(dvc._tint_min*1e6)
        
        # Add wavelengths to spd:
        spd = np.vstack((dvc.wavelengths(),cnts))
        
        # Interpolate to requested wavelength range and stepsize:
        if (wlstep is not None) & (wlstart is not None) & (wlend is not None):
            spd = cie_interp(spd, getwlr([wlstart,wlend,wlstep]), kind = 'spd')
        
        # Convert to units:
        if units == 'cnts':
            pass
        
        elif units == 'cnts/s':
            # Convert counts to counts/s:
            spd[1,:] = cnts/Tint
            
        elif (REFmeas is not None) & (REFspd is not None):
            # Convert cnts/s to radiometric units:
            spd[1,:] = cnts/Tint
            spd, Errors = cntsps_to_radiom_units(spd, RFL = RFL, REFmeas = REFmeas, REFspd = REFspd, Errors = Errors, out='spd,Errors')
        
        if "spd" not in out.split(','):
            close_device = True # force close because dvc is not requested as output!
                
        Errors["get_spd"] = int(np.sum([int(bool(x)) for x in Errors.values() if x is not None]) > 0)
    except:
        Errors["get_spd"] = 'Fails.'
        spd = np.array([np.nan])
    finally:
        #Close device if requested.
        dvc, Errors = dvc_close(dvc, close_device = close_device, verbosity = verbosity, Errors = Errors, out = 'dvc,Errors')
        
        # Generate requested return:
        if out == "spd":
            return spd
        elif out == "dvc":
            return dvc
        elif out == "Errors":
            return Errors
        elif out == "spd,Errors":
            return spd, Errors
        elif out == "spd,dvc":
            return spd, dvc
        elif out == "spd,Errors,dvc":
            return spd, Errors, dvc
        elif out == "spd,dvc,Errors":
            return spd, dvc, Errors
        else:
            raise Exception("Requested output error.")

def plot_spd(ax, spd, Tint, sum_cnts = 0, max_cnts = 0):
    """
    Make a spectrum plot.
    
    Args:
        :ax: 
            | axes handle.
        :Tint: 
            | (max) integration time of spectrum.
        :sum_cnts:
            | 0 or int, optional
            | sum of all counts in spectrum.
        :max_cnts:
            | 0 or int, optional
            | max of all counts in spectrum.
    Returns:
        :None:        
    """
    ax.clear()
    ax.plot(spd[0],spd[1],'b')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('counts/s')
    ax.set_title("integration time = {:1.3f}s, sum_cnts = {:1.0f}, max_cnts = {:1.0f}".format(Tint, sum_cnts,max_cnts))
    plt.pause(0.1)
    return None


#------------------------------------------------------------------------------
# Code testing
if __name__ == '__main__':
    verbosity = 2 # 2: show text and graph output
    
    time.sleep(1) # ensure seabreeze has time to be fully imported.
    
    # Initialize/open spectrometer:
    dvc = 0
    if (isinstance(dvc,int)): 
        dvc, Errors = dvc_open(dvc = dvc, out='dvc,Errors', verbosity = verbosity)
    
    # Set type of measurement    
    case = 'single' # other options: 'single','cont','list','dark'
    
    if case == 'single': # single measurement
        autoTint_max = 3 # set integration time in secs.
        
        # Measure spectrum in cnts/s and correct for dark (when finished auto close spectrometer):
        spd,dvc,Errors = get_spd(dvc = dvc, Tint = 1, autoTint_max = autoTint_max, units = 'cnts',dark_cnts='./data/dark_model.dat', verbosity = verbosity,  out = 'spd,dvc,Errors')
        
        # Make a plot of the measured spectrum:
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plot_spd(ax,spd,autoTint_max, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
        
    elif case == 'cont': # continuous measurement
        
        # Create figure and axes for graphic results:
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        
        # Start continuous loop and stop using ctrl-c (keyboard interrupt)
        autoTint_max = 0.5
        try:
            while True:
                spd,dvc,Errors = get_spd(dvc = dvc,Tint = 0.1, autoTint_max = autoTint_max, verbosity = verbosity, close_device = False, out = 'spd,dvc,Errors')
                plot_spd(ax,spd, autoTint_max, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
   
        except KeyboardInterrupt:
            # manually close spectrometer
            dvc, Errors = dvc_close(dvc, Errors = Errors, close_device = True, verbosity = verbosity, out = 'dvc,Errors')

        
    elif case == 'list': # measure list of integration times

        Tints = np.array([3.2,0.8,1.6,3.2,1.6,0.2,0.2,0.2])/20 # illustrate need for two consecutive measurements to get correct spd (see _getOOcounts())
        Tints = np.array([0.1,0.2,0.3,0.4,0.5])/1 # quick example
        
        # Initialize empty arrays:
        sum_cnts = np.empty(Tints.shape)
        max_cnts = np.empty(Tints.shape)
        
        # Start measurement of list of integration times:
        for i,Tint in enumerate(Tints):
            
            # Measure spectrum and store sum and max:
            spd,dvc,Errors = get_spd(dvc = dvc,Tint = Tint, autoTint_max = None, units='cnts', verbosity = verbosity, close_device = False,  out = 'spd,dvc,Errors')
            sum_cnts[i] = spd[1].sum()
            max_cnts[i] = spd[1].mean()
            
            # Plot spectrum:
            fig = plt.figure()
            ax  = fig.add_subplot(1, 1, 1)
            plot_spd(ax,spd,Tint, sum_cnts = sum_cnts[i],max_cnts = max_cnts[i])
            
        # manually close spectrometer:
        dvc, Errors = dvc_close(dvc, Errors = Errors, close_device = True, verbosity = verbosity, out='dvc,Errors')

        
        # Plot sum and max versus integration times:
        fig2 = plt.figure()
        ax1  = fig2.add_subplot(1, 3, 1)
        ax1.plot(Tints,sum_cnts,'ro-')
        ax2  = fig2.add_subplot(1, 3, 2)
        ax2.plot(np.arange(Tints.size), sum_cnts,'bo-')
        ax3  = fig2.add_subplot(1, 3, 3)
        ax3.plot(np.arange(Tints.size), max_cnts,'bo-')

    elif case == 'dark': # create dark_model for dark light/current and readout noise correction
        Errors = {}
        dark_model,Errors = create_dark_model(dvc, dark_model_Tints = _DARK_MODEL_TINTS, savgol_window = _SAVGOL_WINDOW, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY, verbosity = verbosity, Errors=Errors, out = 'dark_model,Errors')
        
        # write dark model to file
        pd.DataFrame(dark_model).to_csv('./data/dark_model.dat', index=False, header=False, float_format='%1.5f')
        