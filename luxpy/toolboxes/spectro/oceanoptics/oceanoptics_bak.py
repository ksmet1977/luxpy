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
    2. Install python-seabreeze: conda install -c poehlmann python-seabreeze
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
 
Default parameters:
-------------------
 :_INT_TIME_SEC: int, default integration time
 :_CORRECT_DARK_COUNTS: bool, automatic dark count correction supported by some spectrometers
 :_CORRECT_NONLINEARITY: bool, automatic non-linearity correction
 :_TARGET_MAX_CNTS_RATIO: float, aim for e.g. 80% (0.8) of max number of counts
 :_IT_RATIO_INCREASE: float, first stage of int_time optimization: increase int_time by this fraction
 :_MAX_NUMBER_OF_RATIO_INCREASES: int, number of times to apply ration increase before estimating using lin. regr.
 :_DARK_MODEL_INT_TIMES: ndarray, e.g. np.linspace(1e-6, 7.5, 5): array with integration times for dark model
 :_SAVGOL_WINDOW: window for smoothing of dark measurements
 :_SAVGOL_ORDER: order of savgol filter
 :_VERBOSITY: (0: nothing, 1: text, 2: text + graphs)
 :_DARK_MODEL: path and file where default dark_model is stored.
    
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
    
	5. Hopefully, at Ocean Optics, they will, at some point in time, listen to their customers 
	and implement a simple, logical operation of their devices: one that just reads a spectrum 
	at the desired integration time the momemt the function is called and which puts the 
	spectrometer in idle mode when no spectrum is requested.
    
    
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter
from tkinter import messagebox
from scipy.signal import savgol_filter
import time
import os

import seabreeze
seabreeze.use("pyseabreeze")
import seabreeze.spectrometers as sb


__all__ = ['dvc_open','get_spd','create_dark_model','estimate_dark_from_model','plot_spd']

# Init default parameters
_INT_TIME_SEC = 0.5 # default integration time
_CORRECT_DARK_COUNTS = False # automatic dark count correction supported by some spectrometers
_CORRECT_NONLINEARITY = False # automatic non-linearity correction
_TARGET_MAX_CNTS_RATIO = 0.8 # aim for 80% of max number of counts
_IT_RATIO_INCREASE = 1.2 # first stage: increase int_time by this fraction
_MAX_NUMBER_OF_RATIO_INCREASES = 4 # number of times to apply ration increase before estimating using lin. regr.
_DARK_MODEL_INT_TIMES = np.linspace(1e-6, 10, 5) # array with integration times for dark model
_SAVGOL_WINDOW = 1/20.0 # window for smoothing of dark measurements
_SAVGOL_ORDER = 3 # order of savgol filter
_VERBOSITY = 1 # verbosity (0: nothing, 1: text, 2: text + graphs)
_DARK_MODEL = os.path.join(os.path.dirname(__file__),'data','dark_model.dat')

def dvc_open(devnr = 0, verbosity = _VERBOSITY):
    """
    Initialize Ocean Optics spectrometer.
    
    Args:
        :devnr: 
            | 0 or int, optional
            | Number of the device to initialize. Default = 0 (first ocean
            | optics spectrometer of all available)
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
            
    Returns:
        :spec: 
            | handle/struct to initialized spectrometer.
        :devices: 
            | handle to ocean optics device.
    """
    # Get list of connected OO devices:
    devices = []
    while devices == []:
        devices = sb.list_devices()
        time.sleep(0.5)
    if verbosity > 0:
        print("List of Ocean Optics devices:")
        print(devices)
    time.sleep(1)
    
    # Initialize device:
    spec = sb.Spectrometer(devices[devnr])
    time.sleep(1)
    
    # Add other info to spec struct:
    spec._min_int_time_sec = spec._dev.interface._INTEGRATION_TIME_MIN/1e6
    spec._max_int_time_sec = spec._dev.interface._INTEGRATION_TIME_MAX/1e6

    return spec, devices[devnr]

def _getOOcounts(spec, int_time_sec = _INT_TIME_SEC, \
                 correct_dark_counts = _CORRECT_DARK_COUNTS, \
                 correct_nonlinearity = _CORRECT_NONLINEARITY):
    """
    Get a measurement in counts for the specified integration time.
    
    Args:
        :spec: 
            | spectrometer handle
        :int_time_sec: 
            | _INT_TIME_SEC, optional
            | Integration time in seconds.
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
    
    Returns:
        :cnts:
            | ndarray of counts per pixel column (wavelength).
            
    Notes:
        1. Due to the way ocean optics firmware/drivers are implemented, 
    	most spectrometers do not support an abort mode of the standard 'free running mode', 
    	which causes spectra to be continuously stored in a FIFO array. 
    	This first-in-first-out (FIFO) causes a very unpractical behavior of the spectrometers,
    	such that, to ensure one gets a spectrum corresponding to the latest integration time 
    	sent to the device, one is forced to call the spec.intensities() function twice! 
    	This means a simple measurements now takes twice as long, resulting in a sub-optimal efficiency. 
    
    	2. Hopefully, at Ocean Optics, they will, at some point in time, listen to their customers 
    	and implement a simple, logical operation of their devices: one that just reads a spectrum 
    	at the desired integration time the momemt the function is called and which puts the 
    	spectrometer in idle mode when no spectrum is requested.
   """
    spec.integration_time_micros(int_time_sec*1e6) # expects micro secs.
    cnts = spec.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
    cnts = spec.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity) # double call to avoid ending up with wrong buffer values due to crappy programming of ocean optics api
    return cnts


def create_dark_model(spec, dark_model_int_times = _DARK_MODEL_INT_TIMES, \
                      savgol_window = _SAVGOL_WINDOW, \
                      correct_dark_counts = _CORRECT_DARK_COUNTS, \
                      correct_nonlinearity = _CORRECT_NONLINEARITY, \
                      verbosity = _VERBOSITY, auto_close = True):
    """
    Create a dark model to account for readout noise and dark light.
    
    Args:
        :spec: 
            | spectrometer handle
        :dark_model_int_times:
            | _DARK_MODEL_INT_TIMES, optional
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
        :auto_close:
            | True, optional
            | Close spectrometer after measurement.
            
    Returns:
        :dark_model: 
            | ndarray with dark model
            |   first column (from row 1 onwards): integration times (secs)
            |   second column onwards: dark spectra (cnts, with wavelengths
            |   on row 0). 
        
    """
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
            savgol_window = np.int(2*np.round(spec.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
    
    # prepare graphic output:
    if verbosity > 1:
        dark_fig = plt.figure("Dark Model (savgol_window = {:1.1f})". format(savgol_window))    
        ax1 = dark_fig.add_subplot(1, 3, 1) 
        ax2 = dark_fig.add_subplot(1, 3, 2)  
        
    # Measure dark for several integration times:    
    for i,it in enumerate(dark_model_int_times):
        if verbosity > 0:
            print("Measuring dark counts for integration time {:1.0f}/{:1.0f} ({:1.4f}s)".format(i,len(dark_model_int_times),it))
        dark_int_time, dark_cnts = _find_opt_int_time(spec, it, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
        dark_cnts_s = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
        
         # Graphic output
        if verbosity > 1:
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Counts')
            ax1.set_title('Dark Measurements (raw)')
            ax1.plot(spec.wavelengths(), dark_cnts,'.')
         
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Counts')
            ax2.set_title('Dark Measurements (smoothed)')
            ax2.plot(spec.wavelengths(), dark_cnts_s,'.')
            plt.show()
            plt.pause(0.1)
            

        
        if i == 0:
            dark_cnts_arr = dark_cnts
            dark_cnts_s_arr = dark_cnts_s
            sum_dark_cnts = [dark_cnts.sum()]
            dark_its_arr = [dark_int_time]
        else:
            dark_cnts_arr = np.vstack((dark_cnts_arr,dark_cnts))
            dark_cnts_s_arr = np.vstack((dark_cnts_s_arr,dark_cnts_s))
            sum_dark_cnts.append(dark_cnts.sum())
            dark_its_arr.append(dark_int_time)
    
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
        dark_model = np.hstack((np.vstack((np.nan,dark_its_arr[:,None])),\
                            np.vstack((spec.wavelengths(),dark_cnts_s_arr))))
    else: # use non-smoothed dark measurements
        dark_model = np.hstack((np.vstack((np.nan,dark_its_arr[:,None])),\
                            np.vstack((spec.wavelengths(),dark_cnts_arr))))
        
    if auto_close == True:
        spec.close()
        spec = None
        
    return dark_model

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

def estimate_dark_from_model(int_time, dark_model):
    """
    Estimate the dark spectrum for a specified integration time given a dark model.
    
    Args:
        :int_time: 
            | integration time in seconds
        :dark_model: 
            | ndarray with dark model
            |   first column (from row 1 onwards): integration times (secs)
            |   second column onwards: dark spectra (cnts, with wavelengths
            |   on row 0). 
            
    Returns:
        :returns:
            | dark spectrum (row 0: wavelengths, row 1: counts)
    """
    if dark_model.shape[0] > 2: # contains array with dark model
        dark_its_arr = dark_model[1:,0] # integration times
        dark_cnts_arr = dark_model[1:,1:] # dark counts (first axis of dark_model are wavelengths)
        p1,p2 = _find_two_closest(int_time, dark_its_arr)
        dark1 = dark_cnts_arr[p1]
        dark2 = dark_cnts_arr[p2]
        it1 = dark_its_arr[p1]
        it2 = dark_its_arr[p2]
        dark = dark1 + (int_time-it1)*(dark2-dark1)/(it2-it1)
        return np.vstack((dark_model[0,1:],dark)) # add wavelengths and return dark cnts
    elif dark_model.shape[0] == 2: # contains array with dark spectrum
        return dark_model 
    else:
        raise Exception('dark_model does not contain a dark model (.shape[0] > 2) or spectrum (.shape[0] == 2)!')


def _correct_for_dark(spec, cnts, int_time_sec, method = 'dark_model.dat', \
                      savgol_window = _SAVGOL_WINDOW, \
                      correct_dark_counts = _CORRECT_DARK_COUNTS, \
                      correct_nonlinearity = _CORRECT_NONLINEARITY,\
                      verbosity = _VERBOSITY):
    """
    Correct a light spectrum (in counts) with a dark spectrum.
    
    Args:
        :spec: 
            | spectrometer handle
        :cnts: 
            | light spectrum in counts
        :int_time_sec:
            | integration time of light spectrum measurement
        :method:
            | 'dark_model.dat' or str or ndarray, optional
            | If str: 
            |   - 'none': don't perform dark correction
            |   - 'measure': perform a dark measurement with integration time
            |                specified in :int_time_sec:.
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
   
    Returns:
        :returns:
            | ndarray with dark corrected light spectrum in counts.
    """
    if method == 'none':
        return cnts
    elif method == 'measure':
            # Determine odd window_length of savgol filter for smoothing (if 0: no smoothing):
            if savgol_window > 0:
                if isinstance(savgol_window,int):
                    savgol_window = (savgol_window % 2==0) + savgol_window # ensure odd window length
                else:
                    savgol_window = np.int(2*np.round(spec.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
        
            # Ask user response:
            root = tkinter.Tk() #hide tkinter main window
            if verbosity > 0:
                print("Close shutter and press Ok in messagebox to continue with measurement.")
            messagebox.showinfo("Dark Measurement","Close shutter and press Ok to continue with measurement.")
            
            dark_cnts = _getOOcounts(spec, int_time_sec = int_time_sec, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
            if savgol_window > 0: # apply smoothing
                dark_cnts = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
            
            # Ask user response:
            if verbosity > 0:
                print("Dark measurement completed. Press Ok in messagebox to continue with measurement.")
            messagebox.showinfo("Dark Measurement","Dark measurement completed. Press Ok to continue with measurement.")
            root.withdraw()
    else:
        if isinstance(method,str):
            dark_model = pd.read_csv(method, sep =',', header = None).values
        dark_cnts = estimate_dark_from_model(int_time_sec, dark_model)[1] #take second row (first are wavelengths) 
    return cnts - dark_cnts

def _find_opt_int_time(spec, int_time_sec, \
                       correct_dark_counts = False, \
                       correct_nonlinearity = _CORRECT_NONLINEARITY, \
                       verbosity = _VERBOSITY):
    """
    Find optimum integration time and get measured counts.
    
    Args:
        :int_time_sec:
            | == 0: unlimited search for integration time, but < max_int_time
            | >0: fixed integration time
            | <0: find optimum, but <= int_time_sec
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
    
    Returns:
        :int_time_sec:
            | 'optimized' integration time (according to specifications described above)
        :cnts:
            | ndarray with final measured ('optimized') spectrum
            | Note that by default this is a non-dark-count corrected spectrum.
    """
    
    # Get max pixel value and min. integration time:
    max_value = spec._dev.interface._MAX_PIXEL_VALUE
    min_int_time = spec._min_int_time_sec
    
    # Set maximum integration time:
    if int_time_sec > 0:
        max_int_time = None # fixed
            
    elif int_time_sec == 0:
        max_int_time = spec._max_int_time_sec
        int_time_sec = spec._min_int_time_sec
        
    else:
        max_int_time = np.abs(int_time_sec)
        int_time_sec = spec._min_int_time_sec
        if max_int_time > spec._max_int_time_sec:
            max_int_time = spec._max_int_time_sec
    
    # Determine integration time for optimum counts:
    getcnts = lambda it: _getOOcounts(spec, it, correct_dark_counts = False, correct_nonlinearity = correct_nonlinearity)
    is_sat_bool = lambda cnts: (cnts.max() >= max_value) # check for saturation
    counter = 0
    if max_int_time is not None:
        target_cnts_bool = lambda cnts: (cnts.max() < (max_value*_TARGET_MAX_CNTS_RATIO)) # check for max_counts
        target_it_bool = lambda it: (it <= max_int_time) # check for max_int_time
                
        it = min_int_time # start at min. int_time
        cnts = getcnts(it) # get cnts
        its = [it] # store int_time in array
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
                    it = max_int_time
            if verbosity > 0:
                print("Integration time optimization: measuring ... {:1.5f}s".format(it))
            
            # get counts:
            cnts = getcnts(it)

            
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
                print(extra_increase_factor_for_low_light_levels)

#            if verbosity > 0:
#                print('     List of integration times (s):')
#                print(its)
#                print('     List of max. counts:')
#                print(max_cnts)
#                print('\n')

            if verbosity > 1:
                ax_opt1.plot(its[-1],max_cnts[-1],'o')
                plt.show()
                plt.pause(0.1)
            
            # When current fitted int_time or max. cnts differ by less than 10%
            # from previous or when int_time == max_int_time, break while loop 
            # (i.e. sacrifice small gain for increased efficiency):
            if (len(max_cnts) > max_number_of_ratio_increases):
                if ((np.abs(1.0*cnts.max() - max_cnts[-2])/max_cnts[-2]) < 0.1) | ((np.abs(1.0*it - its[-2])/its[-2]) < 0.1) | (it ==  max_int_time): # if max counts changes by less than 1%: break loop
                    if verbosity > 0:
                        print('Break while loop: less than 10% diff between last two max. or int_time values, or int_time == max_int_time.')
                    break

            
        while is_sat_bool(cnts): # if saturated, start reducing int_time again
            it = it / _IT_RATIO_INCREASE
            if verbosity > 0:
                print('Saturated max count value. Reducing integration time to {:1.2f}s'.format(it))
            its.append(it) # keep track of integration times
            cnts = getcnts(it)
            
            max_cnts.append(cnts.max())  # keep track of max counts
            
            if verbosity > 1:
                ax_opt1.plot(its[-1],max_cnts[-1],'s')
                plt.show()
                plt.pause(0.1)
                
        int_time_sec = it
   
    else:
        # Limit integration time to min-max range:
        if int_time_sec < spec._min_int_time_sec:
            int_time_sec = spec._min_int_time_sec
        if int_time_sec > spec._max_int_time_sec:
            int_time_sec = spec._max_int_time_sec

        # get counts:
        cnts = getcnts(int_time_sec)
        if is_sat_bool(cnts):
            if verbosity > 0:
                print('WARNING: Saturated max count value at integration time of {:1.2f}s'.format(int_time_sec))
            
    return int_time_sec, cnts 

def calibrate_spd(cntss, rfltile = None, callamp = None, CALspd = None):
    pass


def get_spd(spec = None, devnr = 0, int_time_sec = _INT_TIME_SEC, \
            correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY, \
            tec_temperature_C = None,  \
            dark_cnts = _DARK_MODEL, savgol_window = _SAVGOL_WINDOW,\
            out = 'spd', spdtype = 'cnts/s', verbosity = _VERBOSITY,
            auto_close = True,\
            callamp = None, CALspd = None):
    """
    Measure a light spectrum.
    
    Args:
        :spec: 
            | spectrometer handle or None, optional
            | If None: function will try to initialize the spectrometer to 
            |   obtain a handle.
        :devnr:
            | Ocean optics device number in a list of all connected OO-devices.
        :int_time_sec:
            | == 0: unlimited search for integration time, but < max_int_time
            | >0: fixed integration time
            | <0: find optimum, but <= int_time_sec
        :correct_dark_counts: 
            | _CORRECT_DARK_COUNTS or boolean, optional
            | True: Automatic (if supported) dark counts subtraction using 'covered'
            | pixels on the spectrometer device.
        :correct_nonlinearity:
            | _CORRECT_NONLINEARITY or boolean, optional
            | True: Automatic non-linearity correction.
        :tec_temperature_C: None, optional
            | Set board temperature on TEC supported spectrometers.
            | NOT YET IMPLEMENTED (13/07/2018)
        :dark_cnts:
            | 'dark_model.dat' or str or ndarray, optional
            | If str: 
            |   - 'none': don't perform dark correction
            |   - 'measure': perform a dark measurement with integration time
            |                specified in :int_time_sec:.
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
        :out: 
            | 'spd' or str, optional
            | Specifies requested output
        :spdtype:
            | 'cnts/s' (default) or 'cnts', optional
            | Output spectrum in counts or in counts/s
        :verbosity:
            | int, optional
            |   0: now intermediate output
            |   1: only text output (print)
            |   2: text + graphical output (print + pyplot)
        :auto_close:
            | True, optional
            | Close spectrometer after measurement.
            
    Returns:
        :spd:
            | ndarray with spectrum. (row 0: wavelengths, row1: cnts(/s))
            
    Notes:
		1. Due to the way ocean optics firmware/drivers are implemented, 
		most spectrometers do not support an abort mode of the standard 
		'free running mode', which causes spectra to be continuously stored 
		in a FIFO array. This first-in-first-out (FIFO) causes a very 
		unpractical behavior of the spectrometers, such that, to ensure one 
		gets a spectrum corresponding to the latest integration time sent to 
		the device, one is forced to call the spec.intensities() function twice! 
		This means a simple measurements now takes twice as long, 
		resulting in a sub-optimal efficiency. 
		
		2. Hopefully, at Ocean Optics, they will, at some point in time, 
		listen to their customers and implement a simple, logical operation 
		of their devices: one that just reads a spectrum at the desired 
		integration time the momemt the function is called and which puts the 
		spectrometer in idle mode when no spectrum is requested.
    """
    
    # Initialize device:
    if spec is None:
        spec, device = dvc_open(devnr = devnr)
    
    # Enable tec and set temperature:
    if tec_temperature_C is not None:
        try:
            spec.tec_set_enable(True)
            spec.tec_set_temperature_C(set_point_C = tec_temperature_C)
            time.sleep(0.5)
            if verbosity > 0:
                print("Device temperature = {:1.1f}°C".format(spec.tec_get_temperature_C()))
        except:
            pass
    
    # Find optimum integration time and get counts (0 unlimited, >0 fixed, <0: find upto)
    int_time_sec, cnts = _find_opt_int_time(spec, int_time_sec, verbosity = verbosity)
    
    # Get cnts anew when correct_dark_counts == True (is set to False in _find_opt_int_time):
    if correct_dark_counts == True:
        cnts = _getOOcounts(spec, int_time_sec, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
    
    # Correct for dark_counts if not supported by device:
    cnts = _correct_for_dark(spec, cnts, int_time_sec, method = dark_cnts, savgol_window = savgol_window, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity, verbosity = verbosity)
    
    # Reset integration time to min. value for fast new measurement (see notes on crappy ocean optics software):
    spec.integration_time_micros(spec._min_int_time_sec*1e6)
    
    # Convert counts to counts/s:
    if spdtype == 'cnts/s':
        cnts = cnts/int_time_sec
        
    # Add wavelengths to spd:
    spd = np.vstack((spec.wavelengths(),cnts))
    
    if auto_close == True:
        spec.close()
        spec = None
    
    # output requested:
    if out == 'spd':
        return spd
    else: 
        return eval(out)

def plot_spd(ax, spd, int_time, sum_cnts = 0, max_cnts = 0):
    """
    Make a spectrum plot.
    
    Args:
        :ax: 
            | axes handle.
        :int_time: 
            | integration time of spectrum.
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
    ax.set_title("integration time = {:1.3f}s, sum_cnts = {:1.0f}, max_cnts = {:1.0f}".format(int_time, sum_cnts,max_cnts))
    plt.pause(0.1)
    return None


#------------------------------------------------------------------------------
# Code testing
if __name__ == '__main__':
    verbosity = 2 # 2: show text and graph output
    
    time.sleep(1) # ensure seabreeze has time to be fully imported.
    
    # Initialize/open spectrometer:
    spec = None
    if ('spec' in locals()) | (spec is None): 
        spec, device = dvc_open(devnr = 0, verbosity = verbosity)
    
    # Set type of measurement    
    case = 'cont' # other options: 'single','cont','list','dark'
    
    if case == 'single': # single measurement
        int_time = -3 # set integration time in secs.
        
        # Measure spectrum in cnts/s and correct for dark (when finished auto close spectrometer):
        spd = get_spd(spec, int_time_sec = int_time, spdtype = 'cnts',dark_cnts='dark_model.dat', verbosity = verbosity)
        
        # Make a plot of the measured spectrum:
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plot_spd(ax,spd,int_time, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
        
    elif case == 'cont': # continuous measurement
        
        # Create figure and axes for graphic results:
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        
        # Start continuous loop and stop using ctrl-c (keyboard interrupt)
        int_time = -0.5
        try:
            while True:
                spd = get_spd(spec,int_time_sec = int_time, verbosity = verbosity, auto_close = False)
                plot_spd(ax,spd,int_time, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
   
        except KeyboardInterrupt:
            spec.close() # manually close spectrometer
            pass
        
    elif case == 'list': # measure list of integration times
        int_times = np.array([3.2,0.8,1.6,3.2,1.6,0.2,0.2,0.2])/20 # illustrate need for two consecutive measurements to get correct spd (see _getOOcounts())
        int_times = np.array([0.1,0.2,0.3,0.4,0.5])/1 # quick example
        
        # Initialize empty arrays:
        sum_cnts = np.empty(int_times.shape)
        max_cnts = np.empty(int_times.shape)
        
        # Start measurement of list of integration times:
        for i,int_time in enumerate(int_times):
            
            # Measure spectrum and store sum and max:
            spd = get_spd(spec,int_time_sec = int_time, spdtype='cnts', verbosity = verbosity, auto_close = False)
            sum_cnts[i] = spd[1].sum()
            max_cnts[i] = spd[1].mean()
            
            # Plot spectrum:
            fig = plt.figure()
            ax  = fig.add_subplot(1, 1, 1)
            plot_spd(ax,spd,int_time, sum_cnts = sum_cnts[i],max_cnts = max_cnts[i])
            
        spec.close() # manually close spectrometer
        
        # Plot sum and max versus integration times:
        fig2 = plt.figure()
        ax1  = fig2.add_subplot(1, 3, 1)
        ax1.plot(int_times,sum_cnts,'ro-')
        ax2  = fig2.add_subplot(1, 3, 2)
        ax2.plot(np.arange(int_times.size), sum_cnts,'bo-')
        ax3  = fig2.add_subplot(1, 3, 3)
        ax3.plot(np.arange(int_times.size), max_cnts,'bo-')

    elif case == 'dark': # create dark_model for dark light/current and readout noise correction
        dark_model = create_dark_model(spec, dark_model_int_times = _DARK_MODEL_INT_TIMES, savgol_window = _SAVGOL_WINDOW, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY, verbosity = verbosity)
        
        # write dark model to file
        pd.DataFrame(dark_model).to_csv('./data/dark_model.dat', index=False, header=False, float_format='%1.4f')
        