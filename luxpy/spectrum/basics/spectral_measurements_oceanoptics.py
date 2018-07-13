# -*- coding: utf-8 -*-
########################################################################
# <LUXPY: a Python package for lighting and color science.>
# Copyright (C) <2017>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
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
==================================================================

Installation:
    1. Download and install the seabreeze installer from sourceforge:
    https://sourceforge.net/projects/seabreeze/files/SeaBreeze/installers/
    2. Windows: Force the spectrometer to use a libusb driver via Zadig 
    (http://zadig.akeo.ie/)
    3. Install pyusb ("import usb.core", "usb.core.find()" should work before proceeding)
    4. Ready to go!
    
    
Note:
 1. Changed read_eeprom_slot() in eeprom.py in pyseabreeze because the 
 ubs output used ',' as decimal separator instead of '.' (probably because
 of french keyboard, despite having system set to use '.' as separator):  
 line 20 in eeprom.py: "return data.rstrip('\x00')" was changed to
 "return data.rstrip('\x00').replace(',','.')"
 2. More info on: https://github.com/ap--/python-seabreeze

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter
from tkinter import messagebox
from scipy.signal import savgol_filter
import time

import seabreeze
seabreeze.use("pyseabreeze")
import seabreeze.spectrometers as sb


__all__ = ['initOOdev','getOOspd','create_dark_model','estimate_dark_from_model','plot_spd']

# Init default parameters
_INT_TIME_SEC = 0.5
_CORRECT_DARK_COUNTS = False
_CORRECT_NONLINEARITY = False
_TARGET_MAX_CNTS_RATIO = 0.8 # aim for 80% of max number of counts
_IT_RATIO_INCREASE = 1.1 # first stage: increase int_time by this fraction
_DARK_MODEL_INT_TIMES = np.linspace(1e-6, 7.5, 5) # array with integration times for dark model
_SAVGOL_WINDOW = 1/20.0 # window for smoothing of dark measurements
_SAVGOL_ORDER = 3 # order of savgol filter


def initOOdev(devnr = 0):
    # Get list of connected OO devices:
    devices = []
    while devices == []:
        devices = sb.list_devices()
        time.sleep(0.5)
    print(devices)
    time.sleep(1)
    
    # Initialize device:
    spec = sb.Spectrometer(devices[devnr])
    time.sleep(1)
    
    # Add other info to spec struct:
    spec._min_int_time_sec = spec._dev.interface._INTEGRATION_TIME_MIN/1e6
    spec._max_int_time_sec = spec._dev.interface._INTEGRATION_TIME_MAX/1e6

    return spec, devices[devnr]

def _getOOcounts(spec, int_time_sec = _INT_TIME_SEC, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY):
    spec.integration_time_micros(int_time_sec*1e6) # expects micro secs.
    cnts = spec.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
    cnts = spec.intensities(correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity) # double call to avoid ending up with wrong buffer values due to crappy programming of ocean optics api
    return cnts


def create_dark_model(spec, dark_model_int_times = _DARK_MODEL_INT_TIMES, savgol_window = _SAVGOL_WINDOW, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY):
    
    root = tkinter.Tk() #hide tkinter main window for messagebox
    messagebox.showinfo("Dark Model Measurements","Close shutter and press Ok to continue with measurement.")
    
    # Determine odd window_length of savgol filter for smoothing (if 0: no smoothing):
    if savgol_window > 0:
        if isinstance(savgol_window,int):
            savgol_window = (savgol_window % 2==0) + savgol_window # ensure odd window length
        else:
            savgol_window = np.int(2*np.round(spec.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
    
    # Measure dark for several integration times
    dark_fig = plt.figure("Dark Model (savgol_window = {:1.1f})". format(savgol_window))    
    ax1 = dark_fig.add_subplot(1, 3, 1) 
    ax2 = dark_fig.add_subplot(1, 3, 2)  
    for i,it in enumerate(dark_model_int_times):
        print("Measuring dark counts for integration time {:1.0f}/{:1.0f} ({:1.4f}s)".format(i,len(dark_model_int_times),it))
        dark_int_time, dark_cnts = _find_opt_int_time(spec, it, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
        dark_cnts_s = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
        
          
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Counts')
        ax1.set_title('Dark Measurements (raw)')
        ax1.plot(spec.wavelengths(), dark_cnts,'.')
         
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Counts')
        ax2.set_title('Dark Measurements (smoothed)')
        ax2.plot(spec.wavelengths(), dark_cnts_s,'.')

        
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
        
    messagebox.showinfo("Dark Model Measurements","All dark measurements have finished. Press Ok to continue with measurement.")
    dark_its_arr = np.asarray(dark_its_arr)
    sum_dark_cnts = np.asarray(sum_dark_cnts)
    
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
    return dark_model

def _find_two_closest(value, values):
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
    dark_its_arr = dark_model[1:,0] # integration times
    dark_cnts_arr = dark_model[1:,1:] # dark counts (first axis of dark_model are wavelengths)
    p1,p2 = _find_two_closest(int_time, dark_its_arr)
    dark1 = dark_cnts_arr[p1]
    dark2 = dark_cnts_arr[p2]
    it1 = dark_its_arr[p1]
    it2 = dark_its_arr[p2]
    dark = dark1 + (int_time-it1)*(dark2-dark1)/(it2-it1)
    return np.vstack((dark_model[0,1:],dark)) # add wavelengths and return dark cnts

def _correct_for_dark(spec, cnts, int_time_sec, method = 'dark_model.dat', savgol_window = _SAVGOL_WINDOW, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY):
    if method == 'none':
        return cnts
    elif method == 'measure':
            # Determine odd window_length of savgol filter for smoothing (if 0: no smoothing):
            if savgol_window > 0:
                if isinstance(savgol_window,int):
                    savgol_window = (savgol_window % 2==0) + savgol_window # ensure odd window length
                else:
                    savgol_window = np.int(2*np.round(spec.wavelengths().shape[0]*savgol_window)+1) # if not int, 1/.. ratio
        
            root = tkinter.Tk() #hide tkinter main window
            messagebox.showinfo("Dark Measurement","Close shutter and press Ok to continue with measurement.")
            
            dark_cnts = _getOOcounts(spec, int_time_sec = int_time_sec, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
            if savgol_window > 0: # apply smoothing
                dark_cnts = savgol_filter(dark_cnts, savgol_window, _SAVGOL_ORDER)
            
            messagebox.showinfo("Dark Measurement","Dark measurement completed. Press Ok to continue with measurement.")
            root.withdraw()
    else:
        if isinstance(method,str):
            dark_model = pd.read_csv(method, sep =',', header = None).values
        dark_cnts = estimate_dark_from_model(int_time_sec, dark_model)[1] #take second row (first are wavelengths) 
    return cnts - dark_cnts

def _find_opt_int_time(spec, int_time_sec, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY):
    # Find optimum integration time and get measured counts
    #
    # int_time_sec == 0: unlimited search for integration time, but < max_int_time
    # int_time_sec >0: fixed integration time
    # int_time_sec <0: find optimum, but <= int_time_sec
    
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
    is_sat_bool = lambda cnts: (cnts.max() > max_value) # check for saturation
    if max_int_time is not None:
        target_cnts_bool = lambda cnts: (cnts.max() < (max_value*_TARGET_MAX_CNTS_RATIO)) # check for max_counts
        target_it_bool = lambda it: (it <= max_int_time) # check for max_int_time
                
        it = min_int_time # start at min. int_time
        cnts = getcnts(it) # get cnts
        its = [it] # store int_time in array
        max_cnts = [cnts.max()] # store max_cnts in array
        max_number_of_ratio_increases = 3
        while (target_cnts_bool(cnts) & target_it_bool(it)) & (not is_sat_bool(cnts)):
            if len(max_cnts < max_number_of_ratio_increases):
                it = it * _IT_RATIO_INCREASE
            else:
                p_max_cnts_vs_its = np.polyfit(max_cnts[-max_number_of_ratio_increases:],its[-max_number_of_ratio_increases:]) # try and predict a value close to target cnts
                it = np.polyval(p_max_cnts_vs_its, max_value*_TARGET_MAX_CNTS_RATIO)
                if not target_it_bool(it):
                    it = max_int_time
                
            its.append(it) # keep track of integration times
            cnts = getcnts(it)
            
            max_cnts.append(cnts.max())  # keep track of max counts
            
        while is_sat_bool(cnts): # if saturated, start reducing int_time again
            it = it / _IT_RATIO_INCREASE
            print('Saturated max count value. Reducing integration time to {:1.2f}s'.format(it))
            its.append(it) # keep track of integration times
            cnts = getcnts(it)
            
            max_cnts.append(cnts.max())  # keep track of max counts
            
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
            print('WARNING: Saturated max count value at integration time of {:1.2f}s'.format(int_time_sec))
            
    return int_time_sec, cnts 

def getOOspd(spec = None, devnr = 0, int_time_sec = _INT_TIME_SEC, \
             correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY, \
             tec_temperature_C = None,  \
             dark_cnts = 'dark_model.dat', savgol_window = _SAVGOL_WINDOW,\
             out = 'spd', spdtype = 'cnts/s'):
    
    # Initialize device:
    if spec is None:
        spec, device = initOOdev(devnr = devnr)
    
    # Enable tec and set temperature:
    if tec_temperature_C is not None:
        try:
            spec.tec_set_enable(True)
            spec.tec_set_temperature_C(set_point_C = tec_temperature_C)
            time.sleep(0.5)
            print("Device temperature = {:1.1f}°C".format(spec.tec_get_temperature_C()))
        except:
            pass
    
    # Find optimum integration time and get counts (0 unlimited, >0 fixed, <0: find upto)
    int_time_sec, cnts = _find_opt_int_time(spec, int_time_sec)
    
    # Get cnts anew when correct_dark_counts == True (is set to False in _find_opt_int_time):
    if correct_dark_counts == True:
        cnts = _getOOcounts(spec, int_time_sec, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
    
    # Correct for dark_counts if not supported by device:
    cnts = _correct_for_dark(spec, cnts, int_time_sec, method = dark_cnts, savgol_window = savgol_window, correct_dark_counts = correct_dark_counts, correct_nonlinearity = correct_nonlinearity)
    
    # Reset integration time to min. value for fast new measurement:
    spec.integration_time_micros(spec._min_int_time_sec*1e6)
    
    # Convert counts to counts/s:
    if spdtype == 'cnts/s':
        cnts = cnts/int_time_sec
        
    # Add wavelengths to spd:
    spd = np.vstack((spec.wavelengths(),cnts))
       
    # output requested:
    if out == 'spd':
        return spd
    else: 
        return eval(out)

def plot_spd(ax, spd, int_time, sum_cnts = 0, max_cnts = 0):
    ax.clear()
    ax.plot(spd[0],spd[1],'b')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('counts/s')
    ax.set_title("integration time = {:1.3f}s, sum_cnts = {:1.0f}, max_cnts = {:1.0f}".format(int_time, sum_cnts,max_cnts))
    plt.pause(0.1)



#------------------------------------------------------------------------------
# Code testing
if __name__ == '__main__':
    time.sleep(1) 
    spec = None
    if ('spec' in locals()) | (spec is None): 
        spec, device = initOOdev(devnr = 0)
    
        
    case = 'single'
    
    if case == 'single': # single measurement
        
        int_time = 3
        spd = getOOspd(spec, int_time_sec = int_time, spdtype = 'cnts/s',dark_cnts='dark_model.dat')
        spec.close() 
        
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plot_spd(ax,spd,int_time, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
        
    elif case == 'cont': # continuous measurement

        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        
        try:
            while True:
                spd = getOOspd(spec,int_time_sec = int_time)
                plot_spd(ax,spd,int_time, sum_cnts = spd[1].sum(),max_cnts = spd[1].max())
   
        except KeyboardInterrupt:
            spec.close() 
            pass
        
    elif case == 'list': # measure list of integration times
        int_times = np.array([3.2,0.8,1.6,3.2,1.6,0.2,0.2,0.2])/20 # illustrate need for two consecutive measurements to get correct spd (see _getOOcounts())
        int_times = np.array([0.1,0.2,0.3,0.4,0.5])/1 # quick example
        sum_cnts = np.empty(int_times.shape)
        max_cnts = np.empty(int_times.shape)
        for i,int_time in enumerate(int_times):
            
            spd = getOOspd(spec,int_time_sec = int_time, spdtype='cnts')
            sum_cnts[i] = spd[1].sum()
            max_cnts[i] = spd[1].mean()
            
            fig = plt.figure()
            ax  = fig.add_subplot(1, 1, 1)
            plot_spd(ax,spd,int_time, sum_cnts = sum_cnts[i],max_cnts = max_cnts[i])
            
        spec.close()
        fig2 = plt.figure()
        ax1  = fig2.add_subplot(1, 3, 1)
        ax1.plot(int_times,sum_cnts,'ro-')
        ax2  = fig2.add_subplot(1, 3, 2)
        ax2.plot(np.arange(int_times.size), sum_cnts,'bo-')
        ax3  = fig2.add_subplot(1, 3, 3)
        ax3.plot(np.arange(int_times.size), max_cnts,'bo-')

    elif case == 'dark': # create dark_model for dark light/current and readout noise correction
        dark_model = create_dark_model(spec, dark_model_int_times = _DARK_MODEL_INT_TIMES, savgol_window = _SAVGOL_WINDOW, correct_dark_counts = _CORRECT_DARK_COUNTS, correct_nonlinearity = _CORRECT_NONLINEARITY)
        spec.close()
        
        # write dark model to file
        pd.DataFrame(dark_model).to_csv('dark_model.dat', index=False, header=False, float_format='%1.4f')
        