# luxpy.spectro.jeti: module for working with JETI specbos 1211 spectroradiometer (windows)

## Installation:
 1. Install jeti drivers (if you're lucky, windows might detect the device and automatically install the correct drivers).
 2. Ready to go.
 
## Functions:

 * dvc_detect(): detect number of connected JETI devices.
 * dvc_open(): open device.
 * close_open(): close device.
 * dvc_reset(): reset device (same as disconnecting and reconnecting USB).
 * start_meas(): start measurement on already opened device.
 * check_meas_status(): check status of initiated measurement.
 * wait_until_meas_is_finished(): wait until a initiated measurement is finished.
 * read_spectral_radiance(): read measured spectral radiance (W/m².sr.nm) from device.
 * set_default(): set all measurement parameters to the default values.
 * get_wavelength_params(): get wavelength calibration parameters of polynomial of order 5.
 * measure_flicker_freq(): measure flicker frequency (Hz)
 * get_laser_status(): get pilot laser status of device.
 * set_laser_status(): set pilot laser status of device.
 * set_laser(): turn laser ON (3 modulation types: 7Hz (1), 28 Hz (2) and 255 Hz (3)) or OFF (0) and set laser intensity.
 * get_calibration_range(): get calibration range.
 * get_shutter_status(): get shutter status of device. 
 * set_shutter_status(): set shutter status of device. 
 * get_integration_time(): get default integration time stored in device.
 * get_min_integration_time(): get the minimum integration time (seconds) which can be used with the connected device.
 * get_max_auto_integration_time(): get the maximum integration time which will be used for adaption (automated Tint selection).
 * set_max_auto_integration_time(): set the maximum integration time which will be used for adaption (automated Tint selection).
 * get_spd(): measure spectral radiance (W/nm.sr.m²).

 
## Default parameters:

 * _TWAIT_STATUS: default time to wait before checking measurement status in wait_until_meas_is_finished().
 * _TINT_MAX: maximum integration time for device. 
 * _TINT_MIN: minimum integration time #set to None -> If None: find it on device (in 'start_meas()' fcn.)
 * _ERROR: error value.
 * _PKG_PATH = path to (sub)-package.
 
