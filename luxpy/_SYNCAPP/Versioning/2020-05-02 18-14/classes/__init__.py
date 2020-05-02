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
Module supporting class basic functionality for spectral and colorimetric data
==============================================================================

SPD fields 
----------

 :self.wl: wavelengths of spectral data
    
 :self.value: values of spectral data
    
 :self.dtype: spectral data type ('S' light source, or 'R', reflectance, ...),
              used to determine interpolation method following CIE15-2018.
    
 :self.shape: self.value.shape

 :self.N = self.shape[0] (number of spectra in instance)
    

SPD methods 
-----------

 :self.read_csv_(): Reads spectral data from file.

 :self.plot(): Make a plot of the spectral data in SPD instance.

 :self.mean(): Take mean of all spectra in SPD instance

 :self.sum(): Sum all spectra in SPD instance.

 :self.dot(): Take dot product with instance of SPD.

 :self.add(): Add instance of SPD.

 :self.sub(): Subtract instance of SPD.

 :self.mul(): Multiply instance of SPD.

 :self.div(): Divide by instance of SPD.

 :self.pow(): Raise SPD instance to power n.

 :self.get_(): Get spd as ndarray in instance of SPD.

 :self.setwlv(): Store spd ndarray in fields wl and values of instance of SPD.

 :self.getwld_(): Get wavelength spacing of SPD instance. 

 :self.normalize(): Normalize spectral power distributions in SPD instance.

 :self.cie_interp(): Interpolate / extrapolate spectral data following CIE15-2018.

 :self.to_xyz(): Calculates xyz tristimulus values from spectral data 
                 and return as instance of class XYZ.
                 
CDATA fields 
------------

 :self.relative: relative (True) or absolute (False) colorimetric data.  
    
 :self.value: values of spectral data
    
 :self.dtype: colorimetric data type ('xyz', 'Yuv', 'lab', ...)
    
 :self.shape: self.value.shape

 :self.cieobs: CMF set used to determine colorimetric data from spectral data.
   
 
CDATA methods
-------------

:self.get_values_(): Get values from data and return ndarray. 

:self.split_(): Split .value along last axis and return list of ndarrays.

 :self.join(): Join data along last axis and return instance.
 
 :self.take_(): Applies numpy.take on .value field.
 
 :self.getax_(): Get elements in .value field along specific axis
 
 :self.dot(): Take dot product with instance.
 
 :self.add(): Add data to instance value field.
 
 :self.sub(): Subtract data from instance value field.
 
 :self.mul(): Multiply data with instance value field.
 
 :self.div(): Divide instance value field by data.
 
 :self.pow(): Raise instance value field to power.
 
 :self.broadcast(): Broadcast instance value field to shape of data.
 
 :self.get_S(): | Get spectral data related to light sources. 
                | (cfr. axis = 1 in xyz ndarrays).
          
 :self.get_R():  | Get spectral data related to reflectance samples.
                 | (cfr. axis = 0 in xyz ndarrays).
            
 :self.get_subset(): | Get spectral data related to specific light source and reflectance data
                     | (cfr. axis = 1 and axis = 0 in xyz ndarrays).



XYZ fields 
----------

Same as CDATA, XYZ inherits from CDATA 



XYZ methods
-----------

 :self.ctf(): Convert XYZ tristimulus values to color space coordinates.
 
 :self.plot(): Plot tristimulus or cone fundamental values.

 :self.to_cspace(): Convert XYZ tristimulus values to ...
                    (Method wrappers for all xyz_to_cspace type functions)
  

          
LAB fields 
----------

| Same as CDATA, LAB inherits from CDATA 
| AND, additionally the following dict field with keys related to color space parameters:
|     
|     self.cspace_par = {}
|     self.cspace_par['cieobs'] = self.cieobs
|   
|    
| # specific to some chromaticity / color space transforms:   
|  
|     self.cspace_par['xyzw'] = xyzw
|     self.cspace_par['M'] = M
|     self.cspace_par['scaling'] = scaling
|     
| # specific to some CAM transforms:
| 
|     self.cspace_par['Lw'] = Lw
|     self.cspace_par['Yw'] = Yw
|     self.cspace_par['Yb'] = Yb
|     self.cspace_par['conditions'] = conditions
|     self.cspace_par['yellowbluepurplecorrect'] = yellowbluepurplecorrect
|     self.cspace_par['mcat'] = mcat
|     self.cspace_par['ucstype'] = ucstype
|     self.cspace_par['fov'] = fov
|     self.cspace_par['parameters'] = parameters


LAB methods
-----------

 :self.ctf(): Convert color space coordinates to XYZ tristimulus values.
 
 :self.to_xyz(): Convert color space coordinates to XYZ tristimulus values. 
 
 :self.plot(): Plot color coordinates.
 
"""

#from .SPD import *
#__all__ = SPD.__all__
#
#from CDATA import *
#__all__ += CDATA.__all___
