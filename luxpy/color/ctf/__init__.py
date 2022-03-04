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
Module with functions related to basic colorimetry
==================================================

Note
----

  Note that colorimetric data is always located in the last axis
  of the data arrays. (See also xyz specification in __doc__ string
  of luxpy.spd_to_xyz())

colortransforms.py
------------------

 :_CSPACE_AXES: dict with list[str,str,str] containing axis labels
                of defined cspaces

Supported chromaticity / colorspace functions:
  | * xyz_to_Yxy(), Yxy_to_xyz(): (X,Y,Z) <-> (Y,x,y);
  | * xyz_to_Yuv(), Yuv_to_Yxy(): (X,Y,Z) <-> CIE 1976 (Y,u',v');
  | * xyz_to_Yuv76(), Yuv76_to_Yxy(): (X,Y,Z) <-> CIE 1976 (Y,u',v');
  | * xyz_to_Yuv60(), Yuv60_to_Yxy(): (X,Y,Z) <-> CIE 1960 (Y,u,v);
  | * xyz_to_xyz(), lms_to_xyz(): (X,Y,Z) <-> (X,Y,Z); for use with colortf()
  | * xyz_to_lms(), lms_to_xyz(): (X,Y,Z) <-> (L,M,S) cone fundamental responses
  | * xyz_to_lab(), lab_to_xyz(): (X,Y,Z) <-> CIE 1976 (L*a*b*)
  | * xyz_to_luv(), luv_to_xyz(): (X,Y,Z) <-> CIE 1976 (L*u*v*)
  | * xyz_to_Vrb_mb(),Vrb_mb_to_xyz(): (X,Y,Z) <-> (V,r,b); [Macleod & Boyton, 1979]
  | * xyz_to_ipt(), ipt_to_xyz(): (X,Y,Z) <-> (I,P,T); (Ebner et al, 1998)
  | * xyz_to_Ydlep(), Ydlep_to_xyz(): (X,Y,Z) <-> (Y,dl, ep); 
  |                   Y, dominant wavelength (dl) and excitation purity (ep)
  | * xyz_to_srgb(), srgb_to_xyz(): (X,Y,Z) <-> sRGB; (IEC:61966 sRGB)

colortf.py
----------
    
 :_COLORTF_DEFAULT_WHITE_POINT: ndarray with XYZ values of default white point 
                                 (equi-energy white) for color transformation 
                                 if none is supplied.

 :colortf(): Calculates conversion between any two color spaces (cspace)
             for which functions xyz_to_cspace() and cspace_to_xyz() are defined.



References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_

    2. `Ebner F, and Fairchild MD (1998).
       Development and testing of a color space (IPT) with improved hue uniformity.
       In IS&T 6th Color Imaging Conference, (Scottsdale, Arizona, USA), pp. 8–13.
       <http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true>`_

    3. `MacLeod DI, and Boynton RM (1979).
       Chromaticity diagram showing cone excitation by stimuli of equal luminance.
       J. Opt. Soc. Am. 69, 1183–1186.
       <https://www.osapublishing.org/josa/abstract.cfm?uri=josa-69-8-1183>`_


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)

"""