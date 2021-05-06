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
cam: sub-package with color appearance models
=============================================

 : _AVAILABLE_MODELS: List with available color appearance models.

 :_UNIQUE_HUE_DATA: | database of unique hues with corresponding 
                    | Hue quadratures and eccentricity factors 
                    | for ciecam02, ciecam16, ciecam97s, cam15u, cam18sl)

 :_SURROUND_PARAMETERS: | database of surround param. c, Nc, F and FLL 
                        | for ciecam02, ciecam16, ciecam97s and cam15u.

 :_NAKA_RUSHTON_PARAMETERS: | database with parameters (n, sig, scaling and noise) 
                            | for the Naka-Rushton function: 
                            | NK(x) = sign(x) * scaling * ((abs(x)**n) / ((abs(x)**n) + (sig**n))) + noise

 :_CAM_UCS_PARAMETERS: | database with parameters specifying the conversion 
                       |  from ciecamX to:
                       |    camXucs (uniform color space), 
                       |    camXlcd (large color diff.), 
                       |    camXscd (small color diff).
                            
 :_CAM15U_PARAMETERS: database with CAM15u model parameters.
 
 :_CAM_SWW16_PARAMETERS: cam_sww16 model parameters.
 
 :_CAM18SL_PARAMETERS: database with CAM18sl model parameters

 :_CAM_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_CAM_DEFAULT_CONDITIONS: Default CAM model parameters for model.
 
 :_CAM_DEFAULT_TYPE: Default CAM (string) [for use in other modules].

 :_CAM_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.
                  
 :deltaH(): Compute a hue difference, dH = 2*C1*C2*sin(dh/2).

 :naka_rushton(): applies a Naka-Rushton function to the input
 
 :hue_angle(): calculates a positive hue angle

 :hue_quadrature(): calculates the Hue quadrature from the hue.

 
 :ciecam02(): | calculates ciecam02 output 
              | `N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, 
                “The CIECAM02 color appearance model,” 
                IS&T/SID Tenth Color Imaging Conference. p. 23, 2002. <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_

 :cam16(): | calculates cam16 output 
           | `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, 
             “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” 
             Color Res. Appl., p. n/a–n/a. <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_
           

 :cam02ucs(): | calculates ucs (or lcd, scd) output based on ciecam02 
              |  (forward + inverse available)
              |  `M. R. Luo, G. Cui, and C. Li, 
                 “Uniform colour spaces based on CIECAM02 colour appearance model,” 
                 Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
                 <http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract>`_

 :cam16ucs(): | calculates ucs (or lcd, scd) output based on cam16 
              |  (forward + inverse available)
              | `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, 
                “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” 
                Color Res. Appl., p. n/a–n/a. <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_

 :cam15u(): | calculates the output for the CAM15u model for self-luminous unrelated stimuli. 
            | `M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, 
              “Experimental driven modelling of the color appearance of 
              unrelated self-luminous stimuli: CAM15u,” 
              Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
              <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-12045&origin=search>`_
            | `M. Withouck, K. A. G. Smet, and P. Hanselaer, (2015), 
            “Brightness prediction of different sized unrelated self-luminous stimuli,” 
            Opt. Express, vol. 23, no. 10, pp. 13455–13466. 
            <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-10-13455&origin=search>`_
            
 :cam_sww16(): | A simple principled color appearance model based on a mapping 
                 of the Munsell color system.
               | `Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). 
                   A simple principled approach for modeling and understanding uniform color metrics. 
                   Journal of the Optical Society of America A, 33(3), A319–A331. 
                   <https://doi.org/10.1364/JOSAA.33.00A319>`_
               
 :cam18sl(): | calculates the output for the CAM18sl model for self-luminous related stimuli. 
             | `Hermans, S., Smet, K. A. G., & Hanselaer, P. (2018). 
               "Color appearance model for self-luminous stimuli."
               Journal of the Optical Society of America A, 35(12), 2000–2009. 
               <https://doi.org/10.1364/JOSAA.35.002000>`_       
               
 :camXucs(): Wraps ciecam02(), ciecam16(), cam02ucs(), cam16ucs().

 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz',
      | 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
      | 'xyz_to_jabM_ciecam16', 'jabM_ciecam16_to_xyz',
      | 'xyz_to_jabC_ciecam16', 'jabC_ciecam16_to_xyz',
      | 'xyz_to_jabz',          'jabz_to_xyz',
      | 'xyz_to_jabM_zcam',     'jabM_zcam_to_xyz', 
      | 'xyz_to_jabC_zcam',     'jabC_zcam_to_xyz']
      | 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
      | 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
      | 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
      | 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz',
      | 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
      | 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz',
      | 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz',
      | 'xyz_to_lab_cam_sww16', 'lab_cam_sww16_to_xyz',
      | 'xyz_to_qabM_cam18sl', 'qabM_cam18sl_to_xyz',
      | 'xyz_to_qabs_cam18sl', 'qabs_cam18sl_to_xyz',
      

 :_update_parameter_dict(): Get parameter dict and update with values in args dict

 :_setup_default_adaptation_field(): Setup a default illuminant adaptation field with Lw = 100 cd/m² for selected CIE observer.

 :_massage_input_and_init_output(): Redimension input data to ensure most they have the appropriate sizes for easy and efficient looping.

 :_massage_output_data_to_original_shape(): Massage output data to restore original shape of original CAM input.
 
 :_get_absolute_xyz_xyzw(): Calculate absolute xyz tristimulus values of stimulus and white point from spectral input or convert relative xyz values to absolute ones.
 
 :_simple_cam(): An example CAM illustration the usage of the functions in luxpy.cam.helpers 


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
#------------------------------------------------------------------------------
# List available CAMs:
_AVAILABLE_MODELS = ['ciecam02','cam02ucs','ciecam16','cam16ucs',
                     'zcam','cam15u','cam18sl','cam_sww16']

__all__ = ['_AVAILABLE_MODELS']

#------------------------------------------------------------------------------
# Utility imports:
from .utils import hue_angle, naka_rushton, deltaH, hue_quadrature

__all__ += ['hue_angle', 'naka_rushton', 'deltaH', 'hue_quadrature']

#------------------------------------------------------------------------------
# Helper functions imports:
from .helpers import (_update_parameter_dict,_setup_default_adaptation_field,
                      _massage_input_and_init_output,_massage_output_data_to_original_shape,
                      _get_absolute_xyz_xyzw,_simple_cam)

__all__ += ['_update_parameter_dict','_setup_default_adaptation_field',
           '_massage_input_and_init_output','_massage_output_data_to_original_shape',
           '_get_absolute_xyz_xyzw','_simple_cam']

#------------------------------------------------------------------------------
# ciecam02 imports:
# import ciecam02 as _ciecam02
from .ciecam02 import run as ciecam02
from .ciecam02 import _AXES as _CIECAM02_AXES
from .ciecam02 import _UNIQUE_HUE_DATA as _CIECAM02_UNIQUE_HUE_DATA
from .ciecam02 import _SURROUND_PARAMETERS as _CIECAM02_SURROUND_PARAMETERS
from .ciecam02 import _NAKA_RUSHTON_PARAMETERS as _CIECAM02_NAKA_RUSHTON_PARAMETERS
from .ciecam02 import _DEFAULT_WHITE_POINT as _CIECAM02_DEFAULT_WHITE_POINT
from .ciecam02 import _DEFAULT_CONDITIONS as _CIECAM02_DEFAULT_CONDITIONS
from .ciecam02 import (xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, 
                       xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz)

__all__ += ['ciecam02'] 

# __all__ += ['_CIECAM02_AXES',
#             '_CIECAM02_UNIQUE_HUE_DATA',
#             '_CIECAM02_DEFAULT_WHITE_POINT',
#             '_CIECAM02_SURROUND_PARAMETERS', 
#             '_CIECAM02_NAKA_RUSHTON_PARAMETERS']

__all__ += ['xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz', 
            'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz']

#------------------------------------------------------------------------------
# cam02ucs imports:
# import cam02ucs as _cam02ucs
from .cam02ucs import run as cam02ucs
from .cam02ucs import _AXES as _CAM02UCS_AXES
from .cam02ucs import _CAM_UCS_PARAMETERS as _CAM02UCS_UCS_PARAMETERS
from .cam02ucs import _DEFAULT_WHITE_POINT as _CAM02UCS_DEFAULT_WHITE_POINT
from .cam02ucs import _DEFAULT_CONDITIONS as _CAM02UCS_DEFAULT_CONDITIONS
from .cam02ucs import (xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, 
                       xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz,
                       xyz_to_jab_cam02scd, jab_cam02scd_to_xyz)

__all__  += ['cam02ucs'] 

# __all__ += ['_CAM02UCS_AXES',
#             '_CAM02UCS_UCS_PARAMETERS',
#             '_CAM02UCS_DEFAULT_WHITE_POINT']

__all__ += ['xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
            'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
            'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz']


#------------------------------------------------------------------------------
# ciecam16 imports:
# import ciecam16 as _ciecam16
from .ciecam16 import run as ciecam16
from .ciecam16 import _AXES as _CIECAM16_AXES
from .ciecam16 import _UNIQUE_HUE_DATA as _CIECAM16_UNIQUE_HUE_DATA
from .ciecam16 import _SURROUND_PARAMETERS as _CIECAM16_SURROUND_PARAMETERS
from .ciecam16 import _NAKA_RUSHTON_PARAMETERS as _CIECAM16_NAKA_RUSHTON_PARAMETERS
from .ciecam16 import _DEFAULT_WHITE_POINT as _CIECAM16_DEFAULT_WHITE_POINT
from .ciecam16 import _DEFAULT_CONDITIONS as _CIECAM16_DEFAULT_CONDITIONS
from .ciecam16 import (xyz_to_jabM_ciecam16, jabM_ciecam16_to_xyz, 
                       xyz_to_jabC_ciecam16, jabC_ciecam16_to_xyz)

__all__ += ['ciecam16'] 

# __all__ += ['_CIECAM16_AXES',
#             '_CIECAM16_UNIQUE_HUE_DATA',
#             '_CIECAM16_DEFAULT_WHITE_POINT',
#             '_CIECAM16_SURROUND_PARAMETERS', 
#             '_CIECAM16_NAKA_RUSHTON_PARAMETERS']

__all__ += ['xyz_to_jabM_ciecam16', 'jabM_ciecam16_to_xyz', 
            'xyz_to_jabC_ciecam16', 'jabC_ciecam16_to_xyz']

#------------------------------------------------------------------------------
# cam16ucs imports:
# import cam16ucs as _cam16ucs
from .cam16ucs import run as cam16ucs
from .cam16ucs import _AXES as _CAM16UCS_AXES
from .cam16ucs import _CAM_UCS_PARAMETERS as _CAM16UCS_UCS_PARAMETERS
from .cam16ucs import _DEFAULT_WHITE_POINT as _CAM16UCS_DEFAULT_WHITE_POINT
from .cam16ucs import _DEFAULT_CONDITIONS as _CAM16UCS_DEFAULT_CONDITIONS
from .cam16ucs import (xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, 
                       xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz,
                       xyz_to_jab_cam16scd, jab_cam16scd_to_xyz)

__all__  += ['cam16ucs'] 

# __all__ += ['_CAM16UCS_AXES',
#             '_CAM16UCS_UCS_PARAMETERS',
#             '_CAM16UCS_DEFAULT_WHITE_POINT']

__all__ += ['xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 
            'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
            'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz']  



#------------------------------------------------------------------------------
# zcam imports:
# import zcam as _zcam
from .zcam import run as zcam
from .zcam import _AXES as _ZCAM_AXES
from .zcam import _UNIQUE_HUE_DATA as _ZCAM_UNIQUE_HUE_DATA
from .zcam import _SURROUND_PARAMETERS as _ZCAM_SURROUND_PARAMETERS
from .zcam import _DEFAULT_WHITE_POINT as _ZCAM_DEFAULT_WHITE_POINT
from .zcam import _DEFAULT_CONDITIONS as _ZCAM_DEFAULT_CONDITIONS
from .zcam import (xyz_to_jabz, jabz_to_xyz,
                   xyz_to_jabM_zcam, jabM_zcam_to_xyz, 
                   xyz_to_jabC_zcam, jabC_zcam_to_xyz)

__all__ += ['zcam'] 


__all__ += ['xyz_to_jabz', 'jabz_to_xyz',
            'xyz_to_jabM_zcam', 'jabM_zcam_to_xyz', 
            'xyz_to_jabC_zcam', 'jabC_zcam_to_xyz']



#------------------------------------------------------------------------------
# cam15 imports:
from .cam15u import  (cam15u, _CAM15U_AXES, _CAM15U_UNIQUE_HUE_DATA, _CAM15U_PARAMETERS,
                      _CAM15U_NAKA_RUSHTON_PARAMETERS, _CAM15U_SURROUND_PARAMETERS,
                      xyz_to_qabW_cam15u, qabW_cam15u_to_xyz)

__all__ += ['cam15u', 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz']



#------------------------------------------------------------------------------
# sww2016 cam imports:
from .sww2016 import (cam_sww16, _CAM_SWW16_AXES, _CAM_SWW16_PARAMETERS,
                      xyz_to_lab_cam_sww16, lab_cam_sww16_to_xyz)

__all__ += ['cam_sww16', 'xyz_to_lab_cam_sww16', 'lab_cam_sww16_to_xyz']



#------------------------------------------------------------------------------
# cam18sl imports:
from .cam18sl import  (cam18sl, _CAM18SL_AXES, _CAM18SL_UNIQUE_HUE_DATA, 
                       _CAM18SL_PARAMETERS,_CAM18SL_NAKA_RUSHTON_PARAMETERS, 
                      xyz_to_qabM_cam18sl, qabM_cam18sl_to_xyz, 
                      xyz_to_qabS_cam18sl, qabS_cam18sl_to_xyz)

__all__ += ['cam18sl','xyz_to_qabM_cam18sl', 'qabM_cam18sl_to_xyz',
            'xyz_to_qabS_cam18sl', 'qabS_cam18sl_to_xyz']


__all__ += ['_CAM_AXES', '_UNIQUE_HUE_DATA','_SURROUND_PARAMETERS','_NAKA_RUSHTON_PARAMETERS']


__all__ += ['_CAM15U_PARAMETERS','_CAM_SWW16_PARAMETERS','_CAM18SL_PARAMETERS']



#------------------------------------------------------------------------------
# Create some dictionaries that group databases of different models: 
# --- cam axes list of strings for ploting ---
_CAM_AXES = {}
_CAM_AXES.update(_CIECAM02_AXES)
_CAM_AXES.update(_CAM02UCS_AXES)
_CAM_AXES.update(_CIECAM16_AXES)
_CAM_AXES.update(_CAM16UCS_AXES)
_CAM_AXES.update(_ZCAM_AXES)
_CAM_AXES['qabW_cam15u'] = _CAM15U_AXES 
_CAM_AXES['lab_cam_sww16'] = _CAM_SWW16_AXES
_CAM_AXES['qabS_cam18sl'] = _CAM18SL_AXES 

__all__ += ['_CAM_AXES']

# --- unique hue data ---
_UNIQUE_HUE_DATA = {'models' : ['ciecam02', 'cam02ucs', 
                                'ciecam16', 'cam16ucs',
                                'zcam'],
                    'ciecam02' : _CIECAM02_UNIQUE_HUE_DATA,
                    'cam02ucs' : _CIECAM02_UNIQUE_HUE_DATA,
                    'ciecam16' : _CIECAM16_UNIQUE_HUE_DATA,
                    'cam16ucs' : _CIECAM16_UNIQUE_HUE_DATA,
                    'zcam' : _ZCAM_UNIQUE_HUE_DATA,
                    }
_UNIQUE_HUE_DATA['cam15u'] = _CAM15U_UNIQUE_HUE_DATA
_UNIQUE_HUE_DATA['models'].append('cam15u')
_UNIQUE_HUE_DATA['cam18sl'] = _CAM18SL_UNIQUE_HUE_DATA
_UNIQUE_HUE_DATA['models'].append('cam18sl')

__all__ += ['_UNIQUE_HUE_DATA']

# --- surround parameter data ---
_SURROUND_PARAMETERS = {'ciecam02' : _CIECAM02_SURROUND_PARAMETERS,
                        'cam02ucs' : _CIECAM02_SURROUND_PARAMETERS,
                        'ciecam16' : _CIECAM16_SURROUND_PARAMETERS,
                        'cam16ucs' : _CIECAM16_SURROUND_PARAMETERS,
                        'zcam'  : _ZCAM_SURROUND_PARAMETERS
                        }
_SURROUND_PARAMETERS['cam15u'] = _CAM15U_SURROUND_PARAMETERS
_SURROUND_PARAMETERS['cam_sww16'] = {} 
_SURROUND_PARAMETERS['cam18sl'] = {}

__all__ += ['_SURROUND_PARAMETERS']

# --- Naka-Rushton function parameter data ---
_NAKA_RUSHTON_PARAMETERS = {'ciecam02' : _CIECAM02_NAKA_RUSHTON_PARAMETERS,
                            'cam02ucs' : _CIECAM02_NAKA_RUSHTON_PARAMETERS,
                            'ciecam16' : _CIECAM16_NAKA_RUSHTON_PARAMETERS,
                            'cam16ucs' : _CIECAM16_NAKA_RUSHTON_PARAMETERS,
                            }
_NAKA_RUSHTON_PARAMETERS['cam15u'] =  _CAM15U_NAKA_RUSHTON_PARAMETERS
_NAKA_RUSHTON_PARAMETERS['cam18sl'] =  _CAM18SL_NAKA_RUSHTON_PARAMETERS

__all__ += ['_NAKA_RUSHTON_PARAMETERS']


# ---- UCS parameters ---
_CAM_UCS_PARAMETERS = {'ciecam02' : _CAM02UCS_UCS_PARAMETERS,
                       'cam02ucs' : _CAM02UCS_UCS_PARAMETERS,
                       'ciecam16' : _CAM16UCS_UCS_PARAMETERS,
                       'cam16ucs' : _CAM16UCS_UCS_PARAMETERS}

__all__ += ['_CAM_UCS_PARAMETERS']

#------------------------------------------------------------------------------
#Set some defaults:
_CAM_DEFAULT_TYPE = 'ciecam02'
_CAM_DEFAULT_WHITE_POINT = _CIECAM02_DEFAULT_WHITE_POINT
_CAM_DEFAULT_CONDITIONS = _CIECAM02_DEFAULT_CONDITIONS
__all__ += ['_CAM_DEFAULT_TYPE', '_CAM_DEFAULT_WHITE_POINT','_CAM_DEFAULT_CONDITIONS']


#------------------------------------------------------------------------------
# Define some extra functions:

def camXucs(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = None, outin = 'J,aM,bM', 
            conditions = None, forward = True, ucstype = 'ucs',
            yellowbluepurplecorrect = False, mcat = None, 
            camtype = _CAM_DEFAULT_TYPE):
    """
    Wraps ciecam02(), ciecam16(), cam02ucs(), cam16ucs().
    
    Args:
        :camtype:
            | _DEFAULT_TYPE, optional
            | String specifying the cam-model.
            
    Notes:
        1. To call ciecam02() or ciecam16(): set ucstype to None !!!
        2. For more info on other input arguments, see doc-strings of those functions.
    """
    if (camtype == 'ciecam02') & (ucstype is None):
        return ciecam02(data, xyzw = xyzw, Yw = Yw, outin = outin,  
                        conditions = conditions, forward = forward, mcat = mcat,
                        yellowbluepurplecorrect = yellowbluepurplecorrect)
    
    elif ((camtype == 'cam02ucs') | (camtype == 'ciecam02')) & (ucstype is not None):
        return cam02ucs(data, xyzw = xyzw, Yw = Yw, ucstype = ucstype, 
                        conditions = conditions, forward = forward, mcat = mcat,
                        yellowbluepurplecorrect = yellowbluepurplecorrect)
    
    elif (camtype == 'ciecam16') & (ucstype is None):
        return ciecam16(data, xyzw = xyzw, Yw = Yw, outin = outin,  
                        conditions = conditions, forward = forward, mcat = mcat)
    
    elif ((camtype == 'cam16ucs') | (camtype == 'ciecam16')) & (ucstype is not None):
        return cam16ucs(data, xyzw = xyzw, Yw = Yw, ucstype = ucstype, 
                        conditions = conditions, forward = forward, mcat = mcat)
    elif ((camtype == 'zcam')):
        return zcam(data, xyzw = xyzw, Yw = Yw, outin = outin,  
                        conditions = conditions, forward = forward, mcat = mcat)

__all__ +=['camXucs'] 