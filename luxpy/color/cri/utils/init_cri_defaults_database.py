# -*- coding: utf-8 -*-
"""
Module with color fidelity and color gamut area parameter dicts
===============================================================

 :_CRI_TYPE_DEFAULT: Default cri_type.

 :CRI_DEFAULTS: default parameters for color fidelity and gamut area metrics 
               (major dict has 13 keys (04 Mar, 2025): 
               sampleset [str/dict], 
               ref_type [str], # specified in luxpy.spectrum.basics.illuminants.py
               calculation_wavelength_range [list],
               cieobs [Dict], 
               cct_mode [str],
               avg [fcn handle], 
               rf_from_avg_rounded_rfi [bool],
               scale [dict], 
               cspace [dict], 
               catf [dict], 
               rg_pars [dict], 
               cri_specific_pars [dict])
            
 :process_cri_type_input(): load a cri_type dict but overwrites any keys that 
                            have a non-None input in calling function


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import copy
import numpy as np

from luxpy import math, _CRI_RFL
from luxpy.utils import put_args_in_db
from .DE_scalers import linear_scale, log_scale, psy_scale

__all__ = ['_CRI_TYPE_DEFAULT', '_CRI_DEFAULTS', 'process_cri_type_input']

#------------------------------------------------------------------------------
# create default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
_CRI_TYPE_DEFAULT = 'ies-tm30'

_CRI_DEFAULTS = {'cri_types' : ['ciera','ciera-8','ciera-14','ciera-15','cierf-224-2017','cierf',
                                'iesrf','iesrf-tm30-15','iesrf-tm30-18','iesrf-tm30-20','iesrf-tm30-24','ies-tm30',
                                'cri2012','cri2012-hl17','cri2012-hl1000','cri2012-real210']}

_CRI_DEFAULTS['ciera-13.3-1995'] = {'sampleset' : "_CRI_RFL['cie-13.3-1995']['8']", 
                         'ref_type' : 'ciera', 
                         'calculation_wavelength_range' : None,
                         'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 
                         'cct_mode' : ('robertson2023',{}),
                         'avg' : np.mean, 
                         'rf_from_avg_rounded_rfi' : True,
                         'scale' :{'fcn' : linear_scale, 'cfactor' : [4.6]}, 
                         'cspace' : {'type':'wuv', 'xyzw' : None}, 
                         'catf': {'xyzw':None, 'mcat':'judd-1945','D':1.0,'La':None,'cattype':'vonkries','Dtype':None, 'catmode' : '1>2'}, 
                         'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                         'cri_specific_pars' : None
                         }

_CRI_DEFAULTS['ciera'] = copy.deepcopy(_CRI_DEFAULTS['ciera-13.3-1995'])
_CRI_DEFAULTS['ciera-8'] = copy.deepcopy(_CRI_DEFAULTS['ciera-13.3-1995'])
_CRI_DEFAULTS['ciera-14'] = copy.deepcopy(_CRI_DEFAULTS['ciera-13.3-1995']) 
_CRI_DEFAULTS['ciera-14']['sampleset'] = "_CRI_RFL['cie-13.3-1995']['14']"
_CRI_DEFAULTS['ciera-15'] = copy.deepcopy(_CRI_DEFAULTS['ciera-13.3-1995']) # R15 is J-Z-8726 sample (asian skin)
_CRI_DEFAULTS['ciera-15']['sampleset'] = "_CRI_RFL['cie-13.3-1995']['15']"



_CRI_DEFAULTS['cierf-224-2017'] = {'sampleset' : "_CRI_RFL['cie-224-2017']['99']['1nm']", # 1 nm is used in TM30, and CIE recommends 1 nm (these are interpolated from 5 nm using Sprague or some sorts)
                                 'ref_type' : 'cierf', 
                                 'calculation_wavelength_range' : [380,780],
                                 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 
                                 'cct_mode' : ('ohno2014', {'f_corr' : 0.9991, 'apply_linear_shift' : False, 'force_tolerance' : False}),
                                 'avg' : np.mean, 
                                 'rf_from_avg_rounded_rfi' : False,
                                 'scale' : {'fcn' : log_scale, 'cfactor' : [6.73]}, 
                                 'cspace' : {'type' : 'jab_cam02ucs' , 'xyzw': None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},
                                 'catf': None, 
                                 'rg_pars' : {'nhbins': 8, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                                 'cri_specific_pars' : None
                                 }

_CRI_DEFAULTS['cierf'] = copy.deepcopy(_CRI_DEFAULTS['cierf-224-2017'])


_CRI_DEFAULTS['iesrf-tm30-15'] = {'sampleset' : "_CRI_RFL['ies-tm30-15']['99']['1nm']", 
                                 'ref_type' : 'iesrf-tm30-15', 
                                 'calculation_wavelength_range' : [380,780],
                                 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 
                                 'cct_mode' : ('ohno2014', {'f_corr' : 1.0, 'apply_linear_shift' : True, 'force_tolerance' : False}),
                                 'avg' : np.mean, 
                                 'rf_from_avg_rounded_rfi' : False,
                                 'scale' :{'fcn' : log_scale, 'cfactor' : [7.54]}, 
                                 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},
                                 'catf': None, 
                                 'rg_pars' : {'nhbins': 16, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                                 'cri_specific_pars' : None
                                 }

_CRI_DEFAULTS['iesrf-tm30-18'] = {'sampleset' : "_CRI_RFL['ies-tm30-18']['99']['1nm']", 
                                 'ref_type' : 'iesrf-tm30-18',
                                 'calculation_wavelength_range' : [380,780], 
                                 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 
                                 'cct_mode' : ('ohno2014', {'f_corr' : 1.0, 'apply_linear_shift' : True, 'force_tolerance' : False}),
                                 'avg' : np.mean, 
                                 'rf_from_avg_rounded_rfi' : False,
                                 'scale' :{'fcn' : log_scale, 'cfactor' : [6.73]}, 
                                 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},
                                 'catf': None, 
                                 'rg_pars' : {'nhbins': 16, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                                 'cri_specific_pars' : None
                                 }

_CRI_DEFAULTS['iesrf-tm30-20'] = copy.deepcopy(_CRI_DEFAULTS['iesrf-tm30-18'])
_CRI_DEFAULTS['iesrf-tm30-20']['sampleset'] = "_CRI_RFL['ies-tm30-20']['99']['1nm']"
_CRI_DEFAULTS['iesrf-tm30-24'] = copy.deepcopy(_CRI_DEFAULTS['iesrf-tm30-18'])
_CRI_DEFAULTS['iesrf-tm30-24']['sampleset'] = "_CRI_RFL['ies-tm30-24']['99']['1nm']"

_CRI_DEFAULTS['iesrf'] = copy.deepcopy(_CRI_DEFAULTS['iesrf-tm30-24'])

_CRI_DEFAULTS['iesrf-tm30'] = copy.deepcopy(_CRI_DEFAULTS['iesrf-tm30-24'])

_CRI_DEFAULTS['ies-tm30'] = copy.deepcopy(_CRI_DEFAULTS['iesrf-tm30-24'])


_CRI_DEFAULTS['cri2012-hl17'] = {'sampleset' : "_CRI_RFL['cri2012']['HL17']", 
                             'ref_type' : 'ciera', 
                             'calculation_wavelength_range' : None,
                             'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 
                             'cct_mode' : ('ohno2014', {'f_corr' : 1.0, 'apply_linear_shift' : False, 'force_tolerance' : False}),
                             'avg' : math.rms, 
                             'rf_from_avg_rounded_rfi' : False,
                             'scale' : {'fcn': psy_scale, 'cfactor' : [1/55, 3/2, 2]}, 
                             'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},
                             'catf': None, 
                             'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                             'cri_specific_pars' : None
                             }

_CRI_DEFAULTS['cri2012-hl1000'] = {'sampleset' : "_CRI_RFL['cri2012']['HL1000']", 
                                   'ref_type' : 'ciera',
                                   'calculation_wavelength_range' : None,
                                   'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 
                                   'cct_mode' : ('ohno2014', {'f_corr' : 1.0, 'apply_linear_shift' : False, 'force_tolerance' : False}),
                                   'avg' : math.rms, 
                                   'rf_from_avg_rounded_rfi' : False,
                                   'scale': {'fcn' : psy_scale, 'cfactor' : [1/50, 3/2, 2]}, 
                                   'cspace' : {'type' : 'jab_cam02ucs','xyzw':None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},
                                   'catf': None, 
                                   'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False,'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                                   'cri_specific_pars' : None
                                   }

_CRI_DEFAULTS['cri2012-real210'] = {'sampleset' : "_CRI_RFL['cri2012']['Real210']",
                                    'ref_type' : 'ciera', 
                                    'calculation_wavelength_range' : None,
                                    'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'},
                                    'cct_mode' : ('ohno2014', {'f_corr' : 1.0, 'apply_linear_shift' : False, 'force_tolerance' : False}),
                                    'avg' : math.rms, 
                                    'rf_from_avg_rounded_rfi' : False,
                                    'scale' : {'fcn' : psy_scale, 'cfactor' : [2/45, 3/2, 2]},
                                    'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':None, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'}, 
                                    'catf': None, 
                                    'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : True}, 
                                    'cri_specific_pars' : None
                                    }

_CRI_DEFAULTS['cri2012'] = copy.deepcopy(_CRI_DEFAULTS['cri2012-hl17'])


#------------------------------------------------------------------------------
def process_cri_type_input(cri_type, args, callerfunction = ''):
    """
    Processes cri_type input in a function (helper function).
    
    | This function replaces the values of keys in the cri_type dict with the 
      corresponding not-None values in args.
    
    Args:
        :cri_type:
            | str or dict
            | Database with CRI model parameters.
        :args:
            | arguments from a caller function
        :callerfunction:
            | str with function the args originated from
        
    Returns:
        :cri_type: 
            | dict with database of CRI model parameters.
    """
    if isinstance(cri_type,str):
        if (cri_type in _CRI_DEFAULTS['cri_types']):
            cri_type = _CRI_DEFAULTS[cri_type].copy()
        else:
            raise Exception('.{}(): Unrecognized cri_type: {}'.format(callerfunction,cri_type))
    elif not isinstance(cri_type,dict):
        raise Exception('.{}(): cri_type is not a dict !'.format(callerfunction))
            
    cri_type = put_args_in_db(cri_type,args)
    return cri_type    

