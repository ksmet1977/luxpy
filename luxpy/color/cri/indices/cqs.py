"""
Module for color quality scale, CQS
===================================

 :_CQS_DEFAULTS: default settings for CQS 
                 (major dict has 13 keys (04-Mar-2024): 
                 sampleset [str/dict], 
                 ref_type [str], 
                 calculation_wavelength_range [list],
                 cieobs [Dict], 
                 cct_mode [str],
                 avg [fcn handle], 
                 rf_from_avg_rounded_rfi [bool],
                 round_daylightphase_Mi_to_cie_recommended [bool],
                 scale [dict], 
                 cspace [dict], 
                 catf [dict], 
                 rg_pars [dict], 
                 cri_specific_pars [dict])
            
 :spd_to_cqs(): | Color Quality Scale
                | versions 7.5 and 9.0 are supported. 

Reference
    1. `W. Davis and Y. Ohno, 
    “Color quality scale,” (2010), 
    Opt. Eng., vol. 49, no. 3, pp. 33602–33616.
    <http://spie.org/Publications/Journal/10.1117/1.3360335>`_


"""
import numpy as np

from luxpy import math
from ..utils.DE_scalers import log_scale
from ..utils.helpers import spd_to_DEi

__all__ = ['spd_to_cqs', '_CQS_DEFAULTS']

_CQS_DEFAULTS = {}
_CQS_DEFAULTS['cqs-v7.5'] = {'sampleset' : "_CRI_RFL['cqs']['v7.5']",
                             'ref_type' : 'ciera', 
                             'calculation_wavelength_range' : None,
                             'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 
                             'cct_mode' : ('ohno2014', {'force_tolerance' : False}),
                             'avg' : math.rms, 
                             'rf_from_avg_rounded_rfi' : False,
                             'round_daylightphase_Mi_to_cie_recommended' : False,
                             'scale' : {'fcn' : log_scale, 'cfactor' : [2.93, 3.10, 3.78]}, 
                             'cspace' : {'type': 'lab', 'xyzw' : None}, 
                             'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 
                             'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 
                             'cri_specific_pars' : {'maxC': None}
                             }
_CQS_DEFAULTS['cqs-v9.0'] = {'sampleset' : "_CRI_RFL['cqs']['v9.0']", 
                             'ref_type' : 'ciera',
                             'calculation_wavelength_range' : None,
                             'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 
                             'cct_mode' : ('ohno2014', {'force_tolerance' : False}),
                             'avg' : math.rms, 
                             'rf_from_avg_rounded_rfi' : False,
                             'round_daylightphase_Mi_to_cie_recommended' : False,
                             'scale' : {'fcn' : log_scale, 'cfactor' : [3.03, 3.20, 3.88]}, 
                             'cspace' : {'type': 'lab', 'xyzw' : None}, 
                             'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 
                             'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 
                             'cri_specific_pars' : {'maxC': 10.0}
                             }

#-----------------------------------------------------------------------------
def  spd_to_cqs(SPD, version = 'v9.0', out = 'Qa',wl = None, interp_settings = None):
    """
    Calculates CQS Qa (Qai) or Qf (Qfi) or Qp (Qpi) for versions v9.0 or v7.5.
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :version: 
            | 'v9.0' or 'v7.5', optional
        :out: 
            | 'Qa' or str, optional
            | Specifies requested output (e.g. 'Qa,Qai,Qf,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation   
    
    Returns:
        :returns:
            | float or ndarray with CQS Qa for :out: 'Qa'
            | Other output is also possible by changing the :out: str value. 
    
    References:
        1. `W. Davis and Y. Ohno, 
        “Color quality scale,” (2010), 
        Opt. Eng., vol. 49, no. 3, pp. 33602–33616.
        <http://spie.org/Publications/Journal/10.1117/1.3360335>`_
    
    """  
    outlist = out.split()    
    if isinstance(version,str):
        cri_type = 'cqs-' + version
    elif isinstance(version, dict):
        cri_type = version
     
    # calculate DEI, labti, labri and get cspace_pars and rg_pars:
    DEi, labti, labri, cct, duv, cri_type = spd_to_DEi(SPD, cri_type = cri_type, out = 'DEi,jabt,jabr,cct,duv,cri_type', wl = wl, 
                                                        interp_settings = interp_settings)
    
    # further unpack cri_type:
    scale_fcn = cri_type['scale']['fcn']     
    scale_factor = cri_type['scale']['cfactor']    
    avg = cri_type['avg']
    cri_specific_pars = cri_type['cri_specific_pars'] 
    rg_pars = cri_type['rg_pars'] 
    
    # get maxC: to limit chroma-enhancement:
    maxC = cri_specific_pars['maxC']
    
    # make 3d:
    test_original_shape = labti.shape
    if len(test_original_shape)<3:
        labti = labti[:,None] 
        labri = labri[:,None] 
        DEi = DEi[:,None] 
        cct = cct[:,None] 

    # calculate Rg for each spd:
    Qf = np.zeros((1,labti.shape[1]))
    Qfi = np.zeros((labti.shape[0],labti.shape[1]))
    
    if version == 'v7.5':
        GA = (9.2672*(1.0e-11))*cct**3.0  - (8.3959*(1.0e-7))*cct**2.0 + 0.00255*cct - 1.612 
        GA = np.where(cct < 3500, GA, 1)
    elif version == 'v9.0':
        GA = np.ones(cct.shape)
    else:
        raise Exception ('.cri.spd_to_cqs(): Unrecognized CQS version.')
      
    if ('Qf' in outlist) | ('Qfi' in outlist):

        # loop of light source spds
        for ii in range(labti.shape[1]):
            Qfi[:,ii] = GA[ii]*scale_fcn(DEi[:,ii],[scale_factor[0]])
            Qf[:,ii] = GA[ii]*scale_fcn(avg(DEi[:,ii,None],axis = 0),[scale_factor[0]])

    if ('Qa' in outlist) | ('Qai' in outlist) | ('Qp' in outlist) | ('Qpi' in outlist):
        
        Qa = Qf.copy()
        Qai = Qfi.copy()
        Qp = Qf.copy()
        Qpi = Qfi.copy()
        
         # loop of light source spds
        for ii in range(labti.shape[1]):
            
            # calculate deltaC:
            deltaC = np.sqrt(np.power(labti[:,ii,1:3],2).sum(axis = 1,keepdims=True)) - np.sqrt(np.power(labri[:,ii,1:3],2).sum(axis = 1,keepdims=True)) 
            # limit chroma increase:
            DEi_Climited = DEi[:,ii,None].copy()
            deltaC_Climited = deltaC.copy()
            if maxC is None:
                maxC = 10000.0
            limitC = np.where(deltaC >= maxC)[0]
            deltaC_Climited[limitC] = maxC
            p_deltaC_pos = np.where(deltaC>0.0)[0]
            DEi_Climited[p_deltaC_pos] = np.sqrt(DEi_Climited[p_deltaC_pos]**2.0 - deltaC_Climited[p_deltaC_pos]**2.0) # increase in chroma is not penalized!

            if ('Qa' in outlist) | ('Qai' in outlist):
                Qai[:,ii,None] = GA[ii]*scale_fcn(DEi_Climited,[scale_factor[1]])
                Qa[:,ii] = GA[ii]*scale_fcn(avg(DEi_Climited,axis = 0),[scale_factor[1]])
                
            if ('Qp' in outlist) | ('Qpi' in outlist):
                deltaC_pos = deltaC_Climited * (deltaC_Climited >= 0.0)
                deltaCmu = np.mean(deltaC_Climited * (deltaC_Climited >= 0.0))
                Qpi[:,ii,None] = GA[ii]*scale_fcn((DEi_Climited - deltaC_pos),[scale_factor[2]]) # or ?? np.sqrt(DEi_Climited**2 - deltaC_pos**2) ??
                Qp[:,ii] = GA[ii]*scale_fcn((avg(DEi_Climited, axis = 0) - deltaCmu),[scale_factor[2]])

    if ('Qg' in outlist):
        Qg = Qf.copy()
        for ii in range(labti.shape[1]):
            Qg[:,ii] = 100.0*math.polyarea(labti[:,ii,1],labti[:,ii,2])/math.polyarea(labri[:,ii,1],labri[:,ii,2]) # calculate Rg =  gamut area ratio of test and ref

     
    if out == 'Qa':
        return Qa
    else:
        return eval(out)
