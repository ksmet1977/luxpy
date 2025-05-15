"""
###############################################################################
# Module for the memory color rendition index (MCRI), Rm
###############################################################################

 :_MCRI_DEFAULTS: default settings for MCRI 
                 (major dict has 13 keys (04-Mar-2025): 
                 sampleset [str/dict], 
                 ref_type [str], 
                 cieobs [Dict], 
                 cct_mode [str],
                 avg [fcn handle], 
                 rf_from_avg_rounded_rfi [Bool],
                 scale [dict], 
                 cspace [dict], 
                 catf [dict], 
                 rg_pars [dict], 
                 cri_specific_pars [dict])

 :spd_to_mcri(): Calculates the memory color rendition index, Rm:
    
Reference
    1. `K.A.G. Smet, W.R. Ryckaert, M.R. Pointer, G. Deconinck, P. Hanselaer,(2012)
    “A memory colour quality metric for white light sources,” 
    Energy Build., vol. 49, no. C, pp. 216–225.
    <http://www.sciencedirect.com/science/article/pii/S0378778812000837>`_

"""
import numpy as np 

from luxpy import cat, math, _CRI_RFL,  spd, spd_to_xyz, colortf, xyz_to_ipt, xyz_to_cct
from luxpy.utils import np2d, asplit
from ..utils.DE_scalers import psy_scale
from ..utils.helpers import _get_hue_bin_data, _hue_bin_data_to_rg


_MCRI_DEFAULTS = {'sampleset': "_CRI_RFL['mcri']", 
                  'ref_type' : None, 
                  'calculation_wavelength_range' : None,
                  'cieobs' : {'xyz' : '1964_10', 'cct': '1931_2'}, 
                  'cct_mode' : ('ohno2014', {}),
                  'avg': math.geomean, 
                  'rf_from_avg_rounded_rfi' : False,
                  'scale' : {'fcn': psy_scale, 'cfactor': [21.7016,   4.2106,   2.4154]}, 
                  'cspace': {'type': 'ipt', 'Mxyz2lms': [[ 0.400070,    0.707270,   -0.080674],[-0.228111, 1.150561,    0.061230],[0.0, 0.0,    0.931757]]}, 
                  'catf': {'xyzw': [94.81,  100.00,  107.32], 'mcat': 'cat02', 'cattype': 'vonkries', 'F':1, 'Yb': 20.0,'Dtype':'cat02', 'catmode' : '1>2'}, 
                  'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100, 'use_bin_avg_DEi' : False}, 
                  'cri_specific_pars' : {'similarity_ai' : np.array([[-0.09651, 0.41354, 40.64, 16.55, -0.17],
                                                                     [0.16548, 0.38877, 58.27,    20.37,    -0.59],
                                                                     [0.32825, 0.49673, 35.97    , 18.05,-6.04],
                                                                     [0.02115, -0.13658, 261.62, 110.99, -44.86], 
                                                                     [-0.12686, -0.22593, 99.06, 55.90, -39.86],
                                                                     [ 0.18488, 0.01172, 58.23, 62.55, -22.86],
                                                                     [-0.03440, 0.23480, 94.71, 32.12, 2.90],
                                                                     [ 0.04258, 0.05040, 205.54, 53.08, -35.20], 
                                                                     [0.15829,  0.13624, 90.21,  70.83, -19.01],
                                                                     [-0.01933, -0.02168, 742.97, 297.66, -227.30]])}
                }


__all__ = ['spd_to_mcri','_MCRI_DEFAULTS']

###############################################################################
def spd_to_mcri(SPD, D = 0.9, E = None, Yb = 20.0, out = 'Rm', wl = None,
                mcri_defaults = None, interp_settings = None):
    """
    Calculates the MCRI or Memory Color Rendition Index, Rm
    
    Args: 
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            |  first axis are the wavelengths)
        :D: 
            | 0.9, optional
            | Degree of adaptation.
        :E: 
            | None, optional
            | Illuminance in lux 
            |  (used to calculate La = (Yb/100)*(E/pi) to then calculate D 
            |  following the 'cat02' model). 
            | If None: the degree is determined by :D:
            |  If (:E: is not None) & (:Yb: is None):  :E: is assumed to contain 
            |  the adapting field luminance La (cd/m²).
        :Yb: 
            | 20.0, optional
            | Luminance factor of background. (used when calculating La from E)
            | If None, E contains La (cd/m²).
        :out: 
            | 'Rm' or str, optional
            | Specifies requested output (e.g. 'Rm,Rmi,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation   
        :mcri_defaults:
            | None, optional
            | Dictionary with structure of _MCRI_DEFAULTS containing everything
            | needed to calculate MCRI.
            | If None: _MCRI_DEFAULTS is used.
    
    Returns:
        :returns: 
            | float or ndarray with MCRI Rm for :out: 'Rm'
            | Other output is also possible by changing the :out: str value.        
          
    References:
        1. `K.A.G. Smet, W.R. Ryckaert, M.R. Pointer, G. Deconinck, P. Hanselaer,(2012)
        “A memory colour quality metric for white light sources,” 
        Energy Build., vol. 49, no. C, pp. 216–225.
        <http://www.sciencedirect.com/science/article/pii/S0378778812000837>`_
    """
    SPD = np2d(SPD)
    
    if wl is not None: 
        SPD = spd(data = SPD, datatype = 'spd', wl = wl, interp_settings = interp_settings)
    
    
    # unpack metric default values:  
    if mcri_defaults is None: mcri_defaults = _MCRI_DEFAULTS
    avg, calculation_wavelength_range, catf, cct_mode, cieobs, cri_specific_pars, cspace, ref_type, rf_from_avg_rounded_rfi, rg_pars, sampleset, scale = [mcri_defaults[x] for x in sorted(mcri_defaults.keys())] 
    similarity_ai = cri_specific_pars['similarity_ai']
    Mxyz2lms = cspace.get('Mxyz2lms',None)
    scale_fcn = scale['fcn']
    scale_factor = scale['cfactor']
    if isinstance(sampleset,str): sampleset = eval(sampleset)

    # Drop wavelengths outside of calculation_wavelength_range:
    if calculation_wavelength_range is not None:
        SPD = SPD[:,(SPD[0]>=calculation_wavelength_range[0]) & (SPD[0]<=calculation_wavelength_range[1])]

    
    # A. calculate xyz:
    xyzti, xyztw = spd_to_xyz(SPD, cieobs = cieobs['xyz'],  rfl = sampleset, out = 2)
    if 'cct' in out.split(','):
        if not (isinstance(cct_mode,tuple) | isinstance(cct_mode, list)): cct_mode = (cct_mode, {})
        cct, duv = xyz_to_cct(xyztw, cieobs = cieobs['cct'], cct_mode = cct_mode[0], out = 'cct,duv', **cct_mode[1])
        cct = np.abs(cct) # out-of-lut ccts are encoded as negative
        
    # B. perform chromatic adaptation to adopted whitepoint of ipt color space, i.e. D65:
    if catf is not None:
        Dtype_cat, F, Yb_cat, catmode_cat, cattype_cat, mcat_cat, xyzw_cat = [catf[x] for x in sorted(catf.keys())]
        
        # calculate degree of adaptationn D:
        if E is not None:
            if Yb is not None:
                La = (Yb/100.0)*(E/np.pi)
            else:
                La = E
            D = cat.get_degree_of_adaptation(Dtype = Dtype_cat, F = F, La = La)
        else:
            Dtype_cat = None # direct input of D

        if (E is None) and (D is None):
            D = 1.0 # set degree of adaptation to 1 !
        if D > 1.0: D = 1.0
        if D < 0.6: D = 0.6 # put a limit on the lowest D

        # apply cat:
        xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw,xyzw0 = None, xyzw2 = xyzw_cat, D = D, mcat = [mcat_cat], Dtype = Dtype_cat)
        xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw,xyzw0 = None, xyzw2 = xyzw_cat, D = D, mcat = [mcat_cat], Dtype = Dtype_cat)
     
    # C. convert xyz to ipt and split:
    if cspace['type'] == 'ipt': 
        ipt = xyz_to_ipt(xyzti, cieobs = cieobs['xyz'], M = Mxyz2lms) #input matrix as published in Smet et al. 2012, Energy and Buildings
    else:
        cspace.pop('type') # get rid of type key
        if 'xyzw' in cspace:
            if cspace['xyzw'] is None:
                cspace['xyzw'] = xyztw
        ipt = colortf(xyzti, fwtf = cspace)
        
    I,P,T = asplit(ipt)  

    # D. calculate specific (hue dependent) similarity indicators, Si:
    if len(xyzti.shape) == 3:
        ai = np.expand_dims(similarity_ai, axis = 1)
    else: 
        ai = similarity_ai
    a1,a2,a3,a4,a5 = asplit(ai)
    mahalanobis_d2 = (a3*np.power((P - a1),2.0) + a4*np.power((T - a2),2.0) + 2.0*a5*(P-a1)*(T-a2))
    if (len(mahalanobis_d2.shape)==3) & (mahalanobis_d2.shape[-1]==1):
        mahalanobis_d2 = mahalanobis_d2[:,:,0].T
    Si = np.exp(-0.5*mahalanobis_d2)

    # E. calculate general similarity indicator, Sa:
    Sa = avg(Si, axis = 0,keepdims = True)

    # F. rescale similarity indicators (Si, Sa) with a 0-1 scale to memory color rendition indices (Rmi, Rm) with a 0 - 100 scale:
    Rmi = scale_fcn(np.log(Si),scale_factor = scale_factor)
    Rm = np2d(scale_fcn(np.log(Sa),scale_factor = scale_factor))

    # G. calculate Rg (polyarea of test / polyarea of memory colours):
    if 'Rg' in out.split(','):
        I = I[...,None] #broadcast_shape(I, target_shape = None,expand_2d_to_3d = 0)
        a1 = a1[:,None]*np.ones(I.shape)#broadcast_shape(a1, target_shape = None,expand_2d_to_3d = 0)
        a2 = a2[:,None]*np.ones(I.shape) #broadcast_shape(a2, target_shape = None,expand_2d_to_3d = 0)
        a12 = np.concatenate((a1,a2),axis=2) #broadcast_shape(np.hstack((a1,a2)), target_shape = ipt.shape,expand_2d_to_3d = 0)
        ipt_mc = np.concatenate((I,a12),axis=2)
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue, use_bin_avg_DEi  = [rg_pars[x] for x in sorted(rg_pars.keys())]
    
        hue_bin_data = _get_hue_bin_data(ipt, ipt_mc, 
                                         start_hue = start_hue, nhbins = nhbins,
                                         normalized_chroma_ref = normalized_chroma_ref)
        Rg = _hue_bin_data_to_rg(hue_bin_data)

    if (out != 'Rm'):
        return  eval(out)
    else:
        return Rm