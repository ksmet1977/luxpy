# -*- coding: utf-8 -*-
"""
Module with Smet et al. (2018) neutral white loci
=================================================
 
 :_UW_NEUTRALITY_PARAMETERS_SMET2014: dict with parameters of the unique white models in Smet et al. (2014)

 :xyz_to_neutrality_smet2018(): Calculate degree of neutrality using the unique white model in Smet et al. (2014) or the normalized (max = 1) degree of chromatic adaptation model from Smet et al. (2017).

 :cct_to_neutral_loci_smet2018():  Calculate the most neutral appearing Duv10 in and the degree of neutrality for a specified CCT using the models in Smet et al. (2018).
 
References
----------
    1. `Smet, K. A. G. (2018). 
    Two Neutral White Illumination Loci Based on Unique White Rating and Degree of Chromatic Adaptation. 
    LEUKOS, 14(2), 55–67.  
    <https://doi.org/10.1080/15502724.2017.1385400>`_  
    
    2.`Smet, K., Deconinck, G., & Hanselaer, P. (2014). 
    Chromaticity of unique white in object mode. 
    Optics Express, 22(21), 25830–25841. 
    <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-21-25830>`_
    
    3. `Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017), 
    Study of chromatic adaptation using memory color matches, 
    Part II: colored illuminants, 
    Opt. Express, 25(7), pp. 8350-8365.
    <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-8350&origin=search)>`_

Added Aug 02, 2019.
"""

from luxpy import np, math, cat, xyz_to_Yuv, cri_ref, spd_to_xyz, xyz_to_cct

_UW_NEUTRALITY_PARAMETERS_SMET2014 = {'L200': np.array([1900.1, 1408.4, 0.2069, 0.4571, -934.1, 8.5, 6568, -0.0088 ])} 
_UW_NEUTRALITY_PARAMETERS_SMET2014['L1000'] = np.array([1418.3, 842.9, 0.2088, 0.4632, -659.8, 8.2, 6076, -0.0076])   
_UW_NEUTRALITY_PARAMETERS_SMET2014['L2000'] = np.array([1055.3, 782.4, 0.2104, 0.4665, -461.4, 7.7, 5798, -0.0073])   
_UW_NEUTRALITY_PARAMETERS_SMET2014['Linvar'] = np.array([1494.9, 981.9, 0.2081, 0.4596, -722.2, 8.1, 6324, -0.0087]) 

__all__ = ['_UW_NEUTRALITY_PARAMETERS_SMET2014', 'xyz_to_neutrality_smet2018','cct_to_neutral_loci_smet2018']


def xyz_to_neutrality_smet2018(xyz10, nlocitype = 'uw', uw_model = 'Linvar'):
    """
    Calculate degree of neutrality using the unique white model in Smet et al. (2014) or the normalized (max = 1) degree of chromatic adaptation model from Smet et al. (2017).
    
    Args:
        :xyz10:
            | ndarray with CIE 1964 10° xyz tristimulus values.
        :nlocitype:
            | 'uw', optional
            | 'uw': use unique white models published in Smet et al. (2014).
            | 'ca': use degree of chromatic adaptation model from Smet et al. (2017).
        :uw_model:
            | 'Linvar', optional
            | Use Luminance invariant unique white model from Smet et al. (2014).
            | Other options: 'L200' (200 cd/m²), 'L1000' (1000 cd/m²) and 'L2000' (2000 cd/m²).
    
    Returns:
        :N: 
            | ndarray with calculated neutrality
            
    References:
        1.`Smet, K., Deconinck, G., & Hanselaer, P. (2014). 
        Chromaticity of unique white in object mode. 
        Optics Express, 22(21), 25830–25841. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-21-25830>`_
        
        2. `Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017), 
        Study of chromatic adaptation using memory color matches, 
        Part II: colored illuminants, 
        Opt. Express, 25(7), pp. 8350-8365.
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-8350&origin=search)>`_
    """
    if nlocitype =='uw':
        uv = xyz_to_Yuv(xyz10)[...,1:]
        G0 = lambda up,vp,a: np.exp(-0.5 * (a[0]*(up-a[2])**2 + a[1]*(vp-a[3])**2 + 2*a[4]*(up-a[2])*(vp-a[3])))
        return G0(uv[...,0:1], uv[...,1:2], _UW_NEUTRALITY_PARAMETERS_SMET2014[uw_model])
    elif nlocitype == 'ca':
         return cat.smet2017_D(xyz10, Dmax = 1)
    else:
        raise Exception('Unrecognized nlocitype')


def cct_to_neutral_loci_smet2018(cct, nlocitype = 'uw', out = 'duv,D'):
    """
    Calculate the most neutral appearing Duv10 in and the degree of neutrality for a specified CCT using the models in Smet et al. (2018).
    
    Args:
        :cct10:
            | ndarray CCT 
        :nlocitype:
            | 'uw', optional
            | 'uw': use unique white models published in Smet et al. (2014).
            | 'ca': use degree of chromatic adaptation model from Smet et al. (2017).
        :out:
            | 'duv,D', optional
            | Specifies requested output (other options: 'duv', 'D').
            
    Returns:
        :duv: ndarray with most neutral Duv10 value corresponding to the cct input.
        :D: ndarray with the degree of neutrality at (cct, duv).
        
    References:
         1. `Smet, K. A. G. (2018). 
        Two Neutral White Illumination Loci Based on Unique White Rating and Degree of Chromatic Adaptation. 
        LEUKOS, 14(2), 55–67.  
        <https://doi.org/10.1080/15502724.2017.1385400>`_  
        
    Notes:
        1. Duv is specified in the CIE 1960 u10v10 chromatity diagram as the 
        models were developed using CIE 1964 10° tristimulus, chromaticity and CCT values.
        2. The parameter +0.0172 in Eq. 4b should be -0.0172
    """
    if nlocitype =='uw':
        duv = 0.0202 * np.log(cct/3325)*np.exp(-1.445*np.log(cct/3325)**2) - 0.0137
        D = np.exp(-(6368*((1/cct) - (1/6410)))**2) # degree of neutrality
    elif nlocitype =='ca':
        duv = 0.0382 * np.log(cct/2194)*np.exp(-0.679*np.log(cct/2194)**2) - 0.0172
        D = np.exp(-(3912*((1/cct) - (1/6795)))**2) # degree of adaptation
    else:
        raise Exception('Unrecognized nlocitype')
        
    if out == 'duv,D':
        return duv, D
    elif out == 'duv':
        return duv
    elif out == 'D':
        return D
    else:
        raise Exception('smet_white_loci(): Requested output unrecognized.')
        
    
if __name__ == '__main__':
    ccts = np.array([6605,6410,6800])
    BBs = cri_ref(ccts, ref_type = ['BB','BB','BB'])
    xyz10 = spd_to_xyz(BBs, cieobs='1964_10')
    ccts_calc = xyz_to_cct(xyz10, cieobs='1964_10')
    
    Dn_uw = xyz_to_neutrality_smet2018(xyz10, nlocitype='uw')
    Dn_ca = xyz_to_neutrality_smet2018(xyz10, nlocitype='ca')
    Duv10_uw, Dn_uw2 = cct_to_neutral_loci_smet2018(ccts, nlocitype='uw', out='duv,D')
    Duv10_ca, Dn_ca2 = cct_to_neutral_loci_smet2018(ccts, nlocitype='ca', out='duv,D')
    
    
    

