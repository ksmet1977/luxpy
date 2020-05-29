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

Extension module for IES TM30 metric calculation with additional Vector Field support
=====================================================================================

 :spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import _CRI_RFL
from luxpy.utils import np

from ..utils.helpers import gamut_slicer, spd_to_cri, jab_to_rhi
from ..utils.init_cri_defaults_database import _CRI_DEFAULTS

from ..VFPX.vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT, VF_colorshift_model
from ..VFPX.VF_PX_models import plot_VF_PX_models

__all__ = ['spd_to_ies_tm30_metrics']

def spd_to_ies_tm30_metrics(SPD, cri_type = None, \
                            hbins = 16, start_hue = 0.0,\
                            scalef = 100, \
                            vf_model_type = _VF_MODEL_TYPE, \
                            vf_pcolorshift = _VF_PCOLORSHIFT,\
                            scale_vf_chroma_to_sample_chroma = False):
    """
    Calculates IES TM30 metrics from spectral data.      
      
      Args:
        :data:
            | numpy.ndarray with spectral data 
        :cri_type:
            | None, optional
            | If None: defaults to cri_type = 'iesrf'.
            | Not none values of :hbins:, :start_hue: and :scalef: overwrite 
            | input in cri_type['rg_pars'] 
        :hbins:
            | None or numpy.ndarray with sorted hue bin centers (°), optional
        :start_hue: 
            | None, optional
        :scalef:
            | None, optional
            | Scale factor for reference circle.
        :vf_pcolorshift:
            | _VF_PCOLORSHIFT or user defined dict, optional
            | The polynomial models of degree 5 and 6 can be fully specified or 
            | summarized by the model parameters themselved OR by calculating the
            | dCoverC and dH at resp. 5 and 6 hues. :VF_pcolorshift: specifies 
            | these hues and chroma level.
        :scale_vf_chroma_to_sample_chroma: 
            | False, optional
            | Scale chroma of reference and test vf fields such that average of 
            | binned reference chroma equals that of the binned sample chroma
            | before calculating hue bin metrics.
            
    Returns:
        :data: 
            | dict with color rendering data:
            | - 'SPD'  : ndarray test SPDs
            | - 'bjabt': ndarray with binned jab data under test SPDs
            | - 'bjabr': ndarray with binned jab data under reference SPDs
            | - 'jabti': ndarray with individual jab data under test SPDs (scaled such that bjabr are on a circle)
            | - 'jabri': ndarray with individual jab data under reference SPDs (scaled such that bjabr are on a circle)
            | - 'hbinnr': ndarray with the hue bin number the samples belong to.
            | - 'cct'  : ndarray with CCT of test SPD
            | - 'duv'  : ndarray with distance to blackbody locus of test SPD
            | - 'Rf'   : ndarray with general color fidelity indices
            | - 'Rg'   : ndarray with gamut area indices
            | - 'Rfi'  : ndarray with specific color fidelity indices
            | - 'Rfhi' : ndarray with local (hue binned) fidelity indices
            | - 'Rcshi': ndarray with local chroma shifts indices
            | - 'Rhshi': ndarray with local hue shifts indices
            | - 'Rt'  : ndarray with general metameric uncertainty index Rt
            | - 'Rti' : ndarray with specific metameric uncertainty indices Rti
            | - 'Rfhi_vf' : ndarray with local (hue binned) fidelity indices 
            |               obtained from VF model predictions at color space
            |               pixel coordinates
            | - 'Rcshi_vf': ndarray with local chroma shifts indices 
            |               (same as above)
            | - 'Rhshi_vf': ndarray with local hue shifts indices 
            |               (same as above)
    """
    if cri_type is None:
        cri_type = 'iesrf'

    #Calculate color rendering measures for SPDs in data:
    out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type'
    if isinstance(cri_type,str): # get dict 
        cri_type = _CRI_DEFAULTS[cri_type].copy()
    if hbins is not None:
        cri_type['rg_pars']['nhbins'] = hbins 
    if start_hue is not None:
        cri_type['rg_pars']['start_hue'] = start_hue
    if scalef is not None:
        cri_type['rg_pars']['normalized_chroma_ref'] = scalef
    Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type = spd_to_cri(SPD, cri_type = cri_type, out = out)
    rg_pars = cri_type['rg_pars']

    
    #Calculate Metameric uncertainty and base color shifts:
    dataVF = VF_colorshift_model(SPD, cri_type = cri_type, model_type = vf_model_type, cspace = cri_type['cspace'], sampleset = eval(cri_type['sampleset']), pool = False, pcolorshift = vf_pcolorshift, vfcolor = 0)
    Rf_ = np.array([dataVF[i]['metrics']['Rf'] for i in range(len(dataVF))]).T
    Rt = np.array([dataVF[i]['metrics']['Rt'] for i in range(len(dataVF))]).T
    Rti = np.array([dataVF[i]['metrics']['Rti'] for i in range(len(dataVF))][0])
    
    # Get normalized and sliced sample data for plotting:
    rg_pars = cri_type['rg_pars']
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
    normalized_chroma_ref = scalef; # np.sqrt((jabr[...,1]**2 + jabr[...,2]**2)).mean(axis = 0).mean()
    
    if scale_vf_chroma_to_sample_chroma == True:
        normalize_gamut = False 
        bjabt, bjabr = gamut_slicer(jabt,jabr, out = 'jabt,jabr', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)
        Cr_s = (np.sqrt(bjabr[:-1,...,1]**2 + bjabr[:-1,...,2]**2)).mean(axis=0) # for rescaling vector field average reference chroma

    normalize_gamut = True #(for plotting)
    bjabt, bjabr, binnrs, jabti, jabri = gamut_slicer(jabt,jabr, out = 'jabt,jabr,binnr,jabti,jabri', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)


    Rfhi_vf = np.empty(Rfhi.shape)
    Rcshi_vf = np.empty(Rcshi.shape)
    Rhshi_vf = np.empty(Rhshi.shape)
    for i in range(cct.shape[0]):
        
        # Get normalized and sliced VF data for hue specific metrics:
        vfjabt = np.hstack((np.ones(dataVF[i]['fielddata']['vectorfield']['axt'].shape),dataVF[i]['fielddata']['vectorfield']['axt'],dataVF[i]['fielddata']['vectorfield']['bxt']))
        vfjabr = np.hstack((np.ones(dataVF[i]['fielddata']['vectorfield']['axr'].shape),dataVF[i]['fielddata']['vectorfield']['axr'],dataVF[i]['fielddata']['vectorfield']['bxr']))
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
        vfbjabt, vfbjabr, vfbDEi = gamut_slicer(vfjabt, vfjabr, out = 'jabt,jabr,DEi', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = False)
        
        if scale_vf_chroma_to_sample_chroma == True:
            #rescale vfbjabt and vfbjabr to same chroma level as bjabr.
            Cr_vfb = np.sqrt(vfbjabr[...,1]**2 + vfbjabr[...,2]**2)
            Cr_vf = np.sqrt(vfjabr[...,1]**2 + vfjabr[...,2]**2)
            hr_vf = np.arctan2(vfjabr[...,2],vfjabr[...,1])
            Ct_vf = np.sqrt(vfjabt[...,1]**2 + vfjabt[...,2]**2)
            ht_vf = np.arctan2(vfjabt[...,2],vfjabt[...,1])
            fC = Cr_s.mean()/Cr_vfb.mean()
            vfjabr[...,1] = fC * Cr_vf*np.cos(hr_vf)
            vfjabr[...,2] = fC * Cr_vf*np.sin(hr_vf)
            vfjabt[...,1] = fC * Ct_vf*np.cos(ht_vf)
            vfjabt[...,2] = fC * Ct_vf*np.sin(ht_vf)
            vfbjabt, vfbjabr, vfbDEi = gamut_slicer(vfjabt, vfjabr, out = 'jabt,jabr,DEi', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = False)

        scale_factor = cri_type['scale']['cfactor']
        scale_fcn = cri_type['scale']['fcn']
        vfRfhi, vfRcshi, vfRhshi = jab_to_rhi(jabt = vfbjabt, jabr = vfbjabr, DEi = vfbDEi, cri_type = cri_type, scale_factor = scale_factor, scale_fcn = scale_fcn, use_bin_avg_DEi = True) # [:-1,...] removes last row from jab as this was added to close the gamut. 

        Rfhi_vf[:,i:i+1] = vfRfhi
        Rhshi_vf[:,i:i+1] = vfRhshi
        Rcshi_vf[:,i:i+1] = vfRcshi

    # Create dict with CRI info:
    data = {'SPD' : SPD, 'cct' : cct, 'duv' : duv, 'bjabt' : bjabt, 'bjabr' : bjabr,\
            'jabti':jabti, 'jabri':jabri, 'hbinnr':binnrs,\
           'Rf' : Rf, 'Rg' : Rg, 'Rfi': Rfi, 'Rfhi' : Rfhi, 'Rcshi' : Rcshi, 'Rhshi' : Rhshi, \
           'Rt' : Rt, 'Rti' : Rti,  'Rfhi_vf' : Rfhi_vf, 'Rfcshi_vf' : Rcshi_vf, 'Rfhshi_vf' : Rhshi_vf, \
           'dataVF' : dataVF,'cri_type' : cri_type}
    return data