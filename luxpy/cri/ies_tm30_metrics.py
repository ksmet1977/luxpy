# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:16:08 2018

@author: kevin.smet
"""
from .. import np
from .colorrendition_indices import _CRI_RFL, _CRI_DEFAULTS, spd_to_cri, gamut_slicer,jab_to_rhi
from .colorrendition_vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT, VF_colorshift_model
from .colorrendition_VF_PX_models import plot_VF_PX_models

__all__ = ['spd_to_ies_tm30_metrics']

def spd_to_ies_tm30_metrics(SPD, cri_type = None, hbins = 16, start_hue = 0.0, scalef = 100, vf_model_type = _VF_MODEL_TYPE, vf_pcolorshift = _VF_PCOLORSHIFT,scale_vf_chroma_to_sample_chroma = False):
    """
    Calculates IES TM30 metrics from spectral data.      
      
      Args:
        :data: numpy.ndarray with spectral data 
        :cri_type: None, optional
            If None: defaults to cri_type = 'iesrf'.
            Not none values of :hbins:, :start_hue: and :scalef: overwrite input in cri_type['rg_pars'] 
        :hbins: None or numpy.ndarray with sorted hue bin centers (°), optional
        :start_hue: None, optional
        :scalef: None, optional
            Scale factor for reference circle.
        :vf_pcolorshift: _VF_PCOLORSHIFT or user defined dict, optional
            The polynomial models of degree 5 and 6 can be fully specified or summarized 
            by the model parameters themselved OR by calculating the dCoverC and dH at resp. 5 and 6 hues.
            :VF_pcolorshift: specifies these hues and chroma level.
        :scale_vf_chroma_to_sample_chroma: False, optional
           Scale chroma of reference and test vf fields such that average of 
           binned reference chroma equals that of the binned sample chroma
           before calculating hue bin metrics.
            
    Returns:
        :returns: data dict with keys:
     
            :data: dict with color rendering data
                - key: 'SPD' : numpy.ndarray test SPDs
                - key: 'bjabt': numpy.ndarray with binned jab data under test SPDs
                - key: 'bjabr': numpy.ndarray with binned jab data under reference SPDs
                - key: 'cct' : numpy.ndarray with correlated color temperatures of test SPD
                - key: 'duv' : numpy.ndarray with distance to blackbody locus of test SPD
                - key: 'Rf'  : numpy.ndarray with general color fidelity indices
                - key: 'Rg'  : numpy.ndarray with gamut area indices
                - key: 'Rfi'  : numpy.ndarray with specific color fidelity indices
                - key: 'Rfhi'  : numpy.ndarray with local (hue binned) color fidelity indices
                - key: 'Rcshi'  : numpy.ndarray with local chroma shifts indices
                - key: 'Rhshi'  : numpy.ndarray with local hue shifts indices
                - key: 'Rfm' : numpy.ndarray with general metameric uncertainty index Rfm
                - key: 'Rfmi' : numpy.ndarray with specific metameric uncertainty indices Rfmi
                - key: 'Rfhi_vf'  : numpy.ndarray with local (hue binned) color fidelity indices 
                                    obtained from VF model predictions at color space pixel coordinates
                - key: 'Rcshi_vf'  : numpy.ndarray with local chroma shifts indices (same as above)
                - key: 'Rhshi_vf'  : numpy.ndarray with local hue shifts indices (same as above)

    """
    if cri_type is None:
        cri_type = 'iesrf'
    print(cri_type)
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
    Rfm = np.array([dataVF[i]['metrics']['Rfm'] for i in range(len(dataVF))]).T
    Rfmi = np.array([dataVF[i]['metrics']['Rfmi'] for i in range(len(dataVF))][0])
    
    # Get normalized and sliced sample data for plotting:
    rg_pars = cri_type['rg_pars']
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
    normalized_chroma_ref = scalef; # np.sqrt((jabr[...,1]**2 + jabr[...,2]**2)).mean(axis = 0).mean()
    
    if scale_vf_chroma_to_sample_chroma == True:
        normalize_gamut = False 
        bjabt, bjabr = gamut_slicer(jabt,jabr, out = 'jabt,jabr', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)
        Cr_s = (np.sqrt(bjabr[:-1,...,1]**2 + bjabr[:-1,...,2]**2)).mean(axis=0) # for rescaling vector field average reference chroma

    normalize_gamut = True #(for plotting)
    bjabt, bjabr = gamut_slicer(jabt,jabr, out = 'jabt,jabr', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)


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
           'Rf' : Rf, 'Rg' : Rg, 'Rfi': Rfi, 'Rfhi' : Rfhi, 'Rchhi' : Rcshi, 'Rhshi' : Rhshi, \
           'Rfm' : Rfm, 'Rfmi' : Rfmi,  'Rfhi_vf' : Rfhi_vf, 'Rfcshi_vf' : Rcshi_vf, 'Rfhshi_vf' : Rhshi_vf, \
           'dataVF' : dataVF,'cri_type' : cri_type}
    return data