# -*- coding: utf-8 -*-

"""
###############################################################################
# Module for color rendition graphical output, 12
###############################################################################
#
# plot_cri_grpahics(): Plots graphical information on color rendition properties.
#
#

Created on Tue Apr  3 20:34:09 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .. import np, plt, colorsys
from .colorrendition_indices import _CRI_RFL, spd_to_cri, gamut_slicer,jab_to_rhi
from .colorrendition_graphics import plot_ColorVectorGraphic
from .colorrendition_vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT, VF_colorshift_model
from .colorrendition_VF_PX_models import plot_VF_PX_models

__all__ = ['plot_cri_graphics']




def plot_cri_graphics(data, cri_type = None, hbins = 16, start_hue = 0.0, scalef = 100, \
                      plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, \
                      plot_center_lines = False, axtype = 'polar', ax = None, force_CVG_layout = True,
                      vf_model_type = _VF_MODEL_TYPE, vf_pcolorshift = _VF_PCOLORSHIFT, vf_color = 'k', \
                      vf_bin_labels = _VF_PCOLORSHIFT['labels'], scale_vf_chroma_to_sample_chroma = False):
    """
    Plot graphical information on color rendition properties.
    
    Args:
        :data: numpy.ndarray with spectral data or dict with pre-calculated info on color rendering properties of SPDs
            If numpy.ndarray: calculate everything in current function, 
            If dict, it must have the following keys:
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
            
        :cri_type: None, optional
            Only needed when :data: is numpy.ndarray of spectral data. If None: defaults to cri_type = 'iesrf'.
        :hbins: 16 or numpy.ndarray with sorted hue bin centers (°), optional
        :start_hue: 0.0, optional
        :scalef: 100, optional
            Scale factor for graphic.
        :plot_axis_labels: False, optional
            Turns axis ticks on/off (True/False).
        :bin_labels: None or list[str] or '#', optional
            Plots labels at the bin center hues.
            - None: don't plot.
            - list[str]: list with str for each bin. (len(:bin_labels:) = :nhbins:)
            - '#': plots number.
        :plot_edge_lines: True or False, optional
            Plot grey bin edge lines with '--'.
        :plot_center_lines: False or True, optional
            Plot colored lines at 'center' of hue bin.
        :axtype: 'polar' or 'cart', optional
            Make polar or Cartesian plot.
        :ax: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes.
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
                :vf_model_type: _VF_MODEL_TYPE or 'M6' or 'M5', optional
            Type of polynomial vector field model to use for the calculation of
            base color shift and metameric uncertainty.
        :vf_pcolorshift: _VF_PCOLORSHIFT or user defined dict, optional
            The polynomial models of degree 5 and 6 can be fully specified or summarized 
            by the model parameters themselved OR by calculating the dCoverC and dH at resp. 5 and 6 hues.
            :VF_pcolorshift: specifies these hues and chroma level.
        :vf_color: 'k', optional
            For plotting the vector fields.
        :scale_vf_chroma_to_sample_chroma: False, optional
           Scale chroma of reference and test vf fields such that average of 
           binned reference chroma equals that of the binned sample chroma
           before calculating hue bin metrics.
        :vf_bin_labels: see :bin_labels:
            Set VF model hue-bin labels.
            
    Returns:
        :returns: data, [plt.gcf(),ax_spd, ax_CVG, ax_locC, ax_locH, ax_VF], cmap 
        
            :data: dict with color rendering data
            :[...]: list with handles to current figure and 5 axes.
            :cmap: list with rgb colors for hue bins (for use in other plotting fcns)
        
    """
    
    if isinstance(data,dict):
        #Unpack dict with pre-calculated data:
        Rcshi, Rf, Rfhi, Rfi, Rg, Rhshi, SPD, bjabr, bjabt, cct, duv = [data[x] for x in sorted(data.keys())]
        
    else:
        if cri_type is None:
            cri_type = 'iesrf'
        
        SPD = data 
        
        #Calculate color rendering measures for SPDs in data:
        out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type'
        Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type = spd_to_cri(SPD, cri_type = cri_type, out = out)
        rg_pars = cri_type['rg_pars']

        
        #Calculate Metameric uncertainty and base color shifts:
        dataVF = VF_colorshift_model(SPD, cri_type = cri_type, model_type = vf_model_type, cspace = cri_type['cspace'], sampleset = eval(cri_type['sampleset']), pool = False, pcolorshift = vf_pcolorshift, vfcolor = vf_color)
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
        data = {'SPD' : data, 'cct' : cct, 'duv' : duv, 'bjabt' : bjabt, 'bjabr' : bjabr,\
               'Rf' : Rf, 'Rg' : Rg, 'Rfi': Rfi, 'Rfhi' : Rfhi, 'Rchhi' : Rcshi, 'Rhshi' : Rhshi, \
               'Rfm' : Rfm, 'Rfmi' : Rfmi,  'Rfhi_vf' : Rfhi_vf, 'Rfchhi_vf' : Rcshi_vf, 'Rfhshi_vf' : Rhshi_vf, \
               'cri_type' : cri_type}
  
    layout = np.array([[3,3,0,0],[1,0,2,2],[0,0,2,1],[2,2,1,1],[0,2,1,1],[1,2,1,1]])
    layout = np.array([[6,6,0,0],[0,3,3,3],[3,3,3,3],[0,0,3,2],[2,2,2,2],[2,0,2,2],[4,0,2,2]])
    layout = np.array([[6,7,0,0],[0,4,3,3],[3,4,3,3],[0,0,4,2],[2,0,2,2],[4,2,2,2],[4,0,2,2],[2,2,2,2]])

    def create_subplot(layout,n, polar = False, frameon = True):
        ax = plt.subplot2grid(layout[0,0:2], layout[n,0:2], colspan = layout[n,2], rowspan = layout[n,3], polar = polar, frameon = frameon)
        return ax

    for i in range(cct.shape[0]):
        
        fig = plt.figure(figsize=(10, 6), dpi=144)

        # Plot CVG:
        ax_CVG = create_subplot(layout,1, polar = True, frameon = False)
        figCVG, ax, cmap = plot_ColorVectorGraphic(bjabt[...,i,:], bjabr[...,i,:], hbins = hbins, axtype = axtype, ax = ax_CVG, plot_center_lines = False, plot_edge_lines = True, scalef = scalef, force_CVG_layout = force_CVG_layout, bin_labels = '#')
        
        # Plot VF:
        ax_VF = create_subplot(layout,2, polar = True, frameon = False)
        if i == 0:
            hbin_cmap = None

        ax_VF, hbin_cmap = plot_VF_PX_models([dataVF[i]], dataPX = None, plot_VF = True, plot_PX = None, axtype = 'polar', ax = ax_VF, \
                           plot_circle_field = False, plot_sample_shifts = False, \
                           plot_samples_shifts_at_pixel_center = False, jabp_sampled = None, \
                           plot_VF_colors = ['k'], plot_PX_colors = ['r'], hbin_cmap = hbin_cmap, force_CVG_layout = True, bin_labels = vf_bin_labels)

        # Plot test SPD:
        ax_spd = create_subplot(layout,3)
        ax_spd.plot(SPD[0],SPD[i+1]/SPD[i+1].max(),'r-')
        ax_spd.text(730,0.9,'CCT = {:1.0f}K'.format(cct[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.8,'Duv = {:1.4f}'.format(duv[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.7,'IES Rf = {:1.0f}'.format(Rf[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.6,'IES Rg = {:1.0f}'.format(Rg[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.5,'Rmu = {:1.0f}'.format(Rfm[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.set_xlabel('Wavelength (nm)', fontsize = 9)
        ax_spd.set_ylabel('Rel. spectral intensity', fontsize = 9)
        ax_spd.set_xlim([360,830])
        
        # Plot local color fidelity, Rfhi:
        ax_Rfi = create_subplot(layout,4)
        for j in np.arange(hbins):
            ax_Rfi.bar(np.arange(hbins)[j],Rfhi[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_Rfi.text(np.arange(hbins)[j],Rfhi[j,i]*1.1, '{:1.0f}'.format(Rfhi[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
        ax_Rfi.set_ylim([0,120])
        xticks = np.arange(hbins)
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in range(hbins)]
        ax_Rfi.set_xticks(xticks)
        ax_Rfi.set_xticklabels(xtickslabels, fontsize = 8)
        ax_Rfi.set_ylabel(r'Local color fidelity $R_{f,hi}$')
        ax_Rfi.set_xlabel('Hue bin #')
        
        # Plot local chroma shift, Rcshi:
        ax_locC = create_subplot(layout,5)
        for j in np.arange(hbins):
            ax_locC.bar(np.arange(hbins)[j],Rcshi[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locC.text(np.arange(hbins)[j],-np.sign(Rcshi[j,i])*0.1, '{:1.0f}%'.format(100*Rcshi[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ylim = np.array([np.abs(Rcshi.min()),np.abs(Rcshi.min()),0.2]).max()*1.5
        ax_locC.set_ylim([-ylim,ylim])
        ax_locC.set_ylabel(r'Local chroma shift, $R_{cs,hi}$')
        ax_locC.set_xticklabels([])
        ax_locC.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locC.set_ylim()], color = 'white')
        
        # Plot local hue shift, Rhshi:
        ax_locH = create_subplot(layout,6)
        for j in np.arange(hbins):
            ax_locH.bar(np.arange(hbins)[j],Rhshi[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locH.text(np.arange(hbins)[j],-np.sign(Rhshi[j,i])*0.2, '{:1.3f}'.format(Rhshi[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ylim = np.array([np.abs(Rhshi.min()),np.abs(Rhshi.min()),0.2]).max()*1.5
        ax_locH.set_ylim([-ylim,ylim])
        ax_locH.set_ylabel(r'Local hue shift, $R_{hs,hi}$')
        ax_locH.set_xticklabels([])
        ax_locH.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locH.set_ylim()], color = 'white')
              
        # Plot local color fidelity of VF, vfRfhi:
        ax_vfRfi = create_subplot(layout,7)
        for j in np.arange(hbins):
            ax_vfRfi.bar(np.arange(hbins)[j],Rfhi_vf[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_vfRfi.text(np.arange(hbins)[j],Rfhi_vf[j,i]*1.1, '{:1.0f}'.format(Rfhi_vf[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
        ax_vfRfi.set_ylim([0,120])
        xticks = np.arange(hbins)
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in range(hbins)]
        ax_vfRfi.set_xticks(xticks)
        ax_vfRfi.set_xticklabels(xtickslabels, fontsize = 8)
        ax_vfRfi.set_ylabel(r'Local VF color fidelity $vfR_{f,hi}$')
        ax_vfRfi.set_xlabel('Hue bin #')
       
        plt.tight_layout()
        
    return  data,  [plt.gcf(),ax_spd, ax_CVG, ax_locC, ax_locH, ax_VF], cmap
