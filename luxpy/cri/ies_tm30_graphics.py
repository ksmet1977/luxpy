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
###############################################################################
# Module for color rendition graphical output, 2
###############################################################################
#
# plot_cri_graphics(): Plots graphical information on color rendition properties based on spectral data input or dict with pre-calculated measures.
#
#

Created on Tue Apr  3 20:34:09 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .. import np, plt, colorsys, _CRI_RFL

from .helpers import gamut_slicer, spd_to_cri, jab_to_rhi
from .init_cri_defaults_database import _CRI_DEFAULTS

from .graphics import plot_ColorVectorGraphic
from .vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT, VF_colorshift_model
from .VF_PX_models import plot_VF_PX_models
from .ies_tm30_metrics import spd_to_ies_tm30_metrics
__all__ = ['plot_cri_graphics']




def plot_cri_graphics(data, cri_type = None, hbins = 16, start_hue = 0.0, scalef = 100, \
                      plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, \
                      plot_center_lines = False, plot_bin_colors = True, axtype = 'polar', ax = None, force_CVG_layout = True,
                      vf_model_type = _VF_MODEL_TYPE, vf_pcolorshift = _VF_PCOLORSHIFT, vf_color = 'k', \
                      vf_bin_labels = _VF_PCOLORSHIFT['labels'], vf_plot_bin_colors = True, scale_vf_chroma_to_sample_chroma = False,\
                      plot_VF = True, plot_CF = False, plot_SF = False):
    """
    Plot graphical information on color rendition properties.
    
    Args:
        :data: numpy.ndarray with spectral data or dict with pre-computed metrics.
        :cri_type: None, optional
            If None: defaults to cri_type = 'iesrf'.
            :hbins:, :start_hue: and :scalef: are ignored when cri_type is not None 
            and values are replaced by those in cri_type['rg_pars']
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
        :plot_bin_colors: True, optional
            Colorize hue bins.
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
        :vf_plot_bin_colors: True, optional
            Colorize hue bins of VF graph.
        :scale_vf_chroma_to_sample_chroma: False, optional
           Scale chroma of reference and test vf fields such that average of 
           binned reference chroma equals that of the binned sample chroma
           before calculating hue bin metrics.
        :vf_bin_labels: see :bin_labels:
            Set VF model hue-bin labels.
        :plot_CF: False, optional
            Plot circle fields.
        :plot_VF: True, optional
            Plot vector fields.
        :plot_SF: True, optional
            Plot sample shifts.   
            
    Returns:
        :returns: data, [plt.gcf(),ax_spd, ax_CVG, ax_locC, ax_locH, ax_VF], cmap 
        
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
            :[...]: list with handles to current figure and 5 axes.
            :cmap: list with rgb colors for hue bins (for use in other plotting fcns)
        
    """
    if not isinstance(data,dict):
        data = spd_to_ies_tm30_metrics(data, cri_type = cri_type, hbins = hbins, start_hue = start_hue, scalef = scalef, vf_model_type = vf_model_type, vf_pcolorshift = vf_pcolorshift, scale_vf_chroma_to_sample_chroma = scale_vf_chroma_to_sample_chroma)

    Rcshi, Rf, Rfchhi_vf, Rfhi, Rfhi_vf, Rfhshi_vf, Rfi, Rfm, Rfmi, Rg, Rhshi, SPD, bjabr, bjabt, cct, cri_type, dataVF, duv = [data[x] for x in sorted(data.keys())]
    hbins = cri_type['rg_pars']['nhbins']
    start_hue = cri_type['rg_pars']['start_hue']
    scalef = cri_type['rg_pars']['normalized_chroma_ref']
        
    #layout = np.array([[3,3,0,0],[1,0,2,2],[0,0,2,1],[2,2,1,1],[0,2,1,1],[1,2,1,1]])
    #layout = np.array([[6,6,0,0],[0,3,3,3],[3,3,3,3],[0,0,3,2],[2,2,2,2],[2,0,2,2],[4,0,2,2]])
    layout = np.array([[6,7,0,0],[0,4,3,3],[3,4,3,3],[0,0,4,2],[2,0,2,2],[4,2,2,2],[4,0,2,2],[2,2,2,2]])
    
    def create_subplot(layout,n, polar = False, frameon = True):
        ax = plt.subplot2grid(layout[0,0:2], layout[n,0:2], colspan = layout[n,2], rowspan = layout[n,3], polar = polar, frameon = frameon)
        return ax
    
    for i in range(cct.shape[0]):
        
        fig = plt.figure(figsize=(10, 6), dpi=144)
    
        # Plot CVG:
        ax_CVG = create_subplot(layout,1, polar = True, frameon = False)
        figCVG, ax, cmap = plot_ColorVectorGraphic(bjabt[...,i,:], bjabr[...,i,:], hbins = hbins, axtype = axtype, ax = ax_CVG, plot_center_lines = plot_center_lines, plot_edge_lines = plot_edge_lines,  plot_bin_colors = plot_bin_colors, scalef = scalef, force_CVG_layout = force_CVG_layout, bin_labels = '#')
        
        # Plot VF:
        ax_VF = create_subplot(layout,2, polar = True, frameon = False)
        if i == 0:
            hbin_cmap = None
    
        ax_VF, hbin_cmap = plot_VF_PX_models([dataVF[i]], dataPX = None, plot_VF = plot_VF, plot_PX = None, axtype = 'polar', ax = ax_VF, \
                           plot_circle_field = plot_CF, plot_sample_shifts = plot_SF, plot_bin_colors = vf_plot_bin_colors, \
                           plot_samples_shifts_at_pixel_center = False, jabp_sampled = None, \
                           plot_VF_colors = [vf_color], plot_PX_colors = ['r'], hbin_cmap = hbin_cmap, force_CVG_layout = True, bin_labels = vf_bin_labels)
    
        # Plot test SPD:
        ax_spd = create_subplot(layout,3)
        ax_spd.plot(SPD[0],SPD[i+1]/SPD[i+1].max(),'r-')
        ax_spd.text(730,0.9,'CCT = {:1.0f} K'.format(cct[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
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
