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
Module for IES color rendition graphical output
=================================================

 :_tm30_process_spd(): Calculate all required parameters for plotting from spd using cri.spd_to_cri()

 :plot_tm30_cvg(): Plot TM30 Color Vector Graphic (CVG).
 
 :plot_tm30_Rfi(): Plot Sample Color Fidelity values (Rfi).
 
 :plot_tm30_Rxhj(): Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj).

 :plot_tm30_Rcshj(): Plot Local Chroma Shifts (Rcshj).

 :plot_tm30_Rhshj(): Plot Local Hue Shifts (Rhshj).

 :plot_tm30_Rfhj(): Plot Local Color Fidelity values (Rfhj).

 :plot_tm30_spd(): Plot test SPD and reference illuminant, both normalized to the same luminous power.

 :plot_tm30_report(): Plot a figure with an ANSI/IES-TM30 color rendition report.
 
 
 :plot_cri_graphics(): Plots graphical information on color rendition 
                       properties based on spectral data input or dict with 
                       pre-calculated measures (cusom design). 
                       Includes Metameric uncertainty index Rt and vector-fields
                       of color rendition shifts.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import colorsys
import matplotlib.patches as patches

from luxpy.utils import np, plt
from luxpy import (math, cat, _CIE_D65, xyz_to_srgb, spd_to_power, 
                   spd_normalize, spd_to_xyz, xyz_to_Yxy, xyz_to_Yuv)
from luxpy.color.cri.utils.helpers import spd_to_cri
from luxpy.color.cri.utils.graphics import plot_ColorVectorGraphic

from luxpy.color.cri.utils.graphics import plot_ColorVectorGraphic
from luxpy.color.cri.VFPX.vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT 
from luxpy.color.cri.VFPX.VF_PX_models import plot_VF_PX_models
from luxpy.color.cri.iestm30.metrics import spd_to_ies_tm30_metrics

_TM30_FONT_SIZE = 8

__all__ = ['_tm30_process_spd','plot_tm30_cvg','plot_tm30_Rfi',
           'plot_tm30_Rxhj','plot_tm30_Rcshj', 'plot_tm30_Rhshj', 
           'plot_tm30_Rfhj', 'plot_tm30_spd','plot_tm30_report', 
           'spd_to_tm30_report',
           'plot_cri_graphics']

def _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs):
    """
    Calculate all required parameters for plotting from spd using cri.spd_to_cri()
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :data:
            | dictionary with required parameters for plotting functions.      
    """

    if not isinstance(spd,dict):
        data = spd_to_cri(spd, cri_type = cri_type, out = 'data', 
                          fit_gamut_ellipse = True, **kwargs)
    else:
        data = spd
    return data

def _get_hue_map(hbins = 16, start_hue = 0.0, 
                 hbinnrs = None, xyzri = None, xyzrw = None, cri_type = None):
    """
    Generate color map for hue bins.
    
    Args:
        :hbins:
            | 16 or ndarray with sorted hue bin centers (°), optional
        :start_hue:
            | 0.0, optional
        :hbinnrs: 
            | None, optional
            | ndarray with hue bin number of each sample.
            | If hbinnrs, xyzri, xyzrw and cri_type are all not-None: 
            |    use these to calculate color map, otherwise just use number of
            |    hue bins :hbins: and :start_hue:
        :xyzri:
            | None, optional
            | relative xyz tristimulus values of samples under ref. illuminant.
            | see :hbinnrs: for more info when this is used.
        :xyzrw:
            | None, optional
            | relative xyz tristimulus values of ref. illuminant white point.
            | see :hbinnrs: for more info when this is used.
        :cri_type:
            | None, optional
            | Specifies dict with default cri model parameters 
            | (needed to get correct :cieobs:) 
            | see :hbinnrs: for more info when this is used.
    
    Returns:
        :cmap:
            | list with rgb values (one for each hue bin) for plotting.
    """
    # Setup hbincenters and hsv_hues:
    if isinstance(hbins,float) | isinstance(hbins,int):
        nhbins = hbins
        dhbins = 360/(nhbins) # hue bin width
        hbincenters = np.arange(start_hue + dhbins/2, 360, dhbins)
        hbincenters = np.sort(hbincenters)
    else:
        hbincenters = hbins
        idx = np.argsort(hbincenters)
        hbincenters = hbincenters[idx]
        nhbins = hbincenters.shape[0]
    
    cmap = []
    if (hbinnrs is not None) & (xyzri is not None) & (xyzrw is not None) & (cri_type is not None):
        xyzw = spd_to_xyz(_CIE_D65, relative = True, cieobs = cri_type['cieobs']['xyz'])
        xyzri = cat.apply(xyzri[:,0,:],xyzw1 = xyzrw, xyzw2 = xyzw)
                
        # Create color from xyz average:
        for i in range(nhbins):
            xyzrhi = xyzri[hbinnrs[:,0] == i,:].mean(axis=0,keepdims=True)
            rgbrhi = xyz_to_srgb(xyzrhi)/255
            cmap.append(rgbrhi)
    else:
        # Create color from hue angle:
            
        # Setup color for plotting hue bins:
        hbincenters = hbincenters*np.pi/180
        hsv_hues = hbincenters - 30*np.pi/180
        hsv_hues = hsv_hues/hsv_hues.max()
            
        for i in range(nhbins):   
            #c = np.abs(np.array(colorsys.hsv_to_rgb(hsv_hues[i], 0.75, 0.85)))
            c = np.abs(np.array(colorsys.hls_to_rgb(hsv_hues[i], 0.45, 0.5)))
            cmap.append(c)
    
    return cmap

def plot_tm30_cvg(spd, cri_type = 'ies-tm30',  
                  gamut_line_color = 'r',
                  gamut_line_style = '-',
                  gamut_line_marker = 'o',
                  gamut_line_label = None,
                  plot_vectors = True,
                  plot_index_values = True,
                  axh = None, axtype = 'cart',
                  **kwargs):
    """
    Plot TM30 Color Vector Graphic (CVG).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :gamut_line_color:
            | 'r', optional
            | Plotting line style for the line connecting the 
            | average test chromaticity in the hue bins.
        :gamut_line_style:
            | 'r', optional
            | Plotting color for the line connecting the 
            | average test chromaticity in the hue bins.
        :gamut_line_marker:
            | '-', optional
            | Markers to plot the test color gamut points for each hue bin in 
            | (only used when plot_vectors = False).
        :gamut_line_label:
            | None, optional
            | Label for gamut line. (only used when plot_vectors = False).
        :plot_vectors:
            | True, optional
            | Plot color shift vectors in CVG (True) or not (False).
        :plot_index_values:
            | True, optional
            | Print Rf, Rg, CCT and Duv in corners of CVG (True) or not (False).
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes. 
        :axtype: 
            | 'cart' (or 'polar'), optional
            | Make Cartesian (default) or polar plot.    
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions. 
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30', **kwargs)
    hdata = data['hue_bin_data']
    
    # Normalized chroma (closed gamut):
    jabt = hdata['jabtn_hj_closed'][:,0,:]
    jabr = hdata['jabrn_hj_closed'][:,0,:]
    
    # Plot color vector graphic
    _, axh, _ = plot_ColorVectorGraphic(jabt = jabt, jabr = jabr, 
                                        hbins = hdata['nhbins'], 
                                        start_hue = hdata['start_hue'], 
                                        bin_labels = '',
                                        gamut_line_color = gamut_line_color,
                                        gamut_line_style = gamut_line_style,
                                        gamut_line_marker = gamut_line_marker,
                                        gamut_line_label = gamut_line_label,
                                        plot_vectors = plot_vectors,
                                        ax = axh, axtype = axtype,
                                        force_CVG_layout = True,
                                        plot_axis_labels = False)
    
    # Print Rf, Rg, CCT and Duv in plot:
    if plot_index_values == True:
        Rf, Rg, cct, duv = data['Rf'], data['Rg'], data['cct'], data['duv']
        scalef = data['hue_bin_data']['normalized_chroma_ref']
        axh.text(-1.30*scalef,1.30*scalef,'{:1.0f}'.format(Rf[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='center',verticalalignment='center',color = 'k')
        axh.text(-1.33*scalef,1.12*scalef,'$R_f$',fontsize = 13, style='italic', horizontalalignment='center',verticalalignment='center',color = 'k')
        axh.text(1.30*scalef,1.30*scalef,'{:1.0f}'.format(Rg[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='center',verticalalignment='center',color = 'k')
        axh.text(1.33*scalef,1.12*scalef,'$R_g$',fontsize = 13, style='italic', horizontalalignment='center',verticalalignment='center',color = 'k')
        axh.text(-1.43*scalef,-1.45*scalef,'{:1.0f}'.format(cct[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='left',verticalalignment='bottom',color = 'k')
        axh.text(-1.43*scalef,-1.25*scalef,'$CCT$',fontsize = 13, style='italic', horizontalalignment='left',verticalalignment='bottom',color = 'k')
        axh.text(1.43*scalef,-1.45*scalef,'{:1.4f}'.format(duv[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='right',verticalalignment='bottom',color = 'k')
        axh.text(1.43*scalef,-1.25*scalef,'$D_{uv}$',fontsize = 13, style='italic', horizontalalignment='right',verticalalignment='bottom',color = 'k')
    axh.set_xticks([])
    axh.set_yticks([])
    return axh, data 


def plot_tm30_spd(spd, cri_type = 'ies-tm30', axh = None, 
                  font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot test SPD and reference illuminant, both normalized to the same luminous power.
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes.   
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.  
  
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    # Normalize Sr to same luminous power as spd:
    Phiv_spd = spd_to_power(data['St'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    #Phiv_Sr = spd_to_power(data['Sr'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    data['Sr'] = spd_normalize(data['Sr'], norm_type = 'pu', norm_f = Phiv_spd, cieobs = data['cri_type']['cieobs']['cct'])
    
    # Plot test and ref SPDs:
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    axh.plot(data['Sr'][0,:], data['Sr'][1,:],'k-', label = 'Reference')
    axh.plot(data['St'][0,:], data['St'][1,:],'r-', label = 'Test')
    axh.set_xlabel('Wavelength (nm)', fontsize = font_size)
    axh.set_ylabel('Radiant power\n(Equal Luminous Flux)', fontsize = font_size)
    axh.set_xlim([360,830]) 
    axh.set_yticklabels([])
    axh.legend(loc = 'upper right', fontsize = font_size)
    
    return axh, data


def plot_tm30_Rfi(spd, cri_type = 'ies-tm30', axh = None, 
                  font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot Sample Color Fidelity values (Rfi).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes.    
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.     
    """
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)

    Rfi = data['Rfi']
    
    # get rgb values representing each sample:
    N = data['xyzri'].shape[0]
    xyzw = spd_to_xyz(_CIE_D65, relative = True, cieobs = data['cri_type']['cieobs']['xyz'])
    xyzri = cat.apply(data['xyzri'][:,0,:],xyzw1 = data['xyzrw'], xyzw2 = xyzw)
    rgbri = xyz_to_srgb(xyzri)/255
    
    # Create color map:
    cmap = []
    for i in range(N):
        cmap.append(rgbri[i,...])

    # Plot sample color fidelity, Rfi:
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in range(N):
        axh.bar(j,Rfi[j,0], color = cmap[j], width = 1,edgecolor = None, alpha = 0.9)
        #axh.text(j,Rfi[j,0]*1.1, '{:1.0f}'.format(Rfi[j,0]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
    xticks = np.arange(0,N,step=2)
    xtickslabels = ['CES{:1.0f}'.format(ii+1) for ii in range(0,N,2)]
    axh.set_xticks(xticks)
    axh.set_xticklabels(xtickslabels, fontsize = font_size, rotation = 90)
    axh.set_ylabel(r'Color Sample Fidelity $(R_{f,CESi})$', fontsize = font_size)
    axh.set_ylim([0,100])
    axh.set_xlim([-0.5,N-0.5])
    
    return axh, data

def plot_tm30_Rfhj(spd, cri_type = 'ies-tm30', axh = None, 
                   xlabel = True, y_offset = 0, 
                   font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot Local Color Fidelity values (Rfhj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes. 
        :xlabel:
            | True, optional
            | If False: don't add label and numbers to x-axis 
            | (useful when plotting plotting all 'Local Rfhi, Rcshi, Rshhi' 
            |  values in 3x1 subplots with 'shared x-axis': saves vertical space)
        :y_offset:
            | 0, optional
            | text-offset from top of bars in barplot.
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.     
    """
    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    hdata = data['hue_bin_data']
    Rfhj = data['Rfhj']
        
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = hdata['nhbins'], start_hue = hdata['start_hue'], 
                        hbinnrs = hdata['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])

    # Plot local color fidelity, Rfhj:
    hbins = range(hdata['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],Rfhj[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = ((np.abs(Rfhj[j,0]) + 2 + y_offset))*np.sign(Rfhj[j,0])
        axh.text(hbins[j],ypos, '{:1.0f}'.format(Rfhj[j,0]),
                 fontsize = font_size,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color = np.array([1,1,1])*0.3)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = font_size)
    else:
        xtickslabels = ['' for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = font_size)
    axh.set_xlim([-0.5,hdata['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Color Fidelity $(R_{f,hj})$', fontsize = font_size)
    axh.set_ylim([0,110])

    return axh, data

def plot_tm30_Rcshj(spd, cri_type = 'ies-tm30', axh = None, 
                    xlabel = True, y_offset = 0, 
                    font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot Local Chroma Shift values (Rcshj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes.  
        :xlabel:
            | True, optional
            | If False: don't add label and numbers to x-axis 
            | (useful when plotting plotting all 'Local Rfhi, Rcshi, Rshhi' 
            |  values in 3x1 subplots with 'shared x-axis': saves vertical space)
        :y_offset:
            | 0, optional
            | text-offset from top of bars in barplot.
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.   
    """

    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    hdata = data['hue_bin_data']
    Rcshj = data['Rcshj']
    
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = hdata['nhbins'], start_hue = hdata['start_hue'], 
                        hbinnrs = hdata['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])
    
    # Plot local chroma shift, Rcshj:
    hbins = range(hdata['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],100*Rcshj[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = 100*((np.abs(Rcshj[j,0]) + 0.05 + y_offset))*np.sign(Rcshj[j,0])
        axh.text(hbins[j]+0.05,ypos, '{:1.0f}%'.format(100*Rcshj[j,0]), 
                 fontsize = font_size,horizontalalignment='center',
                 verticalalignment='center',
                 color = np.array([1,1,1])*0.3, 
                 rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = font_size)
    else:
        xtickslabels = ['' for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = font_size)
    axh.set_xlim([-0.5,hdata['nhbins']-0.5])
    
    yticks = range(-40,50,10)
    axh.set_yticks(yticks)
    ytickslabels = ['{:1.0f}%'.format(ii) for ii in range(-40,50,10)]
    axh.set_yticklabels(ytickslabels, fontsize = font_size)
    axh.set_ylabel(r'Local Chroma Shift $(R_{cs,hj})$', fontsize = font_size)
    axh.set_ylim([min([-50,100*Rcshj.min()]),max([50,100*Rcshj.max()])])
    
    return axh, data

def plot_tm30_Rhshj(spd, cri_type = 'ies-tm30', axh = None, 
                    xlabel = True, y_offset = 0, 
                    font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot Local Hue Shift values (Rhshj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes. 
        :xlabel:
            | True, optional
            | If False: don't add label and numbers to x-axis 
            | (useful when plotting plotting all 'Local Rfhi, Rcshi, Rshhi' 
            |  values in 3x1 subplots with 'shared x-axis': saves vertical space)
        :y_offset:
            | 0, optional
            | text-offset from top of bars in barplot.
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.     
    """

    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    hdata = data['hue_bin_data']
    Rhshj = data['Rhshj']

    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = hdata['nhbins'], start_hue = hdata['start_hue'], 
                        hbinnrs = hdata['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])
    
    # Plot local hue shift, Rhshj:
    hbins = range(hdata['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],Rhshj[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = ((np.abs(Rhshj[j,0]) + 0.05 + y_offset))*np.sign(Rhshj[j,0])
        axh.text(hbins[j]+0.05,ypos, '{:1.2f}'.format(Rhshj[j,0]),
                 fontsize = font_size,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color = np.array([1,1,1])*0.3, 
                 rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = font_size)
    else:
        xtickslabels = ['' for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = font_size)
    axh.set_xlim([-0.5,hdata['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Hue Shift $(R_{hs,hj})$', fontsize = font_size)
    axh.set_ylim([min([-0.55,Rhshj.min()]),max([0.55,Rhshj.max()])])
    
    return axh, data

def plot_tm30_Rxhj(spd, cri_type = 'ies-tm30', axh = None, figsize = (6,15),
                   font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :axh: 
            | None, optional
            | If None: create new figure with single axes, else plot on specified axes. 
        :figsize:
            | (6,15), optional
            | Figure size of pyplot figure.
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh: 
            | handle to figure axes.
        :data:
            | dictionary with required parameters for plotting functions.     
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    if axh is None:
        fig, axh = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = figsize)
    
    plot_tm30_Rcshj(data, axh = axh[0], xlabel = False, y_offset = 0.02, font_size = font_size)
    plot_tm30_Rhshj(data, axh = axh[1], xlabel = False, y_offset = 0.03, font_size = font_size)
    plot_tm30_Rfhj(data, axh = axh[2], xlabel = True, y_offset = 2, font_size = font_size)
    return axh, data

def _split_notes(notes, max_len_notes_line = 40):
    """
    Split a string on white spaces over multiple lines, such that the line length doesn't exceed a specified value.
    
    Args:
        :notes:
            | string to be split
        :max_len_notes_line:
            | 40, optional
            | Maximum length of a single line when splitting the string.
    
    Returns:
        :notes_:
            | string with '\n' added at the right places to not exceed a certain width.
    """
    #l = len(notes)
    #n = l//max_len_notes_line + 1
    notes = notes.split()
    line = ''
    notes_ = ''
    i = 0
    while (i < len(notes)):
        if (len(line) + len(notes[i]) + 1) > max_len_notes_line: 
            notes_ = notes_ + line + '\n'
            line = notes[i] + ' '
        else:
            line = line + notes[i] + ' '
        i += 1
    notes_ = notes_ + line[:-1]
    return notes_

def _plot_tm30_report_top(axh, source = '', manufacturer = '', date = '', model = ''):
    """
    Print source name, source model, manufacturer and date in an empty axes.

    Args:
        :axh: 
            | Plot on specified axes. 
        :source:
            | string with source name.
        :manufacturer:
            | string with source manufacturer.
        :model:
            | string with source model.
        :date:
            | string with source measurement date.

    Returns:
        :axh:
            | handle to figure axes. 
    """
    axh.set_xticks(np.arange(10))
    axh.set_xticklabels(['' for i in np.arange(10)])
    axh.set_yticks(np.arange(2))
    axh.set_yticklabels(['' for i in np.arange(2)])
    axh.set_axis_off()
    axh.set_xlabel([])
    
    axh.text(0,1, 'Source: ' + source, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(0,0, '   Date: ' + date, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(5,1, 'Manufacturer: ' + manufacturer, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(5,0, 'Model: ' + model, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    return axh

def _plot_tm30_report_bottom(axh, spd, notes = '', max_len_notes_line = 40):
    """
    Print some notes, the CIE x, y, u',v' and Ra, R9 values of the source in some empty axes.
    
    Args:
        :axh: 
            | None, optional
            | Plot on specified axes. 
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
        :notes:
            | string to be split
        :max_len_notes_line:
            | 40, optional
            | Maximum length of a single line when splitting the string.
        
    Returns:
        :axh:
            | handle to figure axes.    
    """
    ciera = spd_to_cri(spd, cri_type = 'ciera', out = 'Rf')
    cierai = spd_to_cri(spd, cri_type = 'ciera-14', out = 'Rfi')
    xyzw = spd_to_xyz(spd, cieobs = '1931_2', relative = True)
    Yxyw = xyz_to_Yxy(xyzw)
    Yuvw = xyz_to_Yuv(xyzw)
    
    notes_ = _split_notes(notes, max_len_notes_line = max_len_notes_line)

    axh.set_xticks(np.arange(10))
    axh.set_xticklabels(['' for i in np.arange(10)])
    axh.set_yticks(np.arange(4))
    axh.set_yticklabels(['' for i in np.arange(4)])
    axh.set_axis_off()
    axh.set_xlabel([])
    
    axh.text(0,2.8, 'Notes: ', fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(0.75,2.8,  notes_, fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,2.8, "x   {:1.4f}".format(Yxyw[0,1]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,2.2, "y   {:1.4f}".format(Yxyw[0,2]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,1.6, "u'  {:1.4f}".format(Yuvw[0,1]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,1.0, "v'  {:1.4f}".format(Yuvw[0,2]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,2.8, "CIE 13.3-1995", fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,2.2, "     (CRI)    ", fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,1.6, "    $R_a$  {:1.0f}".format(ciera[0,0]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,1.0, "    $R_9$  {:1.0f}".format(cierai[9,0]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')

    # Create a Rectangle patch
    rect = patches.Rectangle((7.2,0.5),1.7,2.5,linewidth=1,edgecolor='k',facecolor='none')
    
    # Add the patch to the Axes
    axh.add_patch(rect)

    return axh

def plot_tm30_report(spd, cri_type = 'ies-tm30',
                     source = '', manufacturer = '',
                     date = '', model = '', 
                     notes = '', max_len_notes_line = 40,
                     figsize = (7,12),
                     save_fig_name = None, dpi = 300,
                     plot_report_top = True, plot_report_bottom = True,
                     suptitle = 'ANSI/IES TM-30-18 Color Rendition Report',
                     font_size = _TM30_FONT_SIZE, **kwargs):
    """
    Create TM30 Color Rendition Report.
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters (using _tm30_process_spd()).
            |  required keys:
            |   dict_keys(['St', 'Sr', 'xyztw_cct', 'cct', 'duv', 
            |               'xyzti', 'xyztw', 'xyzri', 'xyzrw', 
            |               'DEi', 'DEa', 'Rf', 'Rg', 
            |               'Rcshj', 'Rhshj', 'Rfhj', 'hue_bin_data'])
            | see cri.spd_to_cri() for more info on parameters.
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments (in kwargs) 
            | to the function will override default values in cri_type dict.
        :source:
            | string with source name.
        :manufacturer:
            | string with source manufacturer.
        :model:
            | string with source model.
        :date:
            | string with source measurement date.
        :notes:
            | string to be split
        :max_len_notes_line:
            | 40, optional
            | Maximum length of a single line when splitting the string.
        :figsize:
            | (7,12), optional
            | Figure size of pyplot figure.
        :save_fig_name:
            | None, optional
            | Filename (+path) to which the report will be saved as an image (png).
            | If None: don't save, just display.
        :dpi:
            | 300, optional
            | Dots-Per-Inch of image file (PNG).
        :plot_report_top:
            | execute _plot_tm30_report_top()
        :plot_report_bottom:
            | execute _plot_tm30_report_bottom()
        :suptitle:
            | 'ANSI/IES TM-30-18 Color Rendition Report' or str, optional
            | report title (input for plt.suptitle).  
        :font_size:
            | _TM30_FONT_SIZE, optional
            | Font size of text, axis labels and axis values.
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axs:
            | dictionary with handles to each axes.
        :data:
            | dictionary with required parameters for plotting functions.      
    """
    # Set up subplots:
    fig = plt.figure(constrained_layout=True, figsize = figsize)  
    nrows = int(4 + 1*(plot_report_top) + 1*(plot_report_bottom))
    gs = fig.add_gridspec(nrows, 3,height_ratios=[0.1,0.5,0.5,0.5,0.6,0.3], width_ratios=[1,1,1.5])
    if plot_report_top == True: 
        f_ax_top = fig.add_subplot(gs[0, :])
    else:
        f_ax_top = None
    if plot_report_bottom == True:
        f_ax_bottom = fig.add_subplot(gs[-1, :]) 
    else:
        f_ax_bottom = None
    f_ax_spd = fig.add_subplot(gs[int(0 + 1*(plot_report_top)), 0:2])
    f_ax_cvg = fig.add_subplot(gs[int(1 + 1*(plot_report_top)):int(3 + 1*(plot_report_top)), 0:2])
    f_ax_cshj = fig.add_subplot(gs[int(0 + 1*(plot_report_top)), 2:])
    f_ax_hshj = fig.add_subplot(gs[int(1 + 1*(plot_report_top)), 2:])
    f_ax_fhj = fig.add_subplot(gs[int(2 + 1*(plot_report_top)), 2:])
    f_ax_fi = fig.add_subplot(gs[int(3 + 1*(plot_report_top)),:])
    
    # Get required parameter values from spd:
    data = _tm30_process_spd(spd, cri_type = cri_type, **kwargs)
    spd  = data['St']
    
    # Create all subplots:
    if plot_report_top == True:
        _plot_tm30_report_top(f_ax_top, source = source, manufacturer = manufacturer,
                   date = date, model = model)
    if plot_report_bottom == True:
        _plot_tm30_report_bottom(f_ax_bottom, spd, 
                          notes = notes, max_len_notes_line = max_len_notes_line)

    plot_tm30_spd(data, axh = f_ax_spd, font_size = font_size)
    plot_tm30_cvg(data, axh = f_ax_cvg, font_size = font_size)
    plot_tm30_Rfhj(data, axh = f_ax_fhj, y_offset = 2, font_size = font_size)
    plot_tm30_Rcshj(data, axh = f_ax_cshj, xlabel = False, y_offset = 0.06, font_size = font_size)
    plot_tm30_Rhshj(data, axh = f_ax_hshj, xlabel = False, y_offset = 0.06, font_size = font_size)
    plot_tm30_Rfi(data, axh = f_ax_fi, font_size = font_size)
    fig.suptitle(suptitle, fontsize = 14, fontweight= 'bold')
    
    # Save to file:
    if save_fig_name is not None:
        fig.savefig(save_fig_name, dpi = dpi)
    
    axs = {'fig': fig, 'top' : f_ax_top, 'bottom': f_ax_bottom, 'cvg':f_ax_cvg,
           'rfi': f_ax_fi, 'rfhj':f_ax_fhj, 'rcshj':f_ax_cshj, 'rhshj':f_ax_hshj}
    
    
    return axs, data
        
spd_to_tm30_report = plot_tm30_report





#==================================================================================
def plot_cri_graphics(data, cri_type = None, hbins = 16, start_hue = 0.0, scalef = 100, \
                      plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, \
                      plot_center_lines = False, plot_bin_colors = True, \
                      axtype = 'polar', ax = None, force_CVG_layout = True,
                      vf_model_type = _VF_MODEL_TYPE, vf_pcolorshift = _VF_PCOLORSHIFT, vf_color = 'k', \
                      vf_bin_labels = _VF_PCOLORSHIFT['labels'], vf_plot_bin_colors = True, \
                      scale_vf_chroma_to_sample_chroma = False,\
                      plot_VF = True, plot_CF = False, plot_SF = False,
                      plot_test_sample_coord = False):
    """
    Plot graphical information on color rendition properties (custom design).
    
    Args:
        :data: 
            | ndarray with spectral data or dict with pre-computed metrics.
        :cri_type:
            | None, optional
            | If None: defaults to cri_type = 'iesrf'.
            | :hbins:, :start_hue: and :scalef: are ignored if cri_type not None 
            | and values are replaced by those in cri_type['rg_pars']
        :hbins:
            | 16 or ndarray with sorted hue bin centers (°), optional
        :start_hue: 
            | 0.0, optional
        :scalef:
            | 100, optional
            | Scale factor for graphic.
        :plot_axis_labels:
            | False, optional
            | Turns axis ticks on/off (True/False).
        :bin_labels: 
            | None or list[str] or '#', optional
            | Plots labels at the bin center hues.
            |   - None: don't plot.
            |   - list[str]: list with str for each bin. 
            |                (len(:bin_labels:) = :nhbins:)
            |   - '#': plots number.
        :plot_edge_lines:
            | True or False, optional
            | Plot grey bin edge lines with '--'.
        :plot_center_lines:
            | False or True, optional
            | Plot colored lines at 'center' of hue bin.
        :plot_bin_colors: 
            | True, optional
            | Colorize hue bins.
        :axtype: 
            | 'polar' or 'cart', optional
            | Make polar or Cartesian plot.
        :ax: 
            | None or 'new' or 'same', optional
            |   - None or 'new' creates new plot
            |   - 'same': continue plot on same axes.
            |   - axes handle: plot on specified axes.
        :force_CVG_layout: 
            | True, optional
            | True: Force plot of basis of CVG.
        :vf_model_type: 
            | _VF_MODEL_TYPE or 'M6' or 'M5', optional
            | Type of polynomial vector field model to use for the calculation of
              base color shift and metameric uncertainty.
        :vf_pcolorshift:
            | _VF_PCOLORSHIFT or user defined dict, optional
            | The polynomial models of degree 5 and 6 can be fully specified or 
            | summarized by the model parameters themselved OR by calculating the
            | dCoverC and dH at resp. 5 and 6 hues. :VF_pcolorshift: specifies 
            | these hues and chroma level.
        :vf_color:
            | 'k', optional
            | For plotting the vector fields.
        :vf_plot_bin_colors: 
            | True, optional
            | Colorize hue bins of VF graph.
        :scale_vf_chroma_to_sample_chroma:
            | False, optional
            | Scale chroma of reference and test vf fields such that average of 
            | binned reference chroma equals that of the binned sample chroma
            | before calculating hue bin metrics.
        :vf_bin_labels:
            | see :bin_labels:
            | Set VF model hue-bin labels.
        :plot_CF:
            | False, optional
            | Plot circle fields.
        :plot_VF:
            | True, optional
            | Plot vector fields.
        :plot_SF:
            | True, optional
            | Plot sample shifts.  
        :plot_test_sample_coord:
            | Plot the coordinates of the samples under the test illuminant
            | relative to the mean chromaticity under the reference illuminant (in the CVG plot).
            
    Returns:
        :returns: 
            | (data, 
            | [plt.gcf(),ax_spd, ax_CVG, ax_locC, ax_locH, ax_VF], 
            | cmap )
            | 
            | :data: is a dictionary with color rendering data
            | with keys:
            | - 'St, Sr'  : ndarray of test SPDs and corresponding ref. illuminants.
            | - 'xyz_cct': xyz of white point calculate with cieobs defined for cct calculations in cri_type['cieobs']
            | - 'cct, duv': CCT and Duv obtained with cieobs in cri_type['cieobs']['cct']
            | - 'xyzti, xyzri': ndarray tristimulus values of test and ref. samples (obtained with with cieobs in cri_type['cieobs']['xyz'])
            | - 'xyztw, xyzrw': ndarray tristimulus values of test and ref. white points (obtained with with cieobs in cri_type['cieobs']['xyz'])
            | - 'DEi, DEa': ndarray with individual sample color differences DEi and average DEa between test and ref.       
            | - 'Rf'  : ndarray with general color fidelity index values
            | - 'Rg'  : ndarray with color gamut area index values
            | - 'Rfi'  : ndarray with specific (sample) color fidelity indices
            | - 'Rfhj' : ndarray with local (hue binned) fidelity indices
            | - 'DEhj' : ndarray with local (hue binned) color differences
            | - 'Rcshj': ndarray with local chroma shifts indices
            | - 'Rhshj': ndarray with local hue shifts indices
            | - 'hue_bin_data': dict with output from _get_hue_bin_data() [see its help for more info]
            | - 'cri_type': same as input (for reference purposes)
            | - 'vf' : dictionary with vector field measures and data.
            |         Keys:
            |           - 'Rt'  : ndarray with general metameric uncertainty index Rt
            |           - 'Rti' : ndarray with specific metameric uncertainty indices Rti
            |           - 'Rfhj' : ndarray with local (hue binned) fidelity indices 
            |                            obtained from VF model predictions at color space
            |                            pixel coordinates
            |           - 'DEhj' : ndarray with local (hue binned) color differences
            |                           (same as above)
            |           - 'Rcshj': ndarray with local chroma shifts indices for vectorfield coordinates
            |                           (same as above)
            |           - 'Rhshj': ndarray with local hue shifts indicesfor vectorfield coordinates
            |                           (same as above)
            |           - 'Rfi': ndarray with sample fidelity indices for vectorfield coordinates
            |                           (same as above)
            |           - 'DEi': ndarray with sample color differences for vectorfield coordinates
            |                           (same as above)
            |           - 'hue_bin_data': dict with output from _get_hue_bin_data() for vectorfield coordinates
            |           - 'dataVF': dictionary with output of cri.VFPX.VF_colorshift_model()
            |    
            | :[...]: list with handles to current figure and 5 axes.
            |
            | :cmap: list with rgb colors for hue bins 
                    (for use in other plotting fcns)
        
    """
    if not isinstance(data,dict):
        data = spd_to_ies_tm30_metrics(data, 
                                       cri_type = cri_type, 
                                       hbins = hbins, 
                                       start_hue = start_hue, 
                                       scalef = scalef, 
                                       vf_model_type = vf_model_type, 
                                       vf_pcolorshift = vf_pcolorshift, 
                                       scale_vf_chroma_to_sample_chroma = scale_vf_chroma_to_sample_chroma)

    # Unpack data dictionary:
    (DEa, DEi, Rcshj, Rf,
     Rfhj, Rfi, Rg, Rhshj, 
     Sr, St, cct, cri_type, 
     duv, hue_bin_data, vf,
     xyzri, xyzrw, xyzti, 
     xyztw, xyztw_cct) = [data[x] for x in sorted(data.keys())]
    
    
    # Get some variables from hue_bin_data:
    nhbins = hue_bin_data['nhbins']
    start_hue = hue_bin_data['start_hue']
    scalef = hue_bin_data['normalized_chroma_ref']
    hbinnr = hue_bin_data['hbinnrs']
    

    jabti = hue_bin_data['jabtn']
    jabri = hue_bin_data['jabrn']

    
    #layout = np.array([[3,3,0,0],[1,0,2,2],[0,0,2,1],[2,2,1,1],[0,2,1,1],[1,2,1,1]])
    #layout = np.array([[6,6,0,0],[0,3,3,3],[3,3,3,3],[0,0,3,2],[2,2,2,2],[2,0,2,2],[4,0,2,2]])
    layout = np.array([[6,7,0,0],[0,4,3,3],[3,4,3,3],[0,0,4,2],[2,0,2,2],[4,2,2,2],[4,0,2,2],[2,2,2,2]])
    
    def create_subplot(layout,n, polar = False, frameon = True):
        ax = plt.subplot2grid(layout[0,0:2], layout[n,0:2], 
                              colspan = layout[n,2], 
                              rowspan = layout[n,3], 
                              polar = polar, 
                              frameon = frameon)
        return ax
               
        
    for i in range(cct.shape[0]):
 
        
        fig = plt.figure(figsize=(10, 6), dpi=144)
    
        # Plot CVG:
        ax_CVG = create_subplot(layout,1, polar = True, frameon = False)
        if plot_test_sample_coord == False:  
            jabti = None
            jabri = None
        else:
            jabti_i = jabti[...,i:i+1,:]
            jabri_i = jabri[...,i:i+1,:]

        figCVG, ax, cmap = plot_ColorVectorGraphic(data['hue_bin_data']['jabtn_hj_closed'][...,i,:], 
                                                   data['hue_bin_data']['jabrn_hj_closed'][...,i,:], 
                                                   hbins = nhbins, 
                                                   axtype = axtype, 
                                                   ax = ax_CVG, 
                                                   plot_center_lines = plot_center_lines, 
                                                   plot_edge_lines = plot_edge_lines,  
                                                   plot_bin_colors = plot_bin_colors, 
                                                   scalef = scalef, 
                                                   force_CVG_layout = force_CVG_layout, 
                                                   bin_labels = '#',
                                                   jabti = jabti, 
                                                   jabri = jabri, 
                                                   hbinnr = hbinnr)
                
        # Plot VF:
        ax_VF = create_subplot(layout,2, polar = True, frameon = False)
        if i == 0:
            hbin_cmap = None
    
        ax_VF, hbin_cmap = plot_VF_PX_models([vf['dataVF'][i]], 
                                             dataPX = None, 
                                             plot_VF = plot_VF, 
                                             plot_PX = None, 
                                             axtype = 'polar', 
                                             ax = ax_VF,
                                             plot_circle_field = plot_CF, 
                                             plot_sample_shifts = plot_SF, 
                                             plot_bin_colors = vf_plot_bin_colors,
                                             plot_samples_shifts_at_pixel_center = False, 
                                             jabp_sampled = None,
                                             plot_VF_colors = [vf_color], 
                                             plot_PX_colors = ['r'], 
                                             hbin_cmap = hbin_cmap, 
                                             force_CVG_layout = True, 
                                             bin_labels = vf_bin_labels)
    
        # Plot test SPD:
        ax_spd = create_subplot(layout,3)
        ax_spd.plot(St[0],St[i+1]/St[i+1].max(),'r-')
        ax_spd.text(730,0.9,'CCT = {:1.0f} K'.format(cct[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.8,'Duv = {:1.4f}'.format(duv[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.7,'Rf = {:1.0f}'.format(Rf[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.6,'Rg = {:1.0f}'.format(Rg[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.5,'Rt = {:1.0f}'.format(vf['Rt'][:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.set_xlabel('Wavelength (nm)', fontsize = 9)
        ax_spd.set_ylabel('Rel. spectral intensity', fontsize = 9)
        ax_spd.set_xlim([360,830])
        
        # Plot local color fidelity, Rfhi:
        ax_Rfi = create_subplot(layout,4)
        for j in range(nhbins):
            ax_Rfi.bar(range(nhbins)[j],Rfhj[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_Rfi.text(range(nhbins)[j],Rfhj[j,i]*1.1, '{:1.0f}'.format(Rfhj[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
        ax_Rfi.set_ylim([0,120])
        xticks = np.arange(nhbins)
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in range(nhbins)]
        ax_Rfi.set_xticks(xticks)
        ax_Rfi.set_xticklabels(xtickslabels, fontsize = 8)
        ax_Rfi.set_ylabel(r'Local color fidelity $R_{f,hi}$')
        ax_Rfi.set_xlabel('Hue bin #')
        
        # Plot local chroma shift, Rcshi:
        ax_locC = create_subplot(layout,5)
        for j in range(nhbins):
            ax_locC.bar(range(nhbins)[j],Rcshj[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locC.text(range(nhbins)[j],-np.sign(Rcshj[j,i])*0.1, '{:1.0f}%'.format(100*Rcshj[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ylim = np.array([np.abs(Rcshj.min()),np.abs(Rcshj.min()),0.2]).max()*1.5
        ax_locC.set_ylim([-ylim,ylim])
        ax_locC.set_ylabel(r'Local chroma shift, $R_{cs,hi}$')
        #ax_locC.set_xticks(ax_locC.get_xticks().tolist()) # to avoid warning from matplotlib
        ax_locC.set_xticklabels([])
        ax_locC.set_yticks(ax_locC.get_yticks().tolist()) # to avoid warning from matplotlib
        ax_locC.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locC.get_yticks().tolist()], color = 'white')
        ax_locC.set_ylim([-ylim,ylim])
        
        # Plot local hue shift, Rhshi:
        ax_locH = create_subplot(layout,6)
        for j in range(nhbins):
            ax_locH.bar(range(nhbins)[j],Rhshj[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locH.text(range(nhbins)[j],-np.sign(Rhshj[j,i])*0.2, '{:1.3f}'.format(Rhshj[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ylim = np.array([np.abs(Rhshj.min()),np.abs(Rhshj.min()),0.2]).max()*1.5
        ax_locH.set_ylim([-ylim,ylim])
        ax_locH.set_ylabel(r'Local hue shift, $R_{hs,hi}$')
        #ax_locH.set_xticks(ax_locH.get_xticks().tolist()) # to avoid warning from matplotlib
        ax_locH.set_xticklabels([])
        ax_locH.set_yticks(ax_locH.get_yticks().tolist()) # to avoid warning from matplotlib
        ax_locH.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locH.get_yticks().tolist()], color = 'white')
        ax_locH.set_ylim([-ylim,ylim])  
        
        # Plot local color fidelity of VF, vfRfhi:
        ax_vfRfi = create_subplot(layout,7)
        for j in range(nhbins):
            ax_vfRfi.bar(range(nhbins)[j],vf['Rfhj'][j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_vfRfi.text(range(nhbins)[j],vf['Rfhj'][j,i]*1.1, '{:1.0f}'.format(vf['Rfhj'][j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
        ax_vfRfi.set_ylim([0,120])
        xticks = np.arange(hbins)
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in range(nhbins)]
        ax_vfRfi.set_xticks(xticks)
        ax_vfRfi.set_xticklabels(xtickslabels, fontsize = 8)
        ax_vfRfi.set_ylabel(r'Local VF color fidelity $vfR_{f,hi}$')
        ax_vfRfi.set_xlabel('Hue bin #')
       
        plt.tight_layout()
        
    return  data,  [plt.gcf(),ax_spd, ax_CVG, ax_locC, ax_locH, ax_VF], cmap

    
if __name__ == '__main__':
    import luxpy as lx
    spd = lx._CIE_F4

    # illustration of ANSI/IES TM30 plots:
    plot_tm30_cvg(spd, axtype = 'cart', plot_vectors = True, gamut_line_color = 'r')
    plot_tm30_spd(spd)
    plot_tm30_Rfi(spd)
    plot_tm30_Rfhj(spd)
    plot_tm30_Rcshj(spd)
    plot_tm30_Rhshj(spd)
    plot_tm30_Rxhj(spd)
    plot_tm30_report(spd, source = 'test', font_size = 12,notes = 'This is a test if the note splitting actually works or not.',save_fig_name = 'testfig.png')
    
    
    # Illustration of custom plotter:
    data2 = plot_cri_graphics(spd, 
                             cri_type = 'iesrf', 
                             plot_VF = True,             # cartesian vector fields
                             plot_CF = True,             # circular vector fields
                             plot_SF = False,            # sample shifts
                             plot_bin_colors = True,     # bin colors for CVG
                             vf_plot_bin_colors = False, # bin colors for vector field graph 
                             axtype = 'polar',           # use polar plots
                             ax = None,                  # axes to plot figure in (None: create new)
                             plot_center_lines = False,  # plot lines at center of hue bin
                             plot_edge_lines = True,     # plot edges of hue bins
                             plot_test_sample_coord = False) # plot normalized test sample coordinates (cfr. CIE224:2017)
                
    plt.show()