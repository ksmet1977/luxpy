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
Module 2 for IES color rendition graphical output
=================================================

 :_tm30_process_spd(): Calculate all required parameters for plotting from spd using cri.spd_to_cri()

 :plot_tm30_cvg(): Plot TM30 Color Vector Graphic (CVG).
 
 :plot_tm30_Rfi(): Plot Sample Color Fidelity values (Rfi).
 
 :plot_tm30_Rxhj(): Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj).

 :plot_tm30_Rcshj(): Plot Local Chroma Shifts (Rcshj).

 :plot_tm30_Rhshj(): Plot Local Hue Shifts (Rhshj).

 :plot_tm30_Rfhj(): Plot Local Color Fidelity values (Rfhj).

 :plot_tm30_spd(): Plot test SPD and reference illuminant, both normalized to the same luminous power.

 :plot_tm30_report():

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from luxpy import (_CIE_D65, math, cat, xyz_to_srgb, spd_to_power, 
                   spd_normalize, spd_to_xyz, xyz_to_Yxy, xyz_to_Yuv)
from luxpy.color.cri.utils.helpers import spd_to_cri
from luxpy.color.cri.utils.graphics import plot_ColorVectorGraphic

_TM30_FONT_SIZE = 8

__all__ = ['_tm30_process_spd','plot_tm30_cvg','plot_tm30_Rfi',
           'plot_tm30_Rxhj','plot_tm30_Rcshj', 'plot_tm30_Rhshj', 
           'plot_tm30_Rfhj', 'plot_tm30_spd','plot_tm30_report']

def _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs):
    """
    Calculate all required parameters for plotting from spd using cri.spd_to_cri()
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
    out = 'Rf,Rg,cct,duv,Sr,cri_type,xyzri,xyzrw,binnrs,Rfi,Rfhi,Rcshi,Rhshi,jabt_binned,jabr_binned,nhbins,start_hue,normalize_gamut,normalized_chroma_ref'
    if not isinstance(spd,dict):
        tpl = spd_to_cri(spd, cri_type = cri_type, out = out, **kwargs)
        data = {'spd':spd}
        for i,key in enumerate(out.split(',')):
            if key == 'normalized_chroma_ref': key = 'scalef' # rename
            if key == 'binnrs': key = 'hbinnrs' # rename
            data[key] = tpl[i]
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
                  plot_vectors = True,
                  axh = None, axtype = 'cart',
                  **kwargs):
    """
    Plot TM30 Color Vector Graphic (CVG).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
            | Plotting color for the line connecting the 
            | average test chromaticity in the hue bins.
        :plot_vectors:
            | True, optional
            | Plot color shift vectors in CVG (True) or not (False).
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
        :data:
            | dictionary with required parameters for plotting functions.      
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    # Normalize chroma to scalef:
    scalef = data['scalef']
    jabt = data['jabt_binned'][:,0,:]
    jabr = data['jabr_binned'][:,0,:]
    Cr = (jabr[...,1]**2 + jabr[...,2]**2)**0.5
    Ct = ((jabt[...,1]**2 + jabt[...,2]**2)**0.5)/Cr*scalef
    ht = math.positive_arctan(jabt[...,1],jabt[...,2], htype = 'rad')
    hr = math.positive_arctan(jabr[...,1],jabr[...,2], htype = 'rad')
    jabt[...,1] = Ct*np.cos(ht)
    jabt[...,2] = Ct*np.sin(ht)
    jabr[...,1] = scalef*np.cos(hr)
    jabr[...,2] = scalef*np.sin(hr)
    
    # Plot color vector graphic
    _, axh, _ = plot_ColorVectorGraphic(jabt = jabt, jabr = jabr, 
                                        hbins = data['nhbins'], start_hue = data['start_hue'], 
                                        gamut_line_color = gamut_line_color,
                                        plot_vectors = plot_vectors,
                                        ax = axh, axtype = axtype,
                                        force_CVG_layout = True,
                                        plot_axis_labels = False)
    
    # Print Rf, Rg, CCT and Duv in plot:
    Rf, Rg, cct, duv = data['Rf'], data['Rg'], data['cct'], data['duv']
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
    return axh


def plot_tm30_spd(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    """
    Plot test SPD and reference illuminant, both normalized to the same luminous power.
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    # Normalize Sr to same luminous power as spd:
    Phiv_spd = spd_to_power(data['spd'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    #Phiv_Sr = spd_to_power(data['Sr'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    data['Sr'] = spd_normalize(data['Sr'], norm_type = 'pu', norm_f = Phiv_spd, cieobs = data['cri_type']['cieobs']['cct'])
    
    # Plot test and ref SPDs:
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    axh.plot(data['Sr'][0,:], data['Sr'][1,:],'k-', label = 'Reference')
    axh.plot(data['spd'][0,:], data['spd'][1,:],'r-', label = 'Test')
    axh.set_xlabel('Wavelength (nm)', fontsize = _TM30_FONT_SIZE)
    axh.set_ylabel('Radiant power\n(Equal Luminous Flux)', fontsize = _TM30_FONT_SIZE)
    axh.set_xlim([360,830]) 
    axh.set_yticklabels([])
    axh.legend(loc = 'upper right', fontsize = _TM30_FONT_SIZE)
    
    return axh


def plot_tm30_Rfi(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    """
    Plot Sample Color Fidelity values (Rfi).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
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
    axh.set_xticklabels(xtickslabels, fontsize = _TM30_FONT_SIZE, rotation = 90)
    axh.set_ylabel(r'Color Sample Fidelity $(R_{f,CESi})$', fontsize = _TM30_FONT_SIZE)
    axh.set_ylim([0,100])
    axh.set_xlim([-0.5,N-0.5])
    
    return axh

def plot_tm30_Rfhj(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    """
    Plot Local Color Fidelity values (Rfhj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
    """
    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rfhi = data['Rfhi']
        
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        hbinnrs = data['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])

    # Plot local color fidelity, Rfhi:
    hbins = range(data['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],Rfhi[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = ((np.abs(Rfhi[j,0]) + 2 + y_offset))*np.sign(Rfhi[j,0])
        axh.text(hbins[j],ypos, '{:1.0f}'.format(Rfhi[j,0]) ,fontsize = _TM30_FONT_SIZE,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = _TM30_FONT_SIZE)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = _TM30_FONT_SIZE)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Color Fidelity $(R_{f,hj})$', fontsize = _TM30_FONT_SIZE)
    axh.set_ylim([0,110])

    return axh

def plot_tm30_Rcshj(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    """
    Plot Local Chroma Shift values (Rcshj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
    """

    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rcshi = data['Rcshi']
    
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        hbinnrs = data['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])
    
    # Plot local chroma shift, Rcshi:
    hbins = range(data['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],100*Rcshi[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = 100*((np.abs(Rcshi[j,0]) + 0.05 + y_offset))*np.sign(Rcshi[j,0])
        axh.text(hbins[j],ypos, '{:1.0f}%'.format(100*Rcshi[j,0]), fontsize = _TM30_FONT_SIZE,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = _TM30_FONT_SIZE)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = _TM30_FONT_SIZE)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    yticks = range(-40,50,10)
    axh.set_yticks(yticks)
    ytickslabels = ['{:1.0f}%'.format(ii) for ii in range(-40,50,10)]
    axh.set_yticklabels(ytickslabels, fontsize = _TM30_FONT_SIZE)
    axh.set_ylabel(r'Local Chroma Shift $(R_{cs,hj})$', fontsize = _TM30_FONT_SIZE)
    axh.set_ylim([min([-40,100*Rcshi.min()]),max([40,100*Rcshi.max()])])
    
    return axh

def plot_tm30_Rhshj(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    """
    Plot Local Hue Shift values (Rhshj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
    """

    
    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rhshi = data['Rhshi']

    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        hbinnrs = data['hbinnrs'], 
                        xyzri = data['xyzri'], 
                        xyzrw = data['xyzrw'], 
                        cri_type = data['cri_type'])
    
    # Plot local hue shift, Rhshi:
    hbins = range(data['nhbins'])
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    for j in hbins:
        axh.bar(hbins[j],Rhshi[j,0], color = cmap[j], width = 1,edgecolor = 'k', alpha = 1)
        ypos = ((np.abs(Rhshi[j,0]) + 0.05 + y_offset))*np.sign(Rhshi[j,0])
        axh.text(hbins[j],ypos, '{:1.2f}'.format(Rhshi[j,0]) ,fontsize = _TM30_FONT_SIZE,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = _TM30_FONT_SIZE)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = _TM30_FONT_SIZE)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Hue Shift $(R_{hs,hj})$', fontsize = 9)
    axh.set_ylim([min([-0.5,Rhshi.min()]),max([0.5,Rhshi.max()])])
    
    return axh

def plot_tm30_Rxhj(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    """
    Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj) (one for each hue-bin).
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
        :kwargs:
            | Additional optional keyword arguments, 
            | the same as in cri.spd_to_cri()
            
    Returns:
        :axh:
            | handle to figure axes.      
    """

    data = _tm30_process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    if axh is None:
        fig, axh = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (6,18))
    
    plot_tm30_Rcshj(data, axh = axh[0], xlabel = False, y_offset = 0.02)
    plot_tm30_Rhshj(data, axh = axh[1], xlabel = False, y_offset = 0.03)
    plot_tm30_Rfhj(data, axh = axh[2], xlabel = True, y_offset = 2)
    return axh

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
    l = len(notes)
    n = l//max_len_notes_line + 1
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
    axh.set_yticklabels(['' for i in np.arange(4)])
    axh.set_axis_off()
    axh.set_xlabel([])
    
    axh.text(0,1, 'Source: ' + source, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(0,0, '   Date: ' + date, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(5,1, 'Manufacturer: ' + manufacturer, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')
    axh.text(5,0, 'Model: ' + model, fontsize = 10, horizontalalignment='left',verticalalignment='center',color = 'k')

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
    ciera = spd_to_cri(spd, cri_type = 'ciera')
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
                     save_fig_name = None, dpi = 300,
                     plot_report_top = True, plot_report_bottom = True,
                     suptitle = 'ANSI/IES TM-30-18 Color Rendition Report',
                     **kwargs):
    """
    Create TM30 Color Rendition Report.
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'hbinnrs','Rfi','Rfhi','Rcshi','Rhshi',
            |   'jabt_binned','jabr_binned',
            |   'nhbins','start_hue','normalize_gamut','normalized_chroma_ref'
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
    fig = plt.figure(constrained_layout=True, figsize = (7,12))  
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
    data = _tm30_process_spd(spd, cri_type = cri_type,**kwargs)
    
    # Create all subplots:
    if plot_report_top == True:
        _plot_tm30_report_top(f_ax_top, source = source, manufacturer = manufacturer,
                   date = date, model = model)
    if plot_report_bottom == True:
        _plot_tm30_report_bottom(f_ax_bottom, spd, 
                          notes = notes, max_len_notes_line = max_len_notes_line)

    plot_tm30_spd(data, axh = f_ax_spd)
    plot_tm30_cvg(data, axh = f_ax_cvg)
    plot_tm30_Rfhj(data, axh = f_ax_fhj, y_offset = 2)
    plot_tm30_Rcshj(data, axh = f_ax_cshj, xlabel = False, y_offset = 0.03)
    plot_tm30_Rhshj(data, axh = f_ax_hshj, xlabel = False, y_offset = 0.05)
    plot_tm30_Rfi(data, axh = f_ax_fi)
    fig.suptitle(suptitle, fontsize = 14, fontweight= 'bold')
    
    # Save to file:
    if save_fig_name is not None:
        fig.savefig(save_fig_name, dpi = dpi)
    
    axs = {'fig': fig, 'top' : f_ax_top, 'bottom': f_ax_bottom, 'cvg':f_ax_cvg,
           'rfi': f_ax_fi, 'rfhj':f_ax_fhj, 'rcshj':f_ax_cshj, 'rhshj':f_ax_hshj}
    
    
    return axs, data
        
    
if __name__ == '__main__':
    import luxpy as lx
    spd = lx._CIE_F4
    data, _, _ = lx.cri.plot_cri_graphics(spd, plot_bin_colors = False, \
                                vf_plot_bin_colors = False,\
                                vf_color = 'grey', plot_CF = True, 
                                plot_test_sample_coord = True,
                                hbins = 8)
    plot_tm30_cvg(spd, axtype = 'cart', plot_vectors = True, gamut_line_color = 'r')
    # plot_tm30_spd(spd)
    # plot_tm30_Rfi(spd)
    # plot_tm30_Rfhj(spd)
    # plot_tm30_Rcshj(spd)
    # plot_tm30_Rhshj(spd)
    # plot_tm30_Rxhj(spd)
    # plot_tm30_report(spd, source = 'test', notes = 'This is a test if the note splitting actually works or not.',save_fig_name = 'testfig.png')
    