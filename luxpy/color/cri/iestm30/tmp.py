# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:56:20 2020

@author: u0032318
"""
import os
import imageio
import colorsys
from luxpy.utils import _PKG_PATH, np, plt
from luxpy import _CIE_D65, math, cat, xyz_to_srgb, spd_to_xyz, xyz_to_Yxy, xyz_to_Yuv
from luxpy.color.cri.utils.helpers import spd_to_cri
from luxpy.color.cri.utils.graphics import plot_hue_bins, plot_ColorVectorGraphic
from luxpy import spd_to_power, spd_normalize
_CVG_BG = imageio.imread(os.path.join(_PKG_PATH, 'color','cri','iestm30','cvg_background.jfif'))


def _process_spd(spd, cri_type = 'ies-tm30',**kwargs):
    """
    Calculate all required parameters for plotting from spd using cri.spd_to_cri()
    
    Args:
        :spd:
            | ndarray or dict
            | If ndarray: single spectral power distribution.
            | If dict: dictionary with pre-computed parameters.
            |  required keys:
            |   'Rf','Rg','cct','duv','Sr','cri_type','xyzri','xyzrw',
            |   'binnrs','Rfi','Rfhi','Rcshi','Rhshi',
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
            if key == 'normalized_chroma_ref': key = 'scalef'
            data[key] = tpl[i]
    else:
        data = spd
    return data

def _get_hue_map(hbins = 16, start_hue = 0.0, 
                 binnrs = None, xyzri = None, xyzrw = None, cri_type = None):
    
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
    if (binnrs is not None) & (xyzri is not None) & (xyzrw is not None) & (cri_type is not None):
        xyzw = spd_to_xyz(_CIE_D65, relative = True, cieobs = cri_type['cieobs']['xyz'])
        xyzri = cat.apply(xyzri[:,0,:],xyzw1 = xyzrw, xyzw2 = xyzw)
                
        # Create color from xyz average:
        for i in range(nhbins):
            xyzrhi = xyzri[binnrs[:,0] == i,:].mean(axis=0,keepdims=True)
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
    # (Rf,Rg,cct,duv,Sr,cri_type,xyzri,xyzrw,binnrs,Rfi,Rfhi,Rcshi,Rhshi,
    #  jabt_binned,jabr_binned,
    #  nhbins,start_hue,
    #  normalize_gamut,scalef) = spd_to_cri(spd, 
    #                                       cri_type = cri_type, 
    #                                       out = 'Rf,Rg,cct,duv,Sr,cri_type,xyzri,xyzrw,binnrs,Rfi,Rfhi,Rcshi,Rhshi,jabt_binned,jabr_binned,nhbins,start_hue,normalize_gamut,normalized_chroma_ref', **kwargs)
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
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
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    # Normalize Sr to same luminous power as spd:
    Phiv_spd = spd_to_power(data['spd'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    #Phiv_Sr = spd_to_power(data['Sr'], ptype = 'pu', cieobs = data['cri_type']['cieobs']['cct'])
    data['Sr'] = spd_normalize(data['Sr'], norm_type = 'pu', norm_f = Phiv_spd, cieobs = data['cri_type']['cieobs']['cct'])
    
    # Plot test and ref SPDs:
    if axh is None:
        fig, axh = plt.subplots(nrows = 1, ncols = 1)
    axh.plot(data['Sr'][0,:], data['Sr'][1,:],'k-', label = 'Reference')
    axh.plot(data['spd'][0,:], data['spd'][1,:],'r-', label = 'Test')
    axh.set_xlabel('Wavelength (nm)', fontsize = 8)
    axh.set_ylabel('Radiant power\n(Equal Luminous Flux)', fontsize = 8)
    axh.set_xlim([360,830]) 
    axh.set_yticklabels([])
    axh.legend(loc = 'upper right', fontsize = 8)
    
    return axh


def plot_tm30_Rfi(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
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
    axh.set_xticklabels(xtickslabels, fontsize = 8, rotation = 90)
    axh.set_ylabel(r'Color Sample Fidelity $(R_{f,CESi})$', fontsize = 8)
    axh.set_ylim([0,100])
    axh.set_xlim([-0.5,N-0.5])
    
    return axh

def plot_tm30_Rfhi(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rfhi = data['Rfhi']
        
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        binnrs = data['binnrs'], 
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
        axh.text(hbins[j],ypos, '{:1.0f}'.format(Rfhi[j,0]) ,fontsize = 8,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = 8)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Color Fidelity $(R_{f,hj})$', fontsize = 8)
    axh.set_ylim([0,110])

    return axh

def plot_tm30_Rcshi(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rcshi = data['Rcshi']
    
    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        binnrs = data['binnrs'], 
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
        axh.text(hbins[j],ypos, '{:1.0f}%'.format(100*Rcshi[j,0]), fontsize = 8,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = 8)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    yticks = range(-40,50,10)
    axh.set_yticks(yticks)
    ytickslabels = ['{:1.0f}%'.format(ii) for ii in range(-40,50,10)]
    axh.set_yticklabels(ytickslabels, fontsize = 8)
    axh.set_ylabel(r'Local Chroma Shift $(R_{cs,hj})$', fontsize = 8)
    axh.set_ylim([min([-40,100*Rcshi.min()]),max([40,100*Rcshi.max()])])
    
    return axh

def plot_tm30_Rhshi(spd, cri_type = 'ies-tm30', axh = None, xlabel = True, y_offset = 0, **kwargs):
    
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    Rhshi = data['Rhshi']

    # Get color map based on sample colors:
    cmap = _get_hue_map(hbins = data['nhbins'], start_hue = data['start_hue'], 
                        binnrs = data['binnrs'], 
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
        axh.text(hbins[j],ypos, '{:1.2f}'.format(Rhshi[j,0]) ,fontsize = 8,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    
    xticks = np.array(hbins)
    axh.set_xticks(xticks)
    if xlabel == True:
        xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
        axh.set_xlabel('Hue-Angle Bin (j)', fontsize = 8)
    else:
        xtickslabels = [''.format(ii+1) for ii in hbins]
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    axh.set_ylabel(r'Local Hue Shift $(R_{hs,hj})$', fontsize = 9)
    axh.set_ylim([min([-0.5,Rhshi.min()]),max([0.5,Rhshi.max()])])
    
    return axh

def plot_tm30_Rxhi(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    if axh is None:
        fig, axh = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (6,18))
    
    plot_tm30_Rcshi(data, axh = axh[0], xlabel = False, y_offset = 0.02)
    plot_tm30_Rhshi(data, axh = axh[1], xlabel = False, y_offset = 0.03)
    plot_tm30_Rfhi(data, axh = axh[2], xlabel = True, y_offset = 2)
    return axh

def _split_notes(notes, max_len_notes_line = 40):
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

def plot_tm30_text(axh, source = '', manufacturer = '', date = '', model = ''):
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

def plot_tm30_notes_ciera(axh, spd, notes = '', max_len_notes_line = 40):
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
    
    axh.text(0,3, 'Notes: ', fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(0.75,3,  notes_, fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,3, "x   {:1.4f}".format(Yxyw[0,1]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,2.4, "y   {:1.4f}".format(Yxyw[0,2]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,1.8, "u'  {:1.4f}".format(Yuvw[0,1]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(6,1.2, "v'  {:1.4f}".format(Yuvw[0,2]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,3, "CIE 13.3-1995", fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,2.4, "     (CRI)    ", fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,1.8, "    $R_a$  {:1.0f}".format(ciera[0,0]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')
    axh.text(7.5,1.2, "    $R_9$  {:1.0f}".format(cierai[9,0]), fontsize = 9, horizontalalignment='left',verticalalignment='top',color = 'k')

    

def plot_tm30_report(spd, cri_type = 'ies-tm30',
                     source = '', manufacturer = '',
                     date = '', model = '', 
                     notes = '', max_len_notes_line = 40,
                     save_fig_name = None, dpi = 300,
                     **kwargs):
    title = 'ANSI/IES TM-30-18 Color Rendition Report'
    fig = plt.figure(constrained_layout=True, figsize = (7,12))  
    
    gs = fig.add_gridspec(6, 3,height_ratios=[0.1,0.5,0.5,0.5,0.6,0.3], width_ratios=[1,1,1.5])
    f_ax_top = fig.add_subplot(gs[0, :])
    f_ax_bottom = fig.add_subplot(gs[-1, :])
    
    f_ax_spd = fig.add_subplot(gs[1, 0:2])
    f_ax_cvg = fig.add_subplot(gs[2:4, 0:2])
    f_ax_cshj = fig.add_subplot(gs[1, 2:])
    f_ax_hshj = fig.add_subplot(gs[2, 2:])
    f_ax_fhj = fig.add_subplot(gs[3, 2:])
    f_ax_fi = fig.add_subplot(gs[4,:])
    # fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    data = _process_spd(spd, cri_type = cri_type,**kwargs)
    
    
    plot_tm30_text(f_ax_top, source = source, manufacturer = manufacturer,
                   date = date, model = model)
    plot_tm30_notes_ciera(f_ax_bottom, spd, 
                          notes = notes, max_len_notes_line = max_len_notes_line)
    
    plot_tm30_spd(data, axh = f_ax_spd)
    plot_tm30_cvg(data, axh = f_ax_cvg)
    plot_tm30_Rfhi(data, axh = f_ax_fhj, y_offset = 2)
    plot_tm30_Rcshi(data, axh = f_ax_cshj, xlabel = False, y_offset = 0.03)
    plot_tm30_Rhshi(data, axh = f_ax_hshj, xlabel = False, y_offset = 0.05)
    plot_tm30_Rfi(data, axh = f_ax_fi)
    fig.suptitle(title, fontsize = 14, fontweight= 'bold')
    if save_fig_name is not None:
        fig.savefig(save_fig_name, dpi = dpi)
        
def plot_tm30_reportb(spd, cri_type = 'ies-tm30',
                     source = '', manufacturer = '',
                     date = '', model = '', notes = '',
                     save_fig_name = None, dpi = 300,
                     **kwargs):
    title = 'ANSI/IES TM-30-18 Color Rendition Report'
    fig = plt.figure(constrained_layout=True, figsize = (8,14))  
    
    gs = fig.add_gridspec(6, 2,height_ratios=[0.1,1,1,1,1,0.3], width_ratios=[1,1])
    f_ax_top = fig.add_subplot(gs[0, :])
    f_ax_bottom = fig.add_subplot(gs[-1, :])
    
    f_ax_spd = fig.add_subplot(gs[1, 0])
    f_ax_cvg = fig.add_subplot(gs[2:4, 0])
    f_ax_cshj = fig.add_subplot(gs[1, 1])
    f_ax_hshj = fig.add_subplot(gs[2, 1])
    f_ax_fhj = fig.add_subplot(gs[3, 1])
    f_ax_fi = fig.add_subplot(gs[4,:])
    # fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    
    plot_tm30_text(f_ax_top, source = source, manufacturer = manufacturer,
                   date = date, model = model)
    plot_tm30_notes_ciera(f_ax_bottom, spd, notes = notes)
    
    plot_tm30_spd(data, axh = f_ax_spd)
    plot_tm30_cvg(data, axh = f_ax_cvg)
    plot_tm30_Rfhi(data, axh = f_ax_fhj, y_offset = 2)
    plot_tm30_Rcshi(data, axh = f_ax_cshj, xlabel = False, y_offset = 0.03)
    plot_tm30_Rhshi(data, axh = f_ax_hshj, xlabel = False, y_offset = 0.05)
    plot_tm30_Rfi(data, axh = f_ax_fi)
    fig.suptitle(title, fontsize = 14, fontweight= 'bold')
    if save_fig_name is not None:
        fig.savefig(save_fig_name, dpi = dpi)
    
if __name__ == '__main__':
    import luxpy as lx
    spd = lx._CIE_F4
    plot_tm30_cvg(spd, axtype = 'cart', plot_vectors = True, gamut_line_color = 'r')
    plot_tm30_spd(spd)
    plot_tm30_Rfi(spd)
    plot_tm30_Rfhi(spd)
    plot_tm30_Rcshi(spd)
    plot_tm30_Rhshi(spd)
    plot_tm30_Rxhi(spd)
    plot_tm30_report(spd, source = 'test', notes = 'This is a test if the note splitting actually works or not.',save_fig_name = 'testfig.png')
    