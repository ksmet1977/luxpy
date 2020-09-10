# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:56:20 2020

@author: u0032318
"""
import os
import imageio
import colorsys
from luxpy.utils import _PKG_PATH, np, plt
from luxpy import _CIE_D65, math, cat, xyz_to_srgb, spd_to_xyz
from luxpy.color.cri.utils.helpers import spd_to_cri
from luxpy.color.cri.utils.graphics import plot_hue_bins, plot_ColorVectorGraphic
from luxpy import spd_to_power, spd_normalize
_CVG_BG = imageio.imread(os.path.join(_PKG_PATH, 'color','cri','iestm30','cvg_background.jfif'))


def _process_spd(spd, cri_type = 'ies-tm30',**kwargs):
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
                                        ax = axh, axtype = axtype)
    
    # Print Rf, Rg, CCT and Duv in plot:
    Rf, Rg, cct, duv = data['Rf'], data['Rg'], data['cct'], data['duv']
    axh.text(-1.30*scalef,1.30*scalef,'{:1.0f}'.format(Rf[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='center',verticalalignment='center',color = 'k')
    axh.text(-1.33*scalef,1.15*scalef,'$R_f$',fontsize = 13, style='italic', horizontalalignment='center',verticalalignment='center',color = 'k')
    axh.text(1.30*scalef,1.30*scalef,'{:1.0f}'.format(Rg[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='center',verticalalignment='center',color = 'k')
    axh.text(1.33*scalef,1.15*scalef,'$R_g$',fontsize = 13, style='italic', horizontalalignment='center',verticalalignment='center',color = 'k')
    axh.text(-1.43*scalef,-1.43*scalef,'{:1.0f}'.format(cct[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='left',verticalalignment='bottom',color = 'k')
    axh.text(-1.43*scalef,-1.26*scalef,'$CCT$',fontsize = 13, style='italic', horizontalalignment='left',verticalalignment='bottom',color = 'k')
    axh.text(1.43*scalef,-1.43*scalef,'{:1.4f}'.format(duv[0,0]),fontsize = 15, fontweight='bold', horizontalalignment='right',verticalalignment='bottom',color = 'k')
    axh.text(1.43*scalef,-1.26*scalef,'$D_{uv}$',fontsize = 13, style='italic', horizontalalignment='right',verticalalignment='bottom',color = 'k')
    
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
    axh.set_xlabel('Wavelength (nm)', fontsize = 9)
    axh.set_ylabel('Radiant power\nEqual luminous flux', fontsize = 9)
    axh.set_xlim([360,830]) 
    axh.set_yticklabels([])
    axh.legend(loc = 'upper right')
    
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
    axh.set_ylabel(r'Color Sample Fidelity $(R_{f,CESi})$')
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
        axh.text(hbins[j],ypos, '{:1.0f}'.format(Rfhi[j,0]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
    xticks = np.arange(data['nhbins'])
    xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
    axh.set_xticks(xticks)
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    axh.set_ylabel(r'Local Color Fidelity $(R_{f,hj})$')
    if xlabel == True:
        axh.set_xlabel('Hue-Angle Bin (j)')
    axh.set_ylim([0,110])
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
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
        axh.text(hbins[j],ypos, '{:1.0f}%'.format(100*Rcshi[j,0]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    xticks = np.array(hbins)
    xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
    axh.set_xticks(xticks)
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    yticks = range(-40,50,10)
    ytickslabels = ['{:1.0f}%'.format(ii) for ii in range(-40,50,10)]
    axh.set_yticks(yticks)
    axh.set_yticklabels(ytickslabels, fontsize = 8)
    axh.set_ylabel(r'Local Chroma Shift $(R_{cs,hj})$')
    if xlabel == True:
        axh.set_xlabel('Hue-Angle Bin (j)')
    axh.set_ylim([min([-40,100*Rcshi.min()]),max([40,100*Rcshi.max()])])
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    return axh

def plot_tm30_Rhshi(spd, cri_type = 'ies-tm30', axh = None, xlabel = False, y_offset = 0, **kwargs):
    
    
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
        axh.text(hbins[j],ypos, '{:1.2f}'.format(Rhshi[j,0]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3, rotation = 90)
    xticks = np.array(hbins)
    xtickslabels = ['{:1.0f}'.format(ii+1) for ii in hbins]
    axh.set_xticks(xticks)
    axh.set_xticklabels(xtickslabels, fontsize = 8)
    axh.set_ylabel(r'Local Hue Shift $(R_{hs,hj})$')
    if xlabel == True:
        axh.set_xlabel('Hue-Angle Bin (j)')
    axh.set_ylim([min([-0.5,Rhshi.min()]),max([0.5,Rhshi.max()])])
    axh.set_xlim([-0.5,data['nhbins']-0.5])
    
    return axh

def plot_tm30_Rxhi(spd, cri_type = 'ies-tm30', axh = None, **kwargs):
    
    data = _process_spd(spd, cri_type = 'ies-tm30',**kwargs)
    
    if axh is None:
        fig, axh = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (6,18))
    
    plot_tm30_Rcshi(data, axh = axh[0], xlabel = False, y_offset = 0.02)
    plot_tm30_Rhshi(data, axh = axh[1], xlabel = False, y_offset = 0.03)
    plot_tm30_Rfhi(data, axh = axh[2], xlabel = True, y_offset = 2)
    return axh

def plot_tm30_report(spd, cri_type = 'ies-tm30', **kwargs):
    
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    f_ax1 = fig.add_subplot(gs[0, :])
    f_ax1.set_title('gs[0, :]')
    f_ax2 = fig.add_subplot(gs[1, :-1])
    f_ax2.set_title('gs[1, :-1]')
    f_ax3 = fig.add_subplot(gs[1:, -1])
    f_ax3.set_title('gs[1:, -1]')
    f_ax4 = fig.add_subplot(gs[-1, 0])
    f_ax4.set_title('gs[-1, 0]')
    f_ax5 = fig.add_subplot(gs[-1, -2])
    f_ax5.set_title('gs[-1, -2]')

if __name__ == '__main__':
    import luxpy as lx
    spd = lx._CIE_F4
    # plot_tm30_cvg(spd, axtype = 'cart', plot_vectors = True, gamut_line_color = 'r')
    # # plot_tm30_spd(spd)
    # # plot_tm30_Rfi(spd)
    # plot_tm30_Rfhi(spd)
    # plot_tm30_Rcshi(spd)
    # plot_tm30_Rhshi(spd)
    # plot_tm30_Rxhi(spd)
    plot_tm30_report(spd)
    