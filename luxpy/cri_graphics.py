# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for color rendition graphical output
###############################################################################
#
# plot_hue_bins(): Makes basis plot for Color Vector Graphic (CVG).
#
# plot_ColorVectorGraphic(): Plots Color Vector Graphic (see IES TM30).
#
# plot_cri_grpahics(): Plot graphical information on color rendition properties.
#------------------------------------------------------------------------------


Created on Mon Apr  2 02:00:50 2018

@author: kevin.smet
"""

from luxpy import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys

__all__ = ['plot_hue_bins','plot_ColorVectorGraphic','plot_cri_graphics']

def plot_hue_bins(hbins = 16, start_hue = 0.0, scalef = 100, plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, plot_center_lines = False, axtype = 'polar', fig = None, force_CVG_layout = False):
    """
    Makes basis plot for Color Vector Graphic (CVG).
    
    Args:
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
        :fig: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes.
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
            
    Returns:
        :returns: gcf(), gca(), list with rgb colors for hue bins (for use in other plotting fcns)
        
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
        if isinstance(bin_labels,list):
            bin_labels = bin_labels[idx]
        hbincenters = hbincenters[idx]
        nhbins = hbincenters.shape[0]
    hbincenters = hbincenters*np.pi/180
    
    # Setup hbin labels:
    if bin_labels == '#':
        bin_labels = ['#{:1.0f}'.format(i+1) for i in range(nhbins)]
      
    # initializing the figure
    if (fig == None) or (fig == 'new'):
        fig = plt.figure()
        newfig = True
    else:
        newfig = False
    rect = [0.1, 0.1, 0.8, 0.8] # setting the axis limits in [left, bottom, width, height]

    if axtype == 'polar':
        # the polar axis:
        if newfig == True:
            ax = fig.add_axes(rect, polar=True, frameon=False)
        else:
            ax = fig
    else:
        #cartesian axis:
        if newfig == True:
            ax = fig.add_axes(rect)
        else:
            ax = fig

    if (newfig == True) | (force_CVG_layout == True):
        
        # Calculate hue-bin boundaries:
        r = np.vstack((np.zeros(hbincenters.shape),scalef*np.ones(hbincenters.shape)))
        theta = np.vstack((np.zeros(hbincenters.shape),hbincenters))
        t = hbincenters.copy()
        dU = np.roll(hbincenters.copy(),-1)
        dL = np.roll(hbincenters.copy(),1)
        dtU = dU-hbincenters
        dtL = hbincenters -dL
        dtU[dtU<0] = dtU[dtU<0] + 2*np.pi
        dtL[dtL<0] = dtL[dtL<0] + 2*np.pi
        dL = hbincenters - dtL/2
        dU = hbincenters + dtU/2
        dt = (dU-dL)
        dM = dL + dt/2
        
        # Setup color for plotting hue bins:
        hsv_hues = hbincenters - 30*np.pi/180
        hsv_hues = hsv_hues/hsv_hues.max()

        edges = np.vstack((np.zeros(hbincenters.shape),dL)) # setup hue bin edges array
        
        if axtype == 'cart':
            if plot_center_lines == True:
                hx = r*np.cos(theta)
                hy = r*np.sin(theta)
            if bin_labels is not None:
                hxv = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.cos(hbincenters)))
                hyv = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.sin(hbincenters)))
            if plot_edge_lines == True:
                hxe = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.cos(dL)))
                hye = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.sin(dL)))
            
        # Plot hue-bins:
        for i in range(nhbins):
            
            # Create color from hue angle:
            c = np.abs(np.array(colorsys.hsv_to_rgb(hsv_hues[i], 0.84, 0.9)))
            #c = [abs(c[0]),abs(c[1]),abs(c[2])] # ensure all positive elements
            if i == 0:
                cmap = [c]
            else:
                cmap.append(c)
   
            
            if axtype == 'polar':
                if plot_edge_lines == True:
                    ax.plot(edges[:,i],r[:,i]*1.2,color = 'grey',marker = 'None',linestyle = ':',linewidth = 3, markersize = 2)
                if plot_center_lines == True:
                    if np.mod(i,2) == 1:
                        ax.plot(theta[:,i],r[:,i],color = c,marker = None,linestyle = '--',linewidth = 2)
                    else:
                        ax.plot(theta[:,i],r[:,i],color = c,marker = 'o',linestyle = '-',linewidth = 3,markersize = 10)
                bar = ax.bar(dM[i],r[1,i], width = dt[i],color = c,alpha=0.15)
                if bin_labels is not None:
                    ax.text(hbincenters[i],1.2*scalef,bin_labels[i],fontsize = 12, horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
                if plot_axis_labels == False:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
            else:
                if plot_edge_lines == True:
                    ax.plot(hxe[:,i],hye[:,i],color = 'grey',marker = 'None',linestyle = ':',linewidth = 3, markersize = 2)

                if plot_center_lines == True:
                    if np.mod(i,2) == 1:
                        ax.plot(hx[:,i],hy[:,i],color = c,marker = None,linestyle = '--',linewidth = 2)
                    else:
                        ax.plot(hx[:,i],hy[:,i],color = c,marker = 'o',linestyle = '-',linewidth = 3,markersize = 10)
                if bin_labels is not None:
                    ax.text(hxv[1,i],hyv[1,i],bin_labels[i],fontsize = 12,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
                ax.axis(1.1*np.array([hxv.min(),hxv.max(),hyv.min(),hyv.max()]))
                if plot_axis_labels == False:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                else:
                    plt.xlabel("a'")
                    plt.ylabel("b'")

        plt.plot(0,0,color = 'k',marker = 'o',linestyle = None)

    return plt.gcf(), plt.gca(), cmap

def plot_ColorVectorGraphic(jabt, jabr, hbins = 16, start_hue = 0.0, scalef = 100, plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, plot_center_lines = False, axtype = 'polar', fig = None, force_CVG_layout = False):
    """
    Plot Color Vector Graphic (CVG).
    
    Args:
        :jabt: numpy.ndarray with jab data under test SPD
        :jabr: numpy.ndarray with jab data under reference SPD
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
        :fig: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes.
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
            
    Returns:
        :returns: gcf(), gca(), list with rgb colors for hue bins (for use in other plotting fcns)
        
    """
    
    # Plot basis of CVG:
    figCVG, ax, cmap = plot_hue_bins(hbins = hbins, axtype = axtype, fig = fig, plot_center_lines = plot_center_lines, plot_edge_lines = plot_edge_lines, scalef = scalef, force_CVG_layout = force_CVG_layout)

    if cmap == []:
        cmap = ['k' for i in range(hbins)]
        
    if axtype == 'polar':
       
        jabr_theta, jabr_r = math.cart2pol(jabr[...,1:3], htype = 'rad') 
        
        #ax.quiver(jabrtheta,jabr_r,jabt[...,1]-jabr[...,1], jabt[...,2]-jabr_binned[...,2], color = 'k', headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)
        for j in range(hbins):
            c = cmap[j]
            ax.quiver(jabr_theta[j],jabr_r[j],jabt[j,1]-jabr[j,1], jabt[j,2]-jabr[j,2], edgecolor = 'k',facecolor = c, headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)
    else:
        #ax.quiver(jabr[...,1],jabr[...,2],jabt[...,1]-jabr[...,1], jabt[...,2]-jabr[...,2], color = 'k', headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)
        for j in range(hbins):
            ax.quiver(jabr[j,1],jabr[j,2],jabt[j,1]-jabr[j,1], jabt[j,2]-jabr[j,2], color = cmap[j], headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)

    if axtype == 'cart':
        plt.xlabel("a'")
        plt.ylabel("b'")
    
    return plt.gcf(), plt.gca(), cmap

def plot_cri_graphics(data, cri_type = None, hbins = 16, start_hue = 0.0, scalef = 100, plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, plot_center_lines = False, axtype = 'polar', fig = None, force_CVG_layout = False):
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
        :fig: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes.
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
            
    Returns:
        :returns: gcf(), gca(), list with rgb colors for hue bins (for use in other plotting fcns)
        
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
        spd_to_iestm30 = lambda x: cri.spd_to_cri(x, cri_type = cri_type, out = out)
        Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type = spd_to_iestm30(SPD)
        
        # Get normalized and sliced data for plotting:
        rg_pars = cri_type['rg_pars']
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
        normalize_gamut = True #(for plotting)
        normalized_chroma_ref = scalef; # np.sqrt((jabr[...,1]**2 + jabr[...,2]**2)).mean(axis = 0).mean()
        
        bjabt, bjabr= cri.gamut_slicer(jabt,jabr, out = 'jabt,jabr', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)

        # Create dict with CRI info:
        data = {'SPD' : data, 'cct' : cct, 'duv' : duv, 'bjabt' : bjabt, 'bjabr' : bjabr,\
               'Rf' : Rf, 'Rg' : Rg, 'Rfi': Rfi, 'Rfhi' : Rfhi, 'Rchhi' : Rcshi, 'Rhshi' : Rhshi, 'cri_type' : cri_type}
  
        
    for i in range(cct.shape[0]):
        # Plot test SPD:
        fig = plt.figure(figsize=(10, 6), dpi=144)
        ax_spd = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
        ax_spd.plot(SPD[0],SPD[i+1]/SPD[i+1].max(),'r-')
        ax_spd.text(730,0.9,'CCT = {:1.0f}K'.format(cct[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.8,'Duv = {:1.4f}'.format(duv[i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.7,'IES Rf = {:1.0f}'.format(Rf[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.text(730,0.6,'IES Rg = {:1.0f}'.format(Rg[:,i][0]),fontsize = 9, horizontalalignment='left',verticalalignment='center',rotation = 0, color = np.array([1,1,1])*0.3)
        ax_spd.set_xlabel('Wavelength (nm)')
        ax_spd.set_ylabel('Rel. spectral intensity')
        ax_spd.set_xlim([360,830])
        
        # Plot CVG:
        ax_CVG = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2, polar = True, frameon=False)
        force_CVG_plot = True
        figCVG, ax, cmap = plot_ColorVectorGraphic(bjabt[...,i,:], bjabr[...,i,:], hbins = hbins, axtype = axtype, fig = ax_CVG, plot_center_lines = False, plot_edge_lines = True, scalef = scalef, force_CVG_layout = force_CVG_plot)
        #ax_CVG.set_title('Color Vector Graphic')
    
        
        # Plot local chroma shift, Rcshi:
        ax_locC = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        for j in np.arange(hbins):
            ax_locC.bar(np.arange(hbins)[j],Rcshi[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locC.text(np.arange(hbins)[j],-np.sign(Rcshi[j,i])*0.1, '{:1.0f}%'.format(100*Rcshi[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ax_locC.set_ylim([Rcshi.min()*2,Rcshi.max()*2])
        ax_locC.set_ylabel(r'Local chroma shift, $R_{cs,hi}$')
        ax_locC.set_xticklabels([])
        ax_locC.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locC.set_ylim()], color = 'white')
    
        # Plot local hue shift, Rhshi:
        ax_locH = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        for j in np.arange(hbins):
            ax_locH.bar(np.arange(hbins)[j],Rhshi[j,i], color = cmap[j], width = 1,edgecolor = 'k', alpha = 0.4)
            ax_locH.text(np.arange(hbins)[j],-np.sign(Rhshi[j,i])*0.2, '{:1.3f}'.format(Rhshi[j,i]) ,fontsize = 9,horizontalalignment='center',verticalalignment='center',rotation = 90, color = np.array([1,1,1])*0.3)
        ax_locH.set_ylim([Rhshi.min()*2,Rhshi.max()*2])
        ax_locH.set_ylabel(r'Local hue shift, $R_{hs,hi}$')
        ax_locH.set_xticklabels([])
        ax_locH.set_yticklabels(['{:1.2f}'.format(ii) for ii in ax_locH.set_ylim()], color = 'white')
        
        # Plot local color fidelity, Rfhi:
        ax_Rfi = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
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
        
        plt.tight_layout()
        
    return  data, plt.gcf()
