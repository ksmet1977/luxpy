# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:55:02 2018

@author: kevin.smet
"""

import luxpy as lx
import numpy as np
import matplotlib.pyplot as plt
import colorsys

plot_iestm30_output = True

SPDs = lx._IESTM30['S']['data']

SPD = SPDs[:3]

Nspds = SPD.shape[0] - 1

cri_iestm30_defaults = lx.cri._CRI_DEFAULTS['iesrf']
rg_pars_iesttm30 =cri_iestm30_defaults['rg_pars']

# Calculate metrics:
out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr'
spd_to_iestm30 = lambda x: lx.cri.spd_to_cri(x, cri_type = cri_iestm30_defaults, out = out)
Rf,Rg,cct,duv,Rfi,jabt,jabr = spd_to_iestm30(SPD)



def plot_hue_bins(nhbins = None, start_hue = 0.0, scalef = 100, axis_labels = False, plot_bin_labels = True, plot_edge_lines = True, plot_center_lines = False, color = 'k', axtype = 'polar', fig = None):
    
    dhbins = 360/(nhbins) # hue bin width
    hbincenters = np.arange(0 + dhbins/2, 360, dhbins)
    hbincenters = np.sort(hbincenters)
    huebinlabels = ['#{:1.0f}'.format(i+1) for i in range(nhbins)]
    hsv_hues = (hbincenters - 22.5)/360
    hbincenters = hbincenters*np.pi/180
    
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
            ax = fig.gca()
    else:
        #cartesian axis:
        if newfig == True:
            ax = fig.add_axes(rect)
        else:
            ax = fig.gca()

    if newfig == True:
        
        # Calculate hue-bin boundaries:
        r = np.vstack((np.zeros(hbincenters.shape),scalef*np.ones(hbincenters.shape)))
        theta = np.vstack((np.zeros(hbincenters.shape),hbincenters))
        t = hbincenters.copy()
        dU = hbincenters
        dL = np.roll(hbincenters.copy(),1)
        dt = dU-dL
        dt[dt<0] = dt[dt<0] + 2*np.pi
        dL = hbincenters - dt/2
        dU = hbincenters + dt/2
        dM = (dL + dt/2)
#        print(np.vstack((dL,dU,dM,dt)).T)
        edges = np.vstack((np.zeros(hbincenters.shape),dL))
        if axtype == 'polar':
            pass
        else:
            if plot_center_lines == True:
                hx = r*np.cos(theta)
                hy = r*np.sin(theta)
            if plot_bin_labels == True:
                hxv = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.cos(hbincenters)))
                hyv = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.sin(hbincenters)))
            if plot_edge_lines == True:
                hxe = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.cos(dL)))
                hye = np.vstack((np.zeros(hbincenters.shape),1.2*scalef*np.sin(dL)))
            
        # Plot hue-bins:
        for i,v in enumerate(huebinlabels):
            
            c = colorsys.hsv_to_rgb(hsv_hues[i], 0.84, 0.9)
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
                if plot_bin_labels == True:
                    ax.text(hbincenters[i],1.2*scalef,huebinlabels[i],fontsize = 12, horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
                if axis_labels == False:
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
                if plot_bin_labels == True:
                    ax.text(hxv[1,i],hyv[1,i],huebinlabels[i],fontsize = 12,horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
                ax.axis(1.1*np.array([hxv.min(),hxv.max(),hyv.min(),hyv.max()]))
                if axis_labels == False:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                else:
                    plt.xlabel("a'")
                    plt.ylabel("b'")

        plt.plot(0,0,color = 'k',marker = 'o',linestyle = None)

    return plt.gcf(), plt.gca(), cmap


#______________________________________________________________________________
# Plot IES TM30 output:
if plot_iestm30_output == True:
    
    # Get normalized and sliced data for plotting:
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars_iesttm30[x] for x in sorted(rg_pars_iesttm30.keys())]
    normalize_gamut = True #(for plotting)
    normalized_chroma_ref = 100;#np.sqrt((jabr[...,1]**2 + jabr[...,2]**2)).mean(axis = 0).mean()
    
    jabt_binned, jabr_binned, jabc_binned = lx.cri.gamut_slicer(jabt,jabr, out = 'jabt,jabr,jabc', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)


    # plot ColorVectorGraphic:
    axtype ='polar'
    rect = [0.1, 0.1, 0.8, 0.8]
    
    for i in range(Nspds):
        
        fig, ax, cmap = plot_hue_bins(nhbins, axtype = axtype, fig = None, plot_center_lines = False, plot_edge_lines = True, scalef = normalized_chroma_ref*1.2)
        
        if cmap == []:
            cmap = ['k' for i in range(nhbins)]

        
        if axtype == 'polar':
            # Cart2pol:
            jabr_binned_theta, jabr_binned_r = lx.math.cart2pol(jabr_binned[...,i,1:3], htype = 'rad')
            jabc_binned_theta, jabc_binned_r = lx.math.cart2pol(jabc_binned[...,1:3], htype = 'rad')
            #ax.quiver(jabr_binned_theta,jabr_binned_r,jabt_binned[...,i,1]-jabr_binned[...,i,1], jabt_binned[...,i,2]-jabr_binned[...,i,2], color = 'k', headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)

            for j in range(nhbins):
                c = cmap[j]
                ax.quiver(jabr_binned_theta[j],jabr_binned_r[j],jabt_binned[j,i,1]-jabr_binned[j,i,1], jabt_binned[j,i,2]-jabr_binned[j,i,2], edgecolor = 'k',facecolor = c, headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)

            ax.plot(jabc_binned_theta,jabc_binned_r,'grey',linewidth = 2.5,linestyle=':')
    
        else:
#            ax.quiver(jabr_binned[...,i,1],jabr_binned[...,i,2],jabt_binned[...,i,1]-jabr_binned[...,i,1], jabt_binned[...,i,2]-jabr_binned[...,i,2], color = cmap[j], headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)
            for j in range(nhbins):
                ax.quiver(jabr_binned[j,i,1],jabr_binned[j,i,2],jabt_binned[j,i,1]-jabr_binned[j,i,1], jabt_binned[j,i,2]-jabr_binned[j,i,2], color = cmap[j], headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)
            ax.plot(jabc_binned[...,1],jabc_binned[...,2],'grey',linewidth = 2.5,linestyle=':')
    
    if axtype == 'cart':
        plt.xlabel("a'")
        plt.ylabel("b'")
    plt.show()