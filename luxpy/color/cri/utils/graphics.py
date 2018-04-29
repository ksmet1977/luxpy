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
# Module for color rendition graphical output, 1
###############################################################################
#
# plot_hue_bins(): Makes basis plot for Color Vector Graphic (CVG).
#
# plot_ColorVectorGraphic(): Plots Color Vector Graphic (see IES TM30).
#
#------------------------------------------------------------------------------


Created on Mon Apr  2 02:00:50 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import np, plt, colorsys,math

__all__ = [ 'plot_hue_bins','plot_ColorVectorGraphic']


def plot_hue_bins(hbins = 16, start_hue = 0.0, scalef = 100, plot_axis_labels = False, bin_labels = '#', plot_edge_lines = True, plot_center_lines = False, plot_bin_colors = True, axtype = 'polar', ax = None, force_CVG_layout = False):
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
            - list[str]: list with str for each bin. 
                (len(:bin_labels:) = :nhbins:)
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
            True: Force plot of basis of CVG on first encounter.
            
    Returns:
        :returns: gcf(), gca(), list with rgb colors for hue bins (for use in 
                                 other plotting fcns)
        
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
        if isinstance(bin_labels,list) | isinstance(bin_labels,np.ndarray):
            bin_labels = bin_labels[idx]
        hbincenters = hbincenters[idx]
        nhbins = hbincenters.shape[0]
    hbincenters = hbincenters*np.pi/180
    
    # Setup hbin labels:
    if bin_labels is '#':
        bin_labels = ['#{:1.0f}'.format(i+1) for i in range(nhbins)]
      
    # initializing the figure
    cmap = None
    if (ax == None) or (ax == 'new'):
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
        #cartesian axis:
        if newfig == True:
            ax = fig.add_axes(rect)

    
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
                hxv = np.vstack((np.zeros(hbincenters.shape),1.3*scalef*np.cos(hbincenters)))
                hyv = np.vstack((np.zeros(hbincenters.shape),1.3*scalef*np.sin(hbincenters)))
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
                if plot_bin_colors == True:
                    bar = ax.bar(dM[i],r[1,i], width = dt[i],color = c,alpha=0.15)
                if bin_labels is not None:
                    ax.text(hbincenters[i],1.3*scalef,bin_labels[i],fontsize = 12, horizontalalignment='center',verticalalignment='center',color = np.array([1,1,1])*0.3)
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

def plot_ColorVectorGraphic(jabt, jabr, hbins = 16, start_hue = 0.0, scalef = 100, plot_axis_labels = False, bin_labels = None, plot_edge_lines = True, plot_center_lines = False, plot_bin_colors = True, axtype = 'polar', ax = None, force_CVG_layout = False):
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
            - list[str]: list with str for each bin. 
                (len(:bin_labels:) = :nhbins:)
            - '#': plots number.
        :plot_edge_lines: True or False, optional
            Plot grey bin edge lines with '--'.
        :plot_center_lines: False or True, optional
            Plot colored lines at 'center' of hue bin.
        :plot_bin_colors: True, optional
            Colorize hue-bins.
        :axtype: 'polar' or 'cart', optional
            Make polar or Cartesian plot.
        :ax: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes.
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
            
    Returns:
        :returns: gcf(), gca(), list with rgb colors for hue bins (for use in 
                                 other plotting fcns)
        
    """
    
    # Plot basis of CVG:
    figCVG, ax, cmap = plot_hue_bins(hbins = hbins, start_hue = start_hue, scalef = scalef, axtype = axtype, ax = ax, plot_center_lines = plot_center_lines, plot_edge_lines = plot_edge_lines, force_CVG_layout = force_CVG_layout, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)

    if cmap == []:
        cmap = ['k' for i in range(hbins)]
        
    if axtype == 'polar':
       
        jabr_theta, jabr_r = math.cart2pol(jabr[...,1:3], htype = 'rad') 
        jabt_theta, jabt_r = math.cart2pol(jabt[...,1:3], htype = 'rad') 
        
        #ax.quiver(jabrtheta,jabr_r,jabt[...,1]-jabr[...,1], jabt[...,2]-jabr_binned[...,2], color = 'k', headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)
        ax.plot(jabt_theta,jabt_r, color = 'grey',linewidth = 2)
        for j in range(hbins):
            c = cmap[j]
            ax.quiver(jabr_theta[j],jabr_r[j],jabt[j,1]-jabr[j,1], jabt[j,2]-jabr[j,2], edgecolor = 'k',facecolor = c, headlength=3, angles='uv', scale_units='y', scale = 2,linewidth = 0.5)
    else:
        #ax.quiver(jabr[...,1],jabr[...,2],jabt[...,1]-jabr[...,1], jabt[...,2]-jabr[...,2], color = 'k', headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)
        ax.plot(jabt,jabt, color = 'grey',linewidth = 2)
        for j in range(hbins):
            ax.quiver(jabr[j,1],jabr[j,2],jabt[j,1]-jabr[j,1], jabt[j,2]-jabr[j,2], color = cmap[j], headlength=3, angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)

    if axtype == 'cart':
        plt.xlabel("a'")
        plt.ylabel("b'")
    
    return plt.gcf(), plt.gca(), cmap

