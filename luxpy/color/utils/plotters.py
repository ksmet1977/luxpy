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
Module with functions related to plotting of color data
=======================================================

 :get_cmap(): Get an ndarray of rgb values representing a linearly sampled matplotlib colormap

 :get_subplot_layout(): Calculate layout of multiple subplots.
     
 :plot_color_data(): Plot color data (local helper function)

 :plotDL(): Plot daylight locus. 

 :plotBB(): Plot blackbody locus. 

 :plotSL(): | Plot spectrum locus.  
            | (plotBB() and plotDL() are also called, but can be turned off).

 :plotcerulean(): | Plot cerulean (yellow (577 nm) - blue (472 nm)) line 
                  | (Kuehni, CRA, 2014: Table II: spectral lights)
                  | `Kuehni, R. G. (2014). 
                    Unique hues and their stimuli—state of the art. 
                    Color Research & Application, 39(3), 279–287.
                    <https://doi.org/10.1002/col.21793>`_

 :plotUH(): | Plot unique hue lines from color space center point xyz0. 
            | (Kuehni, CRA, 2014: uY,uB,uG: Table II: spectral lights; 
            | uR: Table IV: Xiao data) 
            | `Kuehni, R. G. (2014). 
              Unique hues and their stimuli—state of the art. 
              Color Research & Application, 39(3), 279–287.
              <https://doi.org/10.1002/col.21793>`_
    
 :plotcircle(): Plot one or more concentric circles.
 
 :plotellipse(): Plot one or more ellipses.
     
 :plot_chromaticity_diagram_colors(): Plot the chromaticity diagram colors.

 :plot_spectrum_colors(): Plot spd with spectrum colors.
 
 :plot_rfl_color_patches(): Create (and plot) an image with colored patches representing a set of reflectance spectra illuminated by a specified illuminant.
 
 :plot_rgb_color_patches(): Create (and plot) an image with patches with specified rgb values.
 
 :plot_cmfs(): Plot CMFs.
 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np

from luxpy import math, _CIEOBS, _CSPACE, _CSPACE_AXES, _CIE_ILLUMINANTS, _CMF, _CIE_D65, daylightlocus, colortf, Yxy_to_xyz, spd_to_xyz, cri_ref, xyz_to_srgb
from luxpy.utils import _EPS, asplit

__all__ = ['get_cmap','get_subplot_layout','plotSL','plotDL','plotBB','plot_color_data',
           'plotceruleanline','plotUH','plotcircle','plotellipse',
           'plot_chromaticity_diagram_colors','plot_spectrum_colors',
           'plot_rfl_color_patches','plot_rgb_color_patches','plot_cmfs']

def get_cmap(N, cmap_name = 'jet'):
    """
    Get an ndarray of rgba values representing a linearly sampled matplotlib colormap.
    
    Args:
        :N:
            | Number of rgba values in returned cmap.
        :cmap_name:
            | 'jet', optional
            | Matplotlib color map name to sample from.
            
    Returns:
        :cmap:
            | ndarray with rgba values.
    """
    from matplotlib import cm # lazy import
    cmap = cm.get_cmap(cmap_name, N)
    cmap = cmap(range(N))
    return cmap

def get_subplot_layout(N, min_1xncols = 3):
    """
    Calculate layout of multiple subplots.
    
    Args:
        :N:
            | Number of plots.
        :min_1xncols:
            | Minimum number of columns before splitting over multiple rows.
            
    Returns:
        :nrows, ncols:
    """
    if N <= min_1xncols:
        return 1, N
    else:
        sqr = np.ceil(N**0.5)
        nrows = N//sqr
        ncols = np.ceil(N/nrows)
        return int(nrows), int(ncols)

def plot_color_data(x,y,z=None, axh=None, show = True, cieobs =_CIEOBS, \
                    cspace = _CSPACE,  formatstr = 'k-', legend_loc = None, **kwargs):
    """
    Plot color data from x,y [,z].
    
    Args: 
        :x: 
            | float or ndarray with x-coordinate data
        :y: 
            | float or ndarray with y-coordinate data
        :z: 
            | None or float or ndarray with Z-coordinate data, optional
            | If None: make 2d plot.
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show: 
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs: 
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str or None, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
            | If None: don't do any formatting of x,y [z] axes.
        :formatstr: 
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :kwargs:
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | handle to current axes (:show: == False)
    """
    import matplotlib.pyplot as plt # lazy import
    
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    if z is not None:
        z = np.atleast_1d(z)
        if axh is None:
            fig = plt.figure()
            axh = plt.axes(projection='3d')
        if 'grid' in kwargs.keys():
            axh.grid(kwargs['grid']);kwargs.pop('grid')
        axh.plot3D(x,y,z,formatstr, linewidth = 2,**kwargs)
        axh.set_zlabel(_CSPACE_AXES[cspace][0], kwargs)
    else:
        if axh is None:
            fig = plt.figure()
            axh = plt.axes()
        if 'grid' in kwargs.keys():
            axh.grid(kwargs['grid']);kwargs.pop('grid')
        axh.plot(x,y,formatstr,linewidth = 2,**kwargs)   
        axh.set_xlabel(_CSPACE_AXES[cspace][1], kwargs)
        axh.set_ylabel(_CSPACE_AXES[cspace][2], kwargs)
    if 'label' in kwargs.keys():
        axh.legend(loc = legend_loc)
    if show == True:
        plt.show()
    return axh



def plotDL(ccts = None, cieobs =_CIEOBS, cspace = _CSPACE, axh = None, \
           show = True, force_daylight_below4000K = False, cspace_pars = {}, \
           formatstr = 'k-', Y_DL = 100,  **kwargs):
    """
    Plot daylight locus.
    
    Args: 
        :ccts: 
            | None or list[float], optional
            | None defaults to [4000 K to 1e11 K] in 100 steps on a log10 scale.
        :force_daylight_below4000K: 
            | False or True, optional
            | CIE daylight phases are not defined below 4000 K. 
            | If True plot anyway.
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show: 
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr:
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | handle to current axes (:show: == False)
    """
    
    if ccts is None:
        ccts = 10**np.linspace(np.log10(4000.0),np.log10(10.0**11.0),100)
        
    xD,yD = daylightlocus(ccts, cieobs = cieobs, force_daylight_below4000K = force_daylight_below4000K)
    Y = Y_DL*np.ones(xD.shape)
    DL =  Yxy_to_xyz(np.vstack((Y, xD,yD)).T)
    DL = colortf(DL, tf = cspace, fwtf = cspace_pars)
    Y,x,y = asplit(DL)
    
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    
    return axh
    
def plotBB(ccts = None, cieobs =_CIEOBS, cspace = _CSPACE, axh = None, cctlabels = True, show = True, cspace_pars = {}, formatstr = 'k-', Y_BB = 100, **kwargs):  
    """
    Plot blackbody locus.
        
    Args: 
        :ccts: 
            | None or list[float], optional
            | None defaults to [1000 to 1e19 K].
            | Range: 
            |     [1000,1500,2000,2500,3000,3500,4000,5000,6000,8000,10000] 
            |    + [15000 K to 1e11 K] in 100 steps on a log10 scale
        :cctlabels:
            | True or False, optional
            | Add cct text labels at various points along the blackbody locus.
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show:
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr:
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | handle to current axes (:show: == False)
    """
    if ccts is None:
        ccts1 = np.array([1000.0,1500.0,2000.0,2500.0,3000.0,3500.0,4000.0,5000.0,6000.0,8000.0,10000.0])
        ccts2 = 10**np.linspace(np.log10(15000.0),np.log10(10.0**11.0),100)
        ccts = np.hstack((ccts1,ccts2))
    else:
        ccts1 = None
    
    BB = cri_ref(ccts,ref_type='BB')
    xyz = spd_to_xyz(BB,cieobs = cieobs)
    xyz = Y_BB*xyz/xyz[...,1:2]
    Yxy = colortf(xyz, tf = cspace, fwtf = cspace_pars)
    Y,x,y = asplit(Yxy)
   
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    

    if (cctlabels == True) & (ccts1 is not None):
        for i in range(ccts1.shape[0]):
            if ccts1[i]>= 3000.0:
                if i%2 == 0.0:
                    axh.plot(x[i],y[i],'k+', color = '0.5')
                    axh.text(x[i]*1.05,y[i]*0.95,'{:1.0f}K'.format(ccts1[i]), color = '0.5')
        axh.plot(x[-1],y[-1],'k+', color = '0.5')
        axh.text(x[-1]*1.05,y[-1]*0.95,'{:1.0e}K'.format(ccts[-1]), color = '0.5')    
    return axh
    
def plotSL(cieobs =_CIEOBS, cspace = _CSPACE, DL = False, BBL = True, D65 = False,\
           EEW = False, cctlabels = False, axh = None, show = True,\
           cspace_pars = {}, formatstr = 'k-',\
           diagram_colors = False, diagram_samples = 100, diagram_opacity = 1.0,\
           diagram_lightness = 0.25, Y_SL = 100,\
           **kwargs):
    """
    Plot spectrum locus for cieobs in cspace.
    Only works / makes sense for Yxy, Yuv, luv (basically any chromaticity diagram where Y or lightness of spectrum locus is not relative that of xyzw, because what would be its value? These are lights!)
 
    
    Args: 
        :DL: 
            | True or False, optional
            | True plots Daylight Locus as well.
        :BBL: 
            | True or False, optional
            | True plots BlackBody Locus as well. 
        :D65: 
            | False or True, optional
            | True plots D65 chromaticity as well. 
        :EEW: 
            | False or True, optional
            | True plots Equi-Energy-White chromaticity as well. 
        :cctlabels:
            | False or True, optional
            | Add cct text labels at various points along the blackbody locus.
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show:
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr:
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
        :diagram_colors:
            | False, optional
            | True: plot colored chromaticity diagram.
        :diagram_samples:
            | 256, optional
            | Sampling resolution of color space.
        :diagram_opacity:
            | 1.0, optional
            | Sets opacity of chromaticity diagram
        :diagram_lightness:
            | 0.25, optional
            | Sets lightness of chromaticity diagram
        :Y_SL:
            | 100, optional
            | Normalize the XYZ of the spectrum locus to this value before converting to cspace.
            | Note that plots of the daylight locus, blackbody locus, etc. are scaled accordingly.

        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | handle to current axes (:show: == False)
    """
    
    if isinstance(cieobs,str):
        SL = _CMF[cieobs]['bar'].copy()
    else:
        SL = cieobs.copy()
    wl, SL = SL[0], SL[1:4].T
    SL = Y_SL*SL/(SL[:,1,None] + _EPS) # normalize so that Y=Y_SL
    cnd = SL.sum(axis=1)>0
    SL = SL[cnd,:] # avoid div by zero in xyz-to-Yxy conversion
    wl = wl[cnd]
    SL = colortf(SL, tf = cspace, fwtf = cspace_pars)
    
    x_coord = SL[...,1]
    dx_coord = np.vstack((*np.diff(x_coord),0))
    plambdamax = np.where((wl>=600) & (dx_coord[:,0]<0))[0][0]
    SL = np.vstack((SL[:(plambdamax+1),:],SL[0])) # add lowest wavelength data and go to max of gamut in x (there is a reversal for some cmf set wavelengths >~700 nm!)
    Y,x,y = asplit(SL)
    
    showcopy = show
    if np.any([DL,BBL,D65,EEW]):
        show = False

    if diagram_colors == True:
        axh = plot_chromaticity_diagram_colors(axh = axh, show = diagram_colors, cspace = cspace, cieobs = cieobs,\
                                                cspace_pars = cspace_pars,\
                                                diagram_samples = diagram_samples,\
                                                diagram_opacity = diagram_opacity,\
                                                diagram_lightness = diagram_lightness, Y_SL = Y_SL,\
                                                label_fontname = None, label_fontsize = None)
     
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show = show, formatstr=formatstr,  **kwargs)


    if DL == True:
        if 'label' in kwargs.keys(): # avoid label also being used for DL
            kwargs.pop('label')
        plotDL(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, formatstr = 'k:', Y_DL = Y_SL, **kwargs)

    if BBL == True:
        if 'label' in kwargs.keys(): # avoid label also being used for BB
            kwargs.pop('label')
        plotBB(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, cctlabels = cctlabels, formatstr = 'k-.', Y_BB = Y_SL, **kwargs)
    
    if D65 == True:
        xyzD65 = spd_to_xyz(_CIE_ILLUMINANTS['D65'], cieobs = cieobs)
        xyzD65 = Y_SL*xyzD65/xyzD65[...,1:2]      
        YxyD65 = colortf(xyzD65, tf = cspace,  fwtf = cspace_pars)
        axh.plot(YxyD65[...,1],YxyD65[...,2],'bo')
    if EEW == True:
        xyzEEW = spd_to_xyz(_CIE_ILLUMINANTS['E'], cieobs = cieobs)
        xyzEEW = Y_SL*xyzEEW/xyzEEW[...,1:2]   
        YxyEEW = colortf(xyzEEW, tf = cspace, fwtf = cspace_pars)
        axh.plot(YxyEEW[...,1],YxyEEW[...,2],'ko')
    
    if showcopy:
        import matplotlib.pyplot as plt # lazy import
        plt.show()
    return axh    
        
def plotceruleanline(cieobs = _CIEOBS, cspace = _CSPACE, axh = None,formatstr = 'ko-', cspace_pars = {}):
    """
    Plot cerulean (yellow (577 nm) - blue (472 nm)) line 
    
    | Kuehni, CRA, 2014: 
    |   Table II: spectral lights.
    
    Args: 
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr:
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
        :kwargs:
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns:
            | handle to cerulean line
        
    References:
        1. `Kuehni, R. G. (2014). 
        Unique hues and their stimuli—state of the art. 
        Color Research & Application, 39(3), 279–287. 
        <https://doi.org/10.1002/col.21793>`_
        (see Table II, IV)
    """
    if isinstance(cieobs,str):
        cmf = _CMF[cieobs]['bar'].copy()
    else:
        cmf = cieobs.copy()
    p_y = cmf[0] == 577.0 #Kuehni, CRA 2013 (mean, table IV)
    p_b = cmf[0] == 472.0 #Kuehni, CRA 2013 (mean, table IV)
    xyz_y = cmf[1:,p_y].T
    xyz_b = cmf[1:,p_b].T
    lab = colortf(np.vstack((xyz_b,xyz_y)),tf = cspace, fwtf = cspace_pars)
    if axh is None:
        import matplotlib.pyplot as plt # lazy import
        axh = plt.gca()
    hcerline = axh.plot(lab[:,1],lab[:,2],formatstr,label = 'Cerulean line')    
    return hcerline

    
def plotUH(xyz0 = None, uhues = [0,1,2,3], cieobs = _CIEOBS, cspace = _CSPACE, axh = None,formatstr = ['yo-.','bo-.','ro-.','go-.'], excludefromlegend = '',cspace_pars = {}):
    """ 
    Plot unique hue lines from color space center point xyz0.
    
    | Kuehni, CRA, 2014: 
    |     uY,uB,uG: Table II: spectral lights; 
    |     uR: Table IV: Xiao data.
    
    Args: 
        :xyz0:
            | None, optional
            | Center of color space (unique hue lines are expected to cross here)
            | None defaults to equi-energy-white.
        :uhues:
            | [0,1,2,3], optional
            | Unique hue lines to plot [0:'yellow',1:'blue',2:'red',3:'green']
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr:
            | ['yo-.','bo-.','ro-.','go-.'] or list[str], optional
            | Format str for plotting the different unique lines 
            | (see also ?matplotlib.pyplot.plot)
        :excludefromlegend:
            | '' or str, optional
            | To exclude certain hues from axes legend.
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
          
    Returns:
        :returns: 
            | list[handles] to unique hue lines
        
    References:
        1. `Kuehni, R. G. (2014). 
        Unique hues and their stimuli—state of the art. 
        Color Research & Application, 39(3), 279–287. 
        <https://doi.org/10.1002/col.21793>`_
        (see Table II, IV)
    """
    hues = ['yellow','blue','red','green']
    if isinstance(cieobs,str):
        cmf = _CMF[cieobs]['bar'].copy()
    else:
        cmf = cieobs.copy()
    p_y = cmf[0] == 577.0 #unique yellow,#Kuehni, CRA 2013 (mean, table IV: spectral data)
    p_b = cmf[0] == 472.0 #unique blue,Kuehni, CRA 2013 (mean, table IV: spectral data)
    p_g = cmf[0] == 514.0 #unique green, Kuehni, CRA 2013 (mean, table II: spectral data)
    p_r = cmf[0] == 650.0 #unique red, Kuehni, CRA 2013 (Xiao data, table IV: display data)
    xyz_y = 100.0*cmf[1:,p_y].T
    xyz_b = 100.0*cmf[1:,p_b].T
    xyz_g = 100.0*cmf[1:,p_g].T
    xyz_r = 100.0*cmf[1:,p_r].T
    xyz_uh = np.vstack((xyz_y,xyz_b,xyz_r,xyz_g))
    huniquehues = []
    if xyz0 is None:
        xyz0 = np.array([100.0,100.0,100.0])
    if axh is None:
        import matplotlib.pyplot as plt # lazy import
        axh = plt.gca()
    for huenr in uhues:
        lab = colortf(np.vstack((xyz0,xyz_uh[huenr])),tf = cspace, fwtf = cspace_pars)
        huh = axh.plot(lab[:,1],lab[:,2],formatstr[huenr],label = excludefromlegend + 'Unique '+ hues[huenr])
        huniquehues = [huniquehues,huh]
    return  huniquehues

#------------------------------------------------------------------------------
def plotcircle(center = np.array([[0.,0.]]),radii = np.arange(0,60,10), 
               angles = np.arange(0,350,10),color = 'k',linestyle = '--', 
               out = None, axh = None, **kwargs):
    """
    Plot one or more concentric circles.
    
    Args:
        :center: 
            | np.array([[0.,0.]]) or ndarray with center coordinates, optional
        :radii:
            | np.arange(0,60,10) or ndarray with radii of circle(s), optional
        :angles:
            | np.arange(0,350,10) or ndarray with angles (°), optional
        :color:
            | 'k', optional
            | Color for plotting.
        :linestyle:
            | '--', optional
            | Linestyle of circles.
        :out: 
            | None, optional
            | If None: plot circles, return (x,y) otherwise.
    """
    xs = np.array([0])
    ys = xs.copy()
    if ((out != 'x,y') & (axh is None)):
        import matplotlib.pyplot as plt # lazy import
        fig, axh = plt.subplots(rows=1,ncols=1)
    for ri in radii:
        x = center[:,0] + ri*np.cos(angles*np.pi/180)
        y = center[:,1] + ri*np.sin(angles*np.pi/180)
        xs = np.hstack((xs,x))
        ys = np.hstack((ys,y))
        if (out != 'x,y'):
            axh.plot(x,y,color = color, linestyle = linestyle, **kwargs)
    if out == 'x,y':
        return xs,ys
    elif out == 'axh':
        return axh

#------------------------------------------------------------------------------
def plotellipse(v, cspace_in = 'Yxy', cspace_out = None, nsamples = 100, \
                show = True, axh = None, \
                line_color = 'darkgray', line_style = ':', line_width = 1, line_marker = '', line_markersize = 4,\
                plot_center = False, center_marker = 'o', center_color = 'darkgray', center_markersize = 4,\
                show_grid = False, llabel = '', label_fontname = 'Times New Roman', label_fontsize = 12,\
                out = None):
    """
    Plot ellipse(s) given in v-format [Rmax,Rmin,xc,yc,theta].
    
    Args:
        :v: 
            | (Nx5) ndarray
            | ellipse parameters [Rmax,Rmin,xc,yc,theta]
        :cspace_in:
            | 'Yxy', optional
            | Color space of v.
            | If None: no color space assumed. Axis labels assumed ('x','y').
        :cspace_out:
            | None, optional
            | Color space to plot ellipse(s) in.
            | If None: plot in cspace_in.
        :nsamples:
            | 100 or int, optional
            | Number of points (samples) in ellipse boundary
        :show:
            | True or boolean, optional
            | Plot ellipse(s) (True) or not (False)
        :axh: 
            | None, optional
            | Ax-handle to plot ellipse(s) in.
            | If None: create new figure with axes.
        :line_color:
            | 'darkgray', optional
            | Color to plot ellipse(s) in.
        :line_style:
            | ':', optional
            | Linestyle of ellipse(s).
        :line_width':
            | 1, optional
            | Width of ellipse boundary line.
        :line_marker:
            | 'none', optional
            | Marker for ellipse boundary.
        :line_markersize:
            | 4, optional
            | Size of markers in ellipse boundary.
        :plot_center:
            | False, optional
            | Plot center of ellipse: yes (True) or no (False)
        :center_color:
            | 'darkgray', optional
            | Color to plot ellipse center in.
        :center_marker:
            | 'o', optional
            | Marker for ellipse center.
        :center_markersize:
            | 4, optional
            | Size of marker of ellipse center.
        :show_grid:
            | False, optional
            | Show grid (True) or not (False)
        :llabel:
            | None,optional
            | Legend label for ellipse boundary.
        :label_fontname: 
            | 'Times New Roman', optional
            | Sets font type of axis labels.
        :label_fontsize:
            | 12, optional
            | Sets font size of axis labels.
        :out:
            | None, optional
            | Output of function
            | If None: returns None. Can be used to output axh of newly created
            |      figure axes or to return Yxys an ndarray with coordinates of 
            |       ellipse boundaries in cspace_out (shape = (nsamples,3,N)) 
            
        
    Returns:
        :returns: None, or whatever set by :out:.
    """
    Yxys = np.zeros((nsamples,3,v.shape[0]))
    ellipse_vs = np.zeros((v.shape[0],5))
    for i,vi in enumerate(v):
        
        # Set sample density of ellipse boundary:
        t = np.linspace(0, 2*np.pi, int(nsamples))
        
        a = vi[0] # major axis
        b = vi[1] # minor axis
        xyc = vi[2:4,None] # center
        theta = vi[-1] # rotation angle
        
        # define rotation matrix:
        R = np.hstack(( np.vstack((np.cos(theta), np.sin(theta))), np.vstack((-np.sin(theta), np.cos(theta)))))
 
        # Calculate ellipses:
        Yxyc = np.vstack((1, xyc)).T
        Yxy = np.vstack((np.ones((1,nsamples)), xyc + np.dot(R, np.vstack((a*np.cos(t), b*np.sin(t))) ))).T
        Yxys[:,:,i] = Yxy
        
        # Convert to requested color space:
        if (cspace_out is not None) & (cspace_in is not None):
            Yxy = colortf(Yxy, cspace_in + '>' + cspace_out)
            Yxyc = colortf(Yxyc, cspace_in + '>' + cspace_out)
            Yxys[:,:,i] = Yxy
            
            # get ellipse parameters in requested color space:
            ellipse_vs[i,:] = math.fit_ellipse(Yxy[:,1:])
            #de = np.sqrt((Yxy[:,1]-Yxyc[:,1])**2 + (Yxy[:,2]-Yxyc[:,2])**2)
            #ellipse_vs[i,:] = np.hstack((de.max(),de.min(),Yxyc[:,1],Yxyc[:,2],np.nan)) # nan because orientation is xy, but request is some other color space. Change later to actual angle when fitellipse() has been implemented

        
        # plot ellipses:
        if show == True:
            if (axh is None) & (i == 0):
                import matplotlib.pyplot as plt # lazy import
                fig = plt.figure()
                axh = fig.add_subplot(111)
            
            if (cspace_in is None):
                xlabel = 'x'
                ylabel = 'y'
            else:
                xlabel = _CSPACE_AXES[cspace_in][1]
                ylabel = _CSPACE_AXES[cspace_in][2]
            
            if (cspace_out is not None):
                xlabel = _CSPACE_AXES[cspace_out][1]
                ylabel = _CSPACE_AXES[cspace_out][2]
            
            if plot_center == True:
                axh.plot(Yxyc[:,1],Yxyc[:,2],color = center_color, linestyle = 'none', marker = center_marker, markersize = center_markersize)
            if llabel is None:
                axh.plot(Yxy[:,1],Yxy[:,2],color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_markersize)
            else:
                axh.plot(Yxy[:,1],Yxy[:,2],color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_markersize,label = llabel)
            
            axh.set_xlabel(xlabel, fontname = label_fontname, fontsize = label_fontsize)
            axh.set_ylabel(ylabel, fontname = label_fontname, fontsize = label_fontsize)
            if show_grid == True:
                axh.grid(True)
            #plt.show()     
    Yxys = np.transpose(Yxys,axes=(0,2,1))       
    if out is not None:
        return eval(out)
    else:
        return None

#------------------------------------------------------------------------------
def plot_chromaticity_diagram_colors(diagram_samples = 256, diagram_opacity = 1.0, diagram_lightness = 0.25, Y_SL = 100,\
                                      cieobs = _CIEOBS, cspace = 'Yxy', cspace_pars = {},\
                                      show = True, axh = None,\
                                      show_grid = False, label_fontname = 'Times New Roman', label_fontsize = 12,\
                                      **kwargs):
    """
    Plot the chromaticity diagram colors. 
    Only works / makes sense for Yxy, Yuv, luv (basically any chromaticity diagram where Y or lightness of spectrum locus is not relative that of xyzw, because what would be its value? These are lights!)
    
    Args:
        :diagram_samples:
            | 256, optional
            | Sampling resolution of color space.
        :diagram_opacity:
            | 1.0, optional
            | Sets opacity of chromaticity diagram
        :diagram_lightness:
            | 0.25, optional
            | Sets lightness of chromaticity diagram
        :Y_SL:
            | 100, optional
            | Normalize the XYZ of the spectrum locus to this value before converting to cspace.
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show:
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :cspace:
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :cspace_pars:
            | {} or dict, optional
            | Dict with parameters required by color space specified in :cspace: 
            | (for use with luxpy.colortf())
        :show_grid:
            | False, optional
            | Show grid (True) or not (False)
        :label_fontname: 
            | 'Times New Roman', optional
            | Sets font type of axis labels.
        :label_fontsize:
            | 12, optional
            | Sets font size of axis labels.
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
        
    Returns:
        
    """
        
    if isinstance(cieobs,str):
        SL = _CMF[cieobs]['bar'].copy()
    else:
        SL = cieobs.copy()
    wl, SL = SL[0], SL[1:4].T
    SL = Y_SL*SL/(SL[:,1,None] + _EPS) # normalize so that Y=Y_SL
    cnd = SL.sum(axis=1)>0
    SL = SL[cnd,:] # avoid div by zero in xyz-to-Yxy conversion
    wl = wl[cnd]
    SL = colortf(SL, tf = cspace, fwtf = cspace_pars)
    
    x_coord = SL[...,1]
    dx_coord = np.vstack((*np.diff(x_coord),0))
    plambdamax = np.where((wl>=600) & (dx_coord[:,0]<0))[0][0]

    SL = np.vstack((SL[:(plambdamax+1),:],SL[0])) # add lowest wavelength data and go to max of gamut in x (there is a reversal for some cmf set wavelengths >~700 nm!)
    Y,x,y = asplit(SL)
    SL = np.vstack((x,y)).T

    # create grid for conversion to srgb
    offset = _EPS
    min_x = min(offset,x.min())
    max_x = max(1,x.max())
    min_y = min(offset,y.min())
    max_y = max(1,y.max())
    #print(cspace, min_x,max_x,min_y,max_y)
    ii, jj = np.meshgrid(np.linspace(min_x - offset, max_x + offset, int(diagram_samples)), np.linspace(max_y + offset, min_y - offset, int(diagram_samples)))
    ij = np.dstack((ii, jj))
    ij[ij==0] = offset


    
    ij2D = ij.reshape((diagram_samples**2,2))
    _diagram_lightness = 1.0
    ij2D = np.hstack((100*np.ones((ij2D.shape[0],1)), ij2D))
    xyz = colortf(ij2D, tf = cspace + '>xyz', bwtf = cspace_pars)

    xyz[xyz < 0] = 0
    keepdims = True if (xyz.ndim == 3) else False
    xyz[np.isinf(xyz.sum(axis=1,keepdims=keepdims)),...] = np.nan
    xyz[np.isnan(xyz.sum(axis=1,keepdims=keepdims)),...] = offset

    Ymax = xyz[...,1].max()
    xyz = xyz/Ymax*100*diagram_lightness
    
    srgb = xyz_to_srgb(xyz)
    srgb = srgb/srgb.max()
    srgb = srgb.reshape((diagram_samples,diagram_samples,3))

    if show == True:
        if axh is None:
            import matplotlib.pyplot as plt # lazy import
            fig = plt.figure()
            axh = fig.add_subplot(111)
        from matplotlib.patches import Polygon # lazy import
        polygon = Polygon(SL, facecolor='none', edgecolor='none')
        axh.add_patch(polygon)
        image = axh.imshow(
            srgb,
            interpolation='bilinear',
            extent = (min_x, max_x, min_y-0.05, max_y),
            clip_path=None,
            alpha=diagram_opacity)
        image.set_clip_path(polygon)
        axh.plot(x,y, color = 'darkgray')
        if (cspace == 'Yxy') & (isinstance(cieobs,str)):
            axh.set_xlim([0,1])
            axh.set_ylim([0,1])
        elif ((cspace == 'Yuv') | (cspace == 'Yuv76')) & (isinstance(cieobs,str)):
            axh.set_xlim([0,0.6])
            axh.set_ylim([0,0.6])
        elif ((cspace == 'Yuv60')) & (isinstance(cieobs,str)):
            axh.set_xlim([0,0.65])
            axh.set_ylim([0,0.4])
        else:
            axh.set_xlim([None,None])
            axh.set_ylim([None,None])
        if (cspace is not None):
            xlabel = _CSPACE_AXES[cspace][1]
            ylabel = _CSPACE_AXES[cspace][2]
            if (label_fontname is not None) & (label_fontsize is not None):
                axh.set_xlabel(xlabel, fontname = label_fontname, fontsize = label_fontsize)
                axh.set_ylabel(ylabel, fontname = label_fontname, fontsize = label_fontsize)
                
        if show_grid == True:
            axh.grid(True)
        #plt.show()
    
        return axh
    else:
        return None

#------------------------------------------------------------------------------
def plot_spectrum_colors(spd = None, spdmax = None,\
                         wavelength_height = -0.05, wavelength_opacity = 1.0, wavelength_lightness = 1.0,\
                         cieobs = _CIEOBS, show = True, axh = None,\
                         show_grid = False,ylabel = 'Spectral intensity (a.u.)',xlim=None,\
                         **kwargs):
    """
    Plot the spectrum colors.
    
    Args:
        :spd:
            | None, optional
            | Spectrum
        :spdmax:
            | None, optional
            | max ylim is set at 1.05 or (1+abs(wavelength_height)*spdmax)
        :wavelength_opacity:
            | 1.0, optional
            | Sets opacity of wavelength rectangle.
        :wavelength_lightness:
            | 1.0, optional
            | Sets lightness of wavelength rectangle.
        :wavelength_height:
            | -0.05 or 'spd', optional
            | Determine wavelength bar height 
            | if not 'spd': x% of spd.max()
        :axh: 
            | None or axes handle, optional
            | Determines axes to plot data in.
            | None: make new figure.
        :show:
            | True or False, optional
            | Invoke matplotlib.pyplot.show() right after plotting
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines CMF set to calculate spectrum locus or other.
        :show_grid:
            | False, optional
            | Show grid (True) or not (False)
        :ylabel:
            | 'Spectral intensity (a.u.)' or str, optional
            | Set y-axis label.
        :xlim:
            | None, optional
            | list or ndarray with xlimits.
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
        
    Returns:
        
    """
    
    if isinstance(cieobs,str):
        cmfs = _CMF[cieobs]['bar'].copy()
    else:
        cmfs = cieobs.copy()
    cmfs = cmfs[:,cmfs[1:].sum(axis=0)>0] # avoid div by zero in xyz-to-Yxy conversion
    cmfs = cmfs[:,~np.isnan(cmfs.sum(axis=0))]
    
    wavs = cmfs[0] # changed from cmfs[0:1].T because this stopped working from around matplotlib version 3.4.1
    SL =  cmfs[1:4].T    
    
    srgb = xyz_to_srgb(wavelength_lightness*100*SL)
    srgb = srgb/srgb.max()
    
    if show == True:
        import matplotlib.pyplot as plt # lazy import
        if axh is None:
            fig = plt.figure()
            axh = fig.add_subplot(111)
         
        if (wavelength_height == 'spd') & (spd is not None):
            if spdmax is None:
                spdmax = np.nanmax(spd[1:,:])
            y_min, y_max = 0.0, spdmax*(1.05)
            if xlim is None:
                x_min, x_max = np.nanmin(spd[0,:]), np.nanmax(spd[0,:])
            else:
                x_min, x_max = xlim

            SLrect = np.vstack([
                (x_min, 0.0),
                spd.T,
                (x_max, 0.0),
                ])
            wavelength_height = y_max        
            spdmax = 1
        else:
            if (spdmax is None) & (spd is not None):
                spdmax = np.nanmax(spd[1:,:])
                y_min, y_max = wavelength_height*spdmax, spdmax*(1 + np.abs(wavelength_height))

            elif (spdmax is None) & (spd is None):
                spdmax = 1
                y_min, y_max = wavelength_height, 0
                
            elif (spdmax is not None):
                y_min, y_max = wavelength_height*spdmax, spdmax#*(1 + np.abs(wavelength_height))

                    
            if xlim is None:
                x_min, x_max = np.nanmin(wavs), np.nanmax(wavs)
            else:
                x_min, x_max = xlim
                
            SLrect = np.vstack([
                (x_min, 0.0),
                (x_min, wavelength_height*spdmax),
                (x_max, wavelength_height*spdmax),
                (x_max, 0.0),
                ])
        
        axh.set_xlim([x_min,x_max])
        axh.set_ylim([y_min,y_max])     

        from matplotlib.patches import Polygon # lazy import
        polygon = Polygon(SLrect, facecolor=None, edgecolor=None)
        axh.add_patch(polygon)
        padding = 0.1
        axh.bar(x = wavs - padding,
               height = wavelength_height*spdmax,
               width = 1 + padding,
               color = srgb,
               align = 'edge',
               linewidth = 0,
               clip_path = polygon) 
        
        if spd is not None:
            axh.plot(spd[0:1,:].T,spd[1:,:].T, color = 'k', label = 'spd')
 
        if show_grid == True:
            plt.grid(True)
        axh.set_xlabel('Wavelength (nm)',kwargs)
        axh.set_ylabel(ylabel, kwargs)        

        #plt.show()
    
        return axh
    else:
        return None
    
#------------------------------------------------------------------------------
def plot_rfl_color_patches(rfl, spd = None, cieobs = '1931_2', patch_shape = (100,100), patch_layout = None, ax = None, show = True):
    """
    Create (and plot) an image with colored patches representing a set of reflectance spectra illuminated by a specified illuminant.
    
    Args:
        :rfl:
            | ndarray with reflectance spectra
        :spd:
            | None, optional
            | ndarray with illuminant spectral power distribution
            | If None: _CIE_D65 is used.
        :cieobs:
            | '1931_2', optional
            | CIE standard observer to use when converting rfl to xyz.
        :patch_shape:
            | (100,100), optional
            | shape of each of the patches in the image
        :patch_layout:
            | None, optional
            | If None: layout is calculated automatically to give a 'good' aspect ratio
        :ax:
            | None, optional
            | Axes to plot the image in. If None: a new axes is created.
        :show:
            | True, optional
            | If True: plot image in axes and return axes handle; else: return ndarray with image.
            
    Return:
        :ax: or :imagae: 
            | Axes is returned if show == True, else: ndarray with rgb image is returned.
    """
    if spd is None:
        spd = _CIE_D65
    xyz = spd_to_xyz(spd, rfl = rfl, cieobs = cieobs)[:,0,:]
    rgb = xyz_to_srgb(xyz).astype('uint8')
    return plot_rgb_color_patches(rgb, ax  = ax, patch_shape = patch_shape, patch_layout = patch_layout, show = show)

def plot_rgb_color_patches(rgb, patch_shape = (100,100), patch_layout = None, ax = None, show = True):
    """
    Create (and plot) an image with patches with specified rgb values.
    
    Args:
        :rgb:
            | ndarray with rgb values for each of the patches
        :patch_shape:
            | (100,100), optional
            | shape of each of the patches in the image
        :patch_layout:
            | None, optional
            | If None: layout is calculated automatically to give a 'good' aspect ratio
        :ax:
            | None, optional
            | Axes to plot the image in. If None: a new axes is created.
        :show:
            | True, optional
            | If True: plot image in axes and return axes handle; else: return ndarray with image.
            
    Return:
        :ax: or :imagae: 
            | Axes is returned if show == True, else: ndarray with rgb image is returned.
    """
    if ax is None:
        import matplotlib.pyplot as plt # lazy import
        fig, ax = plt.subplots(1,1)
        
    if patch_layout is None:
        patch_layout = get_subplot_layout(rgb.shape[0])
    
    image = np.zeros(np.hstack((np.array(patch_shape)*np.array(patch_layout),3)))
    for i in range(rgb.shape[0]):
        r, c = np.unravel_index(i,patch_layout)
        R = int(r*patch_shape[0])
        C = int(c*patch_shape[1])
        image[R:R+patch_shape[0],C:C+patch_shape[1],:] = np.ones(np.hstack((patch_shape,3)))*rgb[i,None,:]

    if show == False:
        return image
    else:
        ax.imshow(image.astype('uint8'))
        ax.axis('off')
        return ax

def plot_cmfs(cmfs, cmf_symbols = ['x','y','z'], cmf_label = '', ylabel = 'Sensitivity',wavelength_bar=True, 
              colors = ['r','g','b'], axh = None, legend = True, **kwargs):
    """
    Plot CMFs.
    
    Args:
        :cmfs: 
            | ndarray with a set of CMFs.
        :cmf_symbols:
            | ['x,'y','z], optional
            | Symbols of the CMFs
            | If not a list but a string, the same label will be used for all CMF
            | and the same color will be used ('k' if colors is a list)
        :cmf_label:
            | '', optional
            | Additional label that will be added in front of the cmf symbols.
        :ylabel:
            | 'Sensitivity', optional
            | label for y-axis.
        :wavelength_bar:
            | True, optional
            | Add a colored wavelength bar with spectral colors.
        :colors:
            | ['r','g','b'], optional
            | Color for plotting each of the individual CMF.
        :axh:
            | None, optional
            | Axes to plot the image in. If None: a new axes is created.
        :kwargs:
            | additional kwargs for plt.plot().
            
    Returns:
        :axh:
            | figure axes handle.
    """
    if isinstance(cmf_symbols,list):
        cmf_symbols = [r'$\overline{'+cmf_symbols[i][0]+r'}'+cmf_symbols[i][1:]+r'(\lambda)$' for i in range(3)]
          
    else:
        cmf_symbols = [cmf_symbols,None,None]
        if isinstance(colors,list):
            colors = ['k']*3
        else:
            colors = [colors]*3
    if axh is None:
        import matplotlib.pyplot as plt # lazy import
        fig, axh = plt.subplots(1,1)

    for i in range(3):
        label = cmf_label + cmf_symbols[i] if cmf_symbols[i] is not None else None
        axh.plot(cmfs[0],cmfs[i+1],color = colors[i], label = label,**kwargs) 
        # try: 
        #     axh.plot(cmfs[0],cmfs[i+1],color = colors[i], label = label,**kwargs)  
        # except: # to deal with matplotlib font issue
        #     cmf_symbols[i] = cmf_symbols_no_mathtext[i]
        #     label = cmf_label + cmf_symbols[i] if cmf_symbols[i] is not None else None
        #     axh.plot(cmfs[0],cmfs[i+1],color = colors[i], label = label,**kwargs)
    
    if wavelength_bar == True:
        axh = plot_spectrum_colors(spd = None,spdmax = np.nanmax(cmfs[1:]), axh = axh, wavelength_height = -0.05)
    axh.set_xlabel('Wavelength (nm)')
    axh.set_ylabel(ylabel)    
    if legend == True:
        axh.legend()
    axh.grid(False)
    return axh