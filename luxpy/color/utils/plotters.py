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
 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import np, plt, math, _EPS, _CIEOBS, _CSPACE, _CSPACE_AXES, _CIE_ILLUMINANTS, _CMF, daylightlocus, colortf, Yxy_to_xyz, asplit, spd_to_xyz, cri_ref, xyz_to_srgb

from matplotlib.patches import Polygon

__all__ = ['plotSL','plotDL','plotBB','plot_color_data','plotceruleanline','plotUH','plotcircle','plotellipse','plot_chromaticity_diagram_colors','plot_spectrum_colors']



def plot_color_data(x,y,z=None, axh=None, show = True, cieobs =_CIEOBS, \
                    cspace = _CSPACE,  formatstr = 'k-', **kwargs):
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
            | luxpy._CSPACE or str, optional
            | Determines color space / chromaticity diagram to plot data in.
            | Note that data is expected to be in specified :cspace:
        :formatstr: 
            | 'k-' or str, optional
            | Format str for plotting (see ?matplotlib.pyplot.plot)
        :kwargs:
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | None (:show: == True) 
            |  or 
            | handle to current axes (:show: == False)
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if 'grid' in kwargs.keys():
        plt.grid(kwargs['grid']);kwargs.pop('grid')
    if z is not None:
        z = np.atleast_1d(z)
        if axh is None:
            fig = plt.figure()
            axh = plt.axes(projection='3d')
        axh.plot3D(x,y,z,formatstr, linewidth = 2,**kwargs)
        plt.xlabel(_CSPACE_AXES[cspace][0], kwargs)
    else:
        plt.plot(x,y,formatstr,linewidth = 2,**kwargs)
        
    plt.xlabel(_CSPACE_AXES[cspace][1], kwargs)
    plt.ylabel(_CSPACE_AXES[cspace][2], kwargs)
    if 'label' in kwargs.keys():
        plt.legend()
    if show == True:
        plt.show()
    else:
        return plt.gca()



def plotDL(ccts = None, cieobs =_CIEOBS, cspace = _CSPACE, axh = None, \
           show = True, force_daylight_below4000K = False, cspace_pars = {}, \
           formatstr = 'k-',  **kwargs):
    """
    Plot daylight locus.
    
    Args: 
        :ccts: 
            | None or list[float], optional
            | None defaults to [4000 K to 1e19 K] in 100 steps on a log10 scale.
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
              (for use with luxpy.colortf())
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | None (:show: == True) 
            |  or 
            | handle to current axes (:show: == False)
    """
    
    if ccts is None:
        ccts = 10**np.linspace(np.log10(4000.0),np.log10(10.0**19.0),100.0)
        
    xD,yD = daylightlocus(ccts, force_daylight_below4000K = force_daylight_below4000K)
    Y = 100*np.ones(xD.shape)
    DL =  Yxy_to_xyz(np.vstack((Y, xD,yD)).T)
    DL = colortf(DL, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(DL)
    
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    
    
    if show == False:
        return axh
    
def plotBB(ccts = None, cieobs =_CIEOBS, cspace = _CSPACE, axh = None, cctlabels = True, show = True, cspace_pars = {}, formatstr = 'k-',  **kwargs):  
    """
    Plot blackbody locus.
        
    Args: 
        :ccts: 
            | None or list[float], optional
            | None defaults to [1000 to 1e19 K].
            | Range: 
            |     [1000,1500,2000,2500,3000,3500,4000,5000,6000,8000,10000] 
            |    + [15000 K to 1e19 K] in 100 steps on a log10 scale
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
              (for use with luxpy.colortf())
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | None (:show: == True) 
            |  or 
            | handle to current axes (:show: == False)
    """
    if ccts is None:
        ccts1 = np.array([1000.0,1500.0,2000.0,2500.0,3000.0,3500.0,4000.0,5000.0,6000.0,8000.0,10000.0])
        ccts2 = 10**np.linspace(np.log10(15000.0),np.log10(10.0**19.0),100.0)
        ccts = np.hstack((ccts1,ccts2))
    else:
        ccts1 = None
    
    BB = cri_ref(ccts,ref_type='BB')
    xyz = spd_to_xyz(BB,cieobs = cieobs)
    Yxy = colortf(xyz, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(Yxy)
   
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    

    if (cctlabels == True) & (ccts1 is not None):
        for i in range(ccts1.shape[0]):
            if ccts1[i]>= 3000.0:
                if i%2 == 0.0:
                    plt.plot(x[i],y[i],'k+', color = '0.5')
                    plt.text(x[i]*1.05,y[i]*0.95,'{:1.0f}K'.format(ccts1[i]), color = '0.5')
        plt.plot(x[-1],y[-1],'k+', color = '0.5')
        plt.text(x[-1]*1.05,y[-1]*0.95,'{:1.3e}K'.format(ccts[-1]), color = '0.5')    
    if show == False:
        return axh
    
def plotSL(cieobs =_CIEOBS, cspace = _CSPACE, DL = True, BBL = True, D65 = False,\
           EEW = False, cctlabels = False, axh = None, show = True,\
           cspace_pars = {}, formatstr = 'k-',\
           diagram_colors = False, diagram_samples = 100, diagram_opacity = 1.0,\
           diagram_lightness = 0.25,\
           **kwargs):
    """
    Plot spectrum locus for cieobs in cspace.
    
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
        :kwargs: 
            | additional keyword arguments for use with matplotlib.pyplot.
    
    Returns:
        :returns: 
            | None (:show: == True) 
            |  or 
            | handle to current axes (:show: == False)
    """
    SL = _CMF[cieobs]['bar'][1:4].T
    SL = np.vstack((SL,SL[0]))
    SL = 100.0*SL/SL[:,1,None]
    SL = colortf(SL, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(SL)
    
    showcopy = show
    if np.any([DL,BBL,D65,EEW]):
        show = False

    axh_ = plot_chromaticity_diagram_colors(axh = axh, show = diagram_colors, cspace = cspace, cieobs = cieobs,\
                                            cspace_pars = cspace_pars,\
                                            diagram_samples = diagram_samples,\
                                            diagram_opacity = diagram_opacity,\
                                            diagram_lightness = diagram_lightness,\
                                            label_fontname = None, label_fontsize = None)
        
    axh_ = plot_color_data(x,y,axh = axh_, cieobs = cieobs, cspace = cspace, show = show, formatstr=formatstr,  **kwargs)


    if DL == True:
        if 'label' in kwargs.keys(): # avoid label also being used for DL
            kwargs.pop('label')
        plotDL(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, formatstr = 'k:',  **kwargs)
    if BBL == True:
        if 'label' in kwargs.keys(): # avoid label also being used for BB
            kwargs.pop('label')
        plotBB(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, cctlabels = cctlabels, formatstr = 'k-.',  **kwargs)
    
    if D65 == True:
        YxyD65 = colortf(spd_to_xyz(_CIE_ILLUMINANTS['D65']), tf = cspace, tfa0 = cspace_pars)
        plt.plot(YxyD65[...,1],YxyD65[...,2],'bo')
    if EEW == True:
        YxyEEW = colortf(spd_to_xyz(_CIE_ILLUMINANTS['E']), tf = cspace, tfa0 = cspace_pars)
        plt.plot(YxyEEW[...,1],YxyEEW[...,2],'ko')
    
    
    
    if showcopy == False:
        return axh_
    else:
        plt.show()
        
        
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
    cmf = _CMF[cieobs]['bar']
    p_y = cmf[0] == 577.0 #Kuehni, CRA 2013 (mean, table IV)
    p_b = cmf[0] == 472.0 #Kuehni, CRA 2013 (mean, table IV)
    xyz_y = cmf[1:,p_y].T
    xyz_b = cmf[1:,p_b].T
    lab = colortf(np.vstack((xyz_b,xyz_y)),tf = cspace, tfa0 = cspace_pars)
    if axh is None:
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
    cmf = _CMF[cieobs]['bar']
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
        axh = plt.gca()
    for huenr in uhues:
        lab = colortf(np.vstack((xyz0,xyz_uh[huenr])),tf = cspace, tfa0 = cspace_pars)
        huh = axh.plot(lab[:,1],lab[:,2],formatstr[huenr],label = excludefromlegend + 'Unique '+ hues[huenr])
        huniquehues = [huniquehues,huh]
    return  huniquehues

#------------------------------------------------------------------------------
def plotcircle(center = np.array([0.,0.]),radii = np.arange(0,60,10), angles = np.arange(0,350,10),color = 'k',linestyle = '--', out = None):
    """
    Plot one or more concentric circles.
    
    Args:
        :center: 
            | np.array([0.,0.]) or ndarray with center coordinates, optional
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
    for ri in radii:
        x = ri*np.cos(angles*np.pi/180)
        y = ri*np.sin(angles*np.pi/180)
        xs = np.hstack((xs,x))
        ys = np.hstack((ys,y))
        if out != 'x,y':
            plt.plot(x,y,color = color, linestyle = linestyle)
    if out == 'x,y':
        return xs,ys

#------------------------------------------------------------------------------
def plotellipse(v, cspace_in = 'Yxy', cspace_out = None, nsamples = 100, \
                show = True, axh = None, \
                line_color = 'darkgray', line_style = ':', line_width = 1, line_marker = '', line_markersize = 4,\
                plot_center = False, center_marker = 'o', center_color = 'darkgray', center_markersize = 4,\
                show_grid = True, label_fontname = 'Times New Roman', label_fontsize = 12,\
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
            | True, optional
            | Show grid (True) or not (False)
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
        t = np.linspace(0, 2*np.pi, nsamples)
        
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
                fig = plt.figure()
                axh = fig.add_subplot(111)
            
            if (cspace_in is None):
                xlabel = 'x'
                ylabel = 'y'
            
            if (cspace_out is not None):
                xlabel = _CSPACE_AXES[cspace_out][1]
                ylabel = _CSPACE_AXES[cspace_out][2]
            
            if plot_center == True:
                plt.plot(Yxyc[:,1],Yxyc[:,2],color = center_color, linestyle = 'none', marker = center_marker, markersize = center_markersize)

            plt.plot(Yxy[:,1],Yxy[:,2],color = line_color, linestyle = line_style, linewidth = line_width, marker = line_marker, markersize = line_markersize)
            plt.xlabel(xlabel, fontname = label_fontname, fontsize = label_fontsize)
            plt.ylabel(ylabel, fontname = label_fontname, fontsize = label_fontsize)
            if show_grid == True:
                plt.grid()
            #plt.show()     
    Yxys = np.transpose(Yxys,axes=(0,2,1))       
    if out is not None:
        return eval(out)
    else:
        return None

#------------------------------------------------------------------------------
def plot_chromaticity_diagram_colors(samples = 256, diagram_opacity = 1.0, diagram_lightness = 0.25,\
                                      cieobs = _CIEOBS, cspace = 'Yxy', cspace_pars = {},\
                                      show = True, axh = None,\
                                      show_grid = True, label_fontname = 'Times New Roman', label_fontsize = 12,\
                                      **kwargs):
    """
    Plot the chromaticity diagram colors.
    
    Args:
        :samples:
            | 256, optional
            | Sampling resolution of color space.
        :diagram_opacity:
            | 1.0, optional
            | Sets opacity of chromaticity diagram
        :diagram_lightness:
            | 0.25, optional
            | Sets lightness of chromaticity diagram
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
            | True, optional
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
    offset = _EPS
    ii, jj = np.meshgrid(np.linspace(offset, 1 + offset, samples), np.linspace(1+offset, offset, samples))
    ij = np.dstack((ii, jj))
    
    SL =  _CMF[cieobs]['bar'][1:4].T
    SL = np.vstack((SL,SL[0]))
    SL = 100.0*SL/SL[:,1,None]
    SL = colortf(SL, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(SL)
    SL = np.vstack((x,y)).T

    
    ij2D = ij.reshape((samples**2,2))
    ij2D = np.hstack((diagram_lightness*100*np.ones((ij2D.shape[0],1)), ij2D))
    xyz = colortf(ij2D, tf = cspace + '>xyz', tfa0 = cspace_pars)

    xyz[xyz < 0] = 0
    xyz[np.isinf(xyz.sum(axis=1)),:] = np.nan
    xyz[np.isnan(xyz.sum(axis=1)),:] = offset
    
    srgb = xyz_to_srgb(xyz)
    srgb = srgb/srgb.max()
    srgb = srgb.reshape((samples,samples,3))

    if show == True:
        if axh is None:
            fig = plt.figure()
            axh = fig.add_subplot(111)
        polygon = Polygon(SL, facecolor='none', edgecolor='none')
        axh.add_patch(polygon)
        image = axh.imshow(
            srgb,
            interpolation='bilinear',
            extent = (0.0, 1, -0.05, 1),
            clip_path=None,
            alpha=diagram_opacity)
        image.set_clip_path(polygon)
        plt.plot(x,y, color = 'darkgray')
        if cspace == 'Yxy':
            plt.xlim([0,1])
            plt.ylim([0,1])
        elif cspace == 'Yuv':
            plt.xlim([0,0.6])
            plt.ylim([0,0.6])
        if (cspace is not None):
            xlabel = _CSPACE_AXES[cspace][1]
            ylabel = _CSPACE_AXES[cspace][2]
            if (label_fontname is not None) & (label_fontsize is not None):
                plt.xlabel(xlabel, fontname = label_fontname, fontsize = label_fontsize)
                plt.ylabel(ylabel, fontname = label_fontname, fontsize = label_fontsize)
                
        if show_grid == True:
            plt.grid()
        #plt.show()
    
        return axh
    else:
        return None

#------------------------------------------------------------------------------
def plot_spectrum_colors(spd = None, spdmax = None,\
                         wavelength_height = -0.05, wavelength_opacity = 1.0, wavelength_lightness = 1.0,\
                         cieobs = _CIEOBS, show = True, axh = None,\
                         show_grid = True,ylabel = 'Spectral intensity (a.u.)',xlim=None,\
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
            | True, optional
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
    
    cmfs = _CMF[cieobs]['bar']
    
    wavs = cmfs[0:1].T
    SL =  cmfs[1:4].T    
    
    srgb = xyz_to_srgb(wavelength_lightness*100*SL)
    srgb = srgb/srgb.max()
    
    if show == True:
        if axh is None:
            fig = plt.figure()
            axh = fig.add_subplot(111)
         
        if (wavelength_height == 'spd') & (spd is not None):
            if spdmax is None:
                spdmax = np.nanmax(spd[1:,:])
            y_min, y_max = 0.0, spdmax*(1.05)
            if xlim is None:
                x_min, x_max = spd[0,:].min(), spd[0,:].max()
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
                x_min, x_max = wavs.min(), wavs.max()
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
            plt.grid()
        axh.set_xlabel('Wavelength (nm)',kwargs)
        axh.set_ylabel(ylabel, kwargs)        

        #plt.show()
    
        return axh
    else:
        return None

