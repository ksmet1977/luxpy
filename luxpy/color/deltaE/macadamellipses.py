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
Module for MacAdam ellipses
==================================================================


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import (np, plt, _CIEOBS,_CSPACE_AXES, _CMF, _EPS, colortf, cKDTree, Yxy_to_xyz, xyz_to_srgb, asplit )

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

def v_to_cik(v):
    """
    Calculate 2x2 'covariance matrix' elements cik 
    
    Args:
        :v: 
            | (Nx5) np.ndarray
            | ellipse parameters [Rmax,Rmin,xc,yc,theta]
    
    Returns:
        :cik: 
            '2x2xN' covariance matrix
    
    Notes:
        | cik is not actually a covariance matrix,
        | only for a Gaussian or normal distribution!

    """
    v = np.atleast_2d(v)
    g11 = (1/v[:,0]*np.cos(v[:,4]))**2 + (1/v[:,1]*np.sin(v[:,4]))**2
    g22 = (1/v[:,0]*np.sin(v[:,4]))**2 + (1/v[:,1]*np.cos(v[:,4]))**2
    g12 = (1/v[:,0]**2 - 1/v[:,1]**2)*np.sin(v[:,4])*np.cos(v[:,4])
    cik = np.zeros((g11.shape[0],2,2))

    for i in np.arange(g11.shape[0]):
        cik[i,:,:] = np.vstack((np.hstack((g11[i],g12[i])), np.hstack((g12[i],g22[i]))))
    
    return cik

def cik_to_v(cik, xyc = None):
    """
    Calculate v-format ellipse descriptor from 2x2 'covariance matrix' cik 
    
    Args:
        :cik: 
            '2x2xN' covariance matrix
            
    Returns:
        :v: 
            | (Nx5) np.ndarray
            | ellipse parameters [Rmax,Rmin,xc,yc,theta]

    Notes:
        | cik is not actually a covariance matrix,
        | only for a Gaussian or normal distribution!

    """
    g11 = cik[:,0,0]
    g22 = cik[:,1,1] 
    g12 = cik[:,0,1]

    theta2 = 1/2*np.arctan2(2*g12,(g11-g22))
    theta = theta2 + (np.pi/2)*(g12<0)
    theta2 = theta
    cottheta = np.cos(theta)/np.sin(theta) #np.cot(theta)
    cottheta[np.isinf(cottheta)] = 0

    a = 1/np.sqrt((g22 + g12*cottheta))
    b = 1/np.sqrt((g11 - g12*cottheta))

    v = np.vstack((a, b, np.zeros(a.shape), np.zeros(a.shape), theta)).T
    
    # add center coordinates:
    if xyc is not None:
        v[:,2:4] = xyc
    
    return v

def macadam_ellipses(xy = None, k_neighbours = 3, nsteps = 10, average_cik = True):
    """
    Estimate n-step MacAdam ellipse at CIE x,y coordinates xy.
    
    Args:
        :xy:
            | None or ndarray, optional
            | If None: output Macadam ellipses, if not None: xy are the 
            | CIE xy coordinates for which ellipses will be estimated.
            
        :k_neighbours:
            | 3, optional
            | Number of nearest ellipses to use to calculate ellipse at xy
        :nsteps:
            | 10, optional
            | Set number of MacAdam steps of ellipse.
        :average_cik:
            | True, optional
            | If True: take distance weighted average of 'covarience ellipse'
            |   elements cik. If False, average major & minor axis lengths and 
            |   ellipse orientation angle directly.
            
    Returns:
        :v_mac_est:
            | estimated MacAdam ellipse(s) in v-format [Rmax,Rmin,xc,yc,theta]
            
    """
    # list of MacAdam ellipses (x10)
    v_mac = np.atleast_2d([
    [0.16,	0.057,	0.0085,	0.0035,	62.5],
    [0.187,	0.118,	0.022,	0.0055,	77],
    [0.253,	0.125,	0.025,	0.005,	55.5],
    [0.15,	0.68,	   0.096,	0.023,	105],
    [0.131,	0.521,	0.047,	0.02,	   112.5],
    [0.212,	0.55,	   0.058,	0.023,	100],
    [0.258,	0.45,	   0.05,	   0.02,	   92],
    [0.152,	0.365,	0.038,	0.019,	110],
    [0.28,	0.385,	0.04,	   0.015,	75.5],
    [0.38,	0.498,	0.044,	0.012,	70],
    [0.16,	0.2,	   0.021,	0.0095,	104],
    [0.228,	0.25,	   0.031,	0.009,	72],
    [0.305,	0.323,	0.023,	0.009,	58],
    [0.385,	0.393,	0.038,	0.016,	65.5],
    [0.472,	0.399,	0.032,	0.014,	51],
    [0.527,	0.35,	   0.026,	0.013,	20],
    [0.475,	0.3,	   0.029,	0.011,	28.5],
    [0.51,	0.236,	0.024,	0.012,	29.5],
    [0.596,	0.283,	0.026,	0.013,	13],
    [0.344,	0.284,	0.023,	0.009,	60],
    [0.39,	0.237,	0.025,	0.01,	   47],
    [0.441,	0.198,	0.028,	0.0095,	34.5],
    [0.278,	0.223,	0.024,	0.0055,	57.5],
    [0.3,	0.163,	0.029,	0.006,	54],
    [0.365,	0.153,	0.036,	0.0095,	40]
    ])
    
    # convert to v-format ([a,b, xc, yc, theta]):
    v_mac = v_mac[:,[2,3,0,1,4]]
    
    # convert last column to rad.:
    v_mac[:,-1] = v_mac[:,-1]*np.pi/180
    
    # convert to desired number of MacAdam-steps:
    v_mac[:,0:2] = v_mac[:,0:2]/10*nsteps
    
    if xy is not None:
        #calculate covariance matrices:
        cik = v_to_cik(v_mac)
        if average_cik == True:
            cik_long = np.hstack((cik[:,0,:],cik[:,1,:]))
        
        # Calculate k_neighbours closest ellipses to xy:
        tree = cKDTree(v_mac[:,2:4], copy_data = True)
        d, inds = tree.query(xy, k = k_neighbours )
    
        if k_neighbours  > 1:
            w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
            v_mac_est = np.sum(w * v_mac[inds,:], axis=1) / np.sum(w, axis=1) # for average xyc
            if average_cik == True:
                cik_long_est = np.sum(w * cik_long[inds,:], axis=1) / np.sum(w, axis=1)
        else:
            v_mac_est = v_mac[inds,:].copy()
        
        # convert cik back to v:
        cik_est = np.dstack((cik_long_est[:,0:2],cik_long_est[:,2:4]))
        if average_cik == True:
            v_mac_est = cik_to_v(cik_est, xyc = v_mac_est[:,2:4])
    else:
        v_mac_est = v_mac
        
    return v_mac_est

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
            plt.show()     
            
    if out is not None:
        return eval(out)
    else:
        return None


def plot_chromaticity_diagram_colours(samples = 256, diagram_opacity = 1.0, diagram_lightness = 0.25,\
                                      cieobs = _CIEOBS, cspace = 'Yxy', cspace_pars = {},\
                                      show = True, axh = None,\
                                      show_grid = True, label_fontname = 'Times New Roman', label_fontsize = 12,\
                                      **kwargs):
    """
    Plot the chromaticity diagram colours.
    
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
    spectral_locus = np.vstack((x,y)).T

    
    ij2D = ij.reshape((samples**2,2))
    ij2D = np.hstack((diagram_lightness*100*np.ones((ij2D.shape[0],1)), ij2D))
    xyz = colortf(ij2D, tf = cspace + '>xyz', tfa0 = cspace_pars)

    xyz[xyz < 0] = 0
    xyz[np.isinf(xyz.sum(axis=1)),:] = np.nan
    xyz[np.isnan(xyz.sum(axis=1)),:] = offset
    
    srgb = xyz_to_srgb(xyz)
    srgb = srgb/srgb.max()
    srgb = srgb.reshape((samples,samples,3))

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.imshow

    if show == True:
        if axh is None:
            fig = plt.figure()
            axh = fig.add_subplot(111)
        polygon = Polygon(spectral_locus, facecolor='none', edgecolor='none')
        axh.add_patch(polygon)
        image = axh.imshow(
            srgb,
            interpolation='bilinear',
            extent = (0.0, 1, -0.05, 1),
            clip_path=None,
            alpha=diagram_opacity)
        image.set_clip_path(polygon)
        plt.plot(x,y, color = 'darkgray')
        plt.xlim([0,1])
        plt.ylim([0,1])
        if (cspace is not None):
            xlabel = _CSPACE_AXES[cspace][1]
            ylabel = _CSPACE_AXES[cspace][2]
            plt.xlabel(xlabel, fontname = label_fontname, fontsize = label_fontsize)
            plt.ylabel(ylabel, fontname = label_fontname, fontsize = label_fontsize)
        if show_grid == True:
            plt.grid()
        plt.show()
    
        return axh
    else:
        return None

def plot_chrom():
    x = np.asarray([0.175596, 0.172787, 0.170806, 0.170085, 0.160343, 0.146958, 0.139149,\
                    0.133536, 0.126688, 0.115830, 0.109616, 0.099146, 0.091310, 0.078130,\
                    0.068717, 0.054675, 0.040763, 0.027497, 0.016270, 0.008169, 0.004876,\
                    0.003983, 0.003859, 0.004646, 0.007988, 0.013870, 0.022244, 0.027273,\
                    0.032820, 0.038851, 0.045327, 0.052175, 0.059323, 0.066713, 0.074299,\
                    0.089937, 0.114155, 0.138695, 0.154714, 0.192865, 0.229607, 0.265760,\
                    0.301588, 0.337346, 0.373083, 0.408717, 0.444043, 0.478755, 0.512467,\
                    0.544767, 0.575132, 0.602914, 0.627018, 0.648215, 0.665746, 0.680061,\
                    0.691487, 0.700589, 0.707901, 0.714015, 0.719017, 0.723016, 0.734674])
    y = np.asarray([ 0.005295, 0.004800, 0.005472, 0.005976, 0.014496, 0.026643, 0.035211,\
                    0.042704, 0.053441, 0.073601, 0.086866, 0.112037, 0.132737, 0.170464,\
                    0.200773, 0.254155, 0.317049, 0.387997, 0.463035, 0.538504, 0.587196,\
                    0.610526, 0.654897, 0.675970, 0.715407, 0.750246, 0.779682, 0.792153,\
                    0.802971, 0.812059, 0.819430, 0.825200, 0.829460, 0.832306, 0.833833,\
                    0.833316, 0.826231, 0.814796, 0.805884, 0.781648, 0.754347, 0.724342,\
                    0.692326, 0.658867, 0.624470, 0.589626, 0.554734, 0.520222, 0.486611,\
                    0.454454, 0.424252, 0.396516, 0.372510, 0.351413, 0.334028, 0.319765,\
                    0.308359, 0.299317, 0.292044, 0.285945, 0.280951, 0.276964, 0.265326])
    N = x.shape[0]
    i = 1
    e = 1/3
    steps = 25
    xy4rgb = np.zeros((N*steps*4, 5))
    for w in np.arange(N):                              # wavelength
        w2 = np.mod(w,N) + 1
        a1 = np.arctan2(y[w] - e, x[w] - e)             # start angle
        a2 = np.arctan2(y[w2] - e, x[w2] - e)           # end angle
        r1 = ((x[w] - e)**2 + (y[w] - e)**2)**0.5       # start radius
        r2 = ((x[w2] - e)**2 + (y[w2] - e)**2)**0.5     # end radius
        xyz = np.zeros((4,3))
        for c in np.arange(steps):                      # colorfulness
            # patch polygon
            xyz[0,0] = e + r1*np.cos(a1)*c/steps
            xyz[0,1] = e + r1*np.sin(a1)*c/steps
            xyz[0,2] = 1 - xyz[0,0] - xyz[0,1]
            xyz[1,0] = e + r1*np.cos(a1)*(c-1)/steps
            xyz[1,1] = e + r1*np.sin(a1)*(c-1)/steps
            xyz[1,2] = 1 - xyz[1,0] - xyz[1,1]
            xyz[2,0] = e + r2*np.cos(a2)*(c-1)/steps
            xyz[2,1] = e + r2*np.sin(a2)*(c-1)/steps
            xyz[2,2] = 1 - xyz[2,0] - xyz[2,1]
            xyz[3,0] = e + r2*np.cos(a2)*c/steps
            xyz[3,1] = e + r2*np.sin(a2)*c/steps
            xyz[3,2] = 1 - xyz[3,0] - xyz[3,1]
            # compute sRGB for vertices
            rgb = xyz_to_srgb(xyz)
            # store the results
            xy4rgb[i:i+2,0:2] = xyz[:,0:2]
            xy4rgb[i:i+2,2:5] = rgb
            i = i + 4


    rows = xy4rgb.shape[0]
    f = [1, 2, 3, 4]
    v = zeros((4,3))
#    for i = 1:4:rows
#        v(:,1:2) = xy4rgb(i:i+3,1:2)
#        patch('Vertices',v, 'Faces',f, 'EdgeColor','none', ...
#            'FaceVertexCData',xy4rgb[i:i+3,3:5],'FaceColor','interp')



if __name__ == '__main__':
    
    # Get MacAdam ellipses:
    v_mac = macadam_ellipses(xy = None)
    
    # Estimate MacAdam ellipse at test xy:
    xy_test = np.asarray([[1/2,1/3],[1/3,1/3]])
    v_mac_est = macadam_ellipses(xy_test)

    # Plot results:
    cspace = 'Yuv'
    axh = plot_chromaticity_diagram_colours(cspace = cspace)
    axh = plotellipse(v_mac, show = True, axh = axh, cspace_out = cspace,plot_center = False, center_color = 'r', out = 'axh', line_style = ':', line_color ='k',line_width = 1.5)
    plotellipse(v_mac_est, show = True, axh = axh, cspace_out = cspace,line_color = 'k', plot_center = True, center_color = 'k')
    if cspace == 'Yuv':
        axh.set_xlim([0,0.6])
        axh.set_ylim([0,0.6])
    
    
    
