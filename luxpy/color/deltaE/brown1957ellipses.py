# -*- coding: utf-8 -*-
"""
Module for Brown color discrimination ellipses.
===============================================

 :get_brown1957_ellipse(): Estimate n-step Brown (1957) ellipse at CIE x,y coordinates.  
     
References:
  1. Brown, W. R. J. (1957). Color Discrimination of Twelve Observers*. Journal of the Optical Society of America, 47(2), 137–143. https://doi.org/10.1364/JOSA.47.000137


Created on Thu Jul 15 13:52:25 2021

@author: ksmet1877 [at] gmail.com
"""
import numpy as np

from luxpy import (math, plotSL, plotellipse)
from luxpy.utils import _EPS
eps = _EPS/10.0

brown1957 = {
            'color_center' : [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'fov' :          [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  2, 10, 10, 10, 10, 10, 10, 10, 10, 10],
			'x': [0.308, 0.339, 0.426, 0.536, 0.457, 0.288, 0.254, 0.674, 0.209, 0.340, 0.422, 0.262, 0.368, 0.368, 0.643, 0.385, 0.270, 0.308, 0.562, 0.217, 0.196, 0.161, 0.257],
			'y': [0.280, 0.493, 0.219, 0.411, 0.368, 0.630, 0.456, 0.308, 0.137, 0.178, 0.483, 0.343, 0.352, 0.352, 0.304, 0.546, 0.549, 0.307, 0.252, 0.606, 0.293, 0.410, 0.202],
			'Y_ftL': [4.6, 6.4, 4.5, 6.1, 5.1, 5.0, 4.4, 5.3, 3.0, 3.3, 4.4, 4.7, 6.3, 6.3, 4.1, 4.7, 5.0, 6.0, 4.0, 5.5, 4.9, 5.1, 4.3],
			'straight_averages' : {
									'a': [0.00303, 0.00530, 0.00346, 0.00385, 0.00264, 0.00606, 0.00300, 0.002180, 0.003220, 0.00398, 0.00545, 0.00186, 0.00199, 0.00320, 0.002720, 0.00388, 0.00473, 0.00186, 0.002490, 0.00635, 0.00235, 0.00344, 0.00362],
									'b': [0.00139, 0.00209, 0.00174, 0.00190, 0.00149, 0.00239, 0.00175, 0.000824, 0.000889, 0.00183, 0.00324, 0.00159, 0.00130, 0.00227, 0.000960, 0.00339, 0.00223, 0.00146, 0.000995, 0.00274, 0.00175, 0.00193, 0.00130],
									'theta': [64, 84, 43, 161, 171, 103, 110, 170, 79, 83, 123, 81, 171, 0, 177, 113, 125, 41, 13, 128, 7, 154, 25],
								   },
			'weighted_averages' : {
									'a': [0.00211, 0.00373, 0.00228, 0.00290, 0.00190, 0.00535, 0.00182, 0.00184, 0.00232, 0.00230, 0.00339, 0.00185, 0.00189, 0.00262, 0.00242, 0.00429, 0.00451, 0.00184, 0.00205, 0.00605, 0.00236, 0.00304, 0.00281],
									'b': [0.00108, 0.00157, 0.00131, 0.00145, 0.00098, 0.00205, 0.00102, 0.00070, 0.00074, 0.00108, 0.00212, 0.00147, 0.00104, 0.00168, 0.00090, 0.00288, 0.00204, 0.00143, 0.00086, 0.00245, 0.00148, 0.00155, 0.00102],
									'theta': [67, 88, 43, 169, 4, 104, 112, 169, 77, 65, 86, 47, 165, 159, 175, 128, 127, 35, 14, 128, 6, 158, 40],
								   }
			}

brown1957_weighted = np.vstack((np.array(brown1957['weighted_averages']['a'])*10,
                                np.array(brown1957['weighted_averages']['b'])*10,
                                brown1957['x'],
                                brown1957['y'],
                                np.array(brown1957['weighted_averages']['theta'])*1.0)).T

brown1957_straight = np.vstack((np.array(brown1957['straight_averages']['a'])*10,
                                np.array(brown1957['straight_averages']['b'])*10,
                                brown1957['x'],
                                brown1957['y'],
                                np.array(brown1957['straight_averages']['theta'])*1.0)).T

__all__ = ['get_brown1957_ellipse','brown1957']



def get_brown1957_ellipse(xy = None, weighted = True, k_neighbours = 3, nsteps = 10, average_cik = True):
    """
    Estimate n-step Brown1957 ellipse at CIE x,y coordinates xy by calculating 
    average inverse covariance ellipse of the k_neighbours closest ellipses.
    
    Args:
        :xy:
            | None or ndarray, optional
            | If None: output Brown1957 ellipses, if not None: xy are the 
            | CIE xy coordinates for which ellipses will be estimated.
        :weighted:
            | True, optional
            | If True: use weighted averages from Table III in Brown 1957 paper, else use the straight averages.
        :k_neighbours:
            | 3, optional
            | Number of nearest ellipses to use to calculate ellipse at xy
        :nsteps:
            | 10, optional
            | Set number of steps of ellipse.
        :average_cik:
            | True, optional
            | If True: take distance weighted average of inverse 
            |   'covariance ellipse' elements cik. 
            | If False: average major & minor axis lengths and 
            |   ellipse orientation angles directly.
            
    Returns:
        :v_brown_est:
            | estimated Brown1957 ellipse(s) in v-format [Rmax,Rmin,xc,yc,theta]
    
    References:
        1. Brown, W.R.J. (1957). Color Discrimination of Twelve Observers*. Journal of the Optical Society of America, 47(2), 137–143. https://doi.org/10.1364/JOSA.47.000137
    """
    # Create list of 22 Brown ellipses (x10)
    v_brown1957 = brown1957_weighted.copy() if weighted else brown1957_straight.copy()
    
    # convert last column to rad.:
    v_brown1957[:,-1] = v_brown1957[:,-1]*np.pi/180
    
    # convert to desired number of brown1957-steps:
    v_brown1957[:,0:2] = v_brown1957[:,0:2]/10*nsteps
    
    if xy is not None:
        #calculate inverse covariance matrices:
        cik = math.v_to_cik(v_brown1957, inverse = True)
        if average_cik == True:
            cik_long = np.hstack((cik[:,0,:],cik[:,1,:]))
        
        # Calculate k_neighbours closest ellipses to xy:
        import scipy # lazy import
        tree = scipy.spatial.cKDTree(v_brown1957[:,2:4], copy_data = True)
        d, inds = tree.query(xy, k = k_neighbours)
    
        if k_neighbours  > 1:
            pd = 1
            w = (1.0 / (np.abs(d) + eps)**pd)[:,:,None] # inverse distance weigthing
            if average_cik == True:
                cik_long_est = np.sum(w * cik_long[inds,:], axis=1) / np.sum(w, axis=1)
            else:
                v_brown1957_est = np.sum(w * v_brown1957[inds,:], axis=1) / np.sum(w, axis=1) # for average xyc

        else:
            v_brown1957_est = v_brown1957[inds,:].copy()
        
        # convert cik back to v:
        if (average_cik == True) & (k_neighbours >1):
            cik_est = np.dstack((cik_long_est[:,0:2],cik_long_est[:,2:4]))
            v_brown1957_est = math.cik_to_v(cik_est, inverse = True)
        v_brown1957_est[:,2:4] = xy
    else:
        v_brown1957_est = v_brown1957
        
    return v_brown1957_est




if __name__ == '__main__':
    
    # Get brown1957 ellipses:
    v_brown1957_weighted = get_brown1957_ellipse(xy = None, weighted = True)
    v_brown1957_straight = get_brown1957_ellipse(xy = None, weighted = False)
    
    # Estimate brown1957 ellipse at test xy:
    xy_test = np.asarray([[1/2,1/3],[1/3,1/3]])
    
    v_brown1957_est = get_brown1957_ellipse(xy_test, weighted = True)

    # Plot results:
    cspace = 'Yuv'
    #axh = plot_chromaticity_diagram_colors(cspace = cspace)
    axh = plotSL(cspace = cspace, cieobs = '1931_2', show = False, diagram_colors = True)
    axh = plotellipse(v_brown1957_weighted, show = True, axh = axh, cspace_out = cspace,plot_center = False, center_color = 'k', out = 'axh', line_style = ':', line_color ='k',line_width = 1.5)
    #axh = plotellipse(v_brown1957_straight, show = True, axh = axh, cspace_out = cspace,plot_center = False, center_color = 'w', out = 'axh', line_style = ':', line_color ='w',line_width = 1.5)
    #plotellipse(v_brown1957_est, show = True, axh = axh, cspace_out = cspace,line_color = 'k', plot_center = True, center_color = 'k')
    if cspace == 'Yuv':
        axh.set_xlim([0,0.6])
        axh.set_ylim([0,0.6])
    
    