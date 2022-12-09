# -*- coding: utf-8 -*-
"""
Module for generation of RGB training and XYZ test sets for display colorimetric characterization
-------------------------------------------------------------------------------------------------

:generate_training_data(): Generate RGB training pairs by creating a cube of RGB values. 

:generate_test_data(): Generate XYZ test values by creating a cube of CIELAB L*a*b* values, then converting these to XYZ values. 

:plot_rgb_xyz_lab_of_set(): Make 3d-plots of the RGB, XYZ and L*a*b* cubes of the data in rgb_xyz_lab. 

:split_ramps_from_cube(): Split a cube data set in pure RGB (ramps) and non-pure (remainder of cube). 

:is_random_sampling_of_pure_rgbs(): Return boolean indicating if the RGB cube axes (=single channel ramps) are sampled (different increment) independently from the remainder of the cube.

Created on Fri Oct 28 13:16:50 2022

@author: u0032318
"""
import copy
import warnings
import itertools

import numpy as np 
import matplotlib.pyplot as plt

from luxpy import xyz_to_lab, lab_to_xyz, spd_to_xyz, _CIE_D65, _CIEOBS
from luxpy import math

import dispcal as dc

np.seterr(divide = 'raise', invalid = 'raise')


__all__ = ['_generate_training_data','generate_training_data','generate_test_data',
           'split_ramps_from_cube', 'is_random_sampling_of_pure_rgbs', 'plot_rgb_xyz_lab_of_set']

#------------------------------------------------------------------------------
# training and test data generation 
#------------------------------------------------------------------------------    
def _generate_training_data(inc = 10, inc_offset = 0, nbit = 8, include_max = True, include_min = True,
                            seed = 0, randomize_order = True):
    """
    Generate RGB training data by creating a cube of RGB values.
    
    Args:
        :inc:
            | 10, optional
            | Increment along each channel (=R,G,B) axes in the RGB cube.
        :inc_offset:
            | 0, optional
            | The offset along each channel axes from which to start incrementing.
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :include_max:
            | True, optional
            | If True: ensure all combinations of max value (e.g. 255 for nbit = 8) are included in RGB cube. 
        :include_min:
            | True, optional
            | If True: ensure all combinations of min value 0 are included in RGB cube. 
        :seed:
            | 0, optional
            | Seed for setting the state of numpy's random number generator.
        :randomize_order:
            | True, optional
            | Randomize the order of the (xyz,rgb) pairs before output.
            
    Returns:
        :rgb:
            | ndarray with RGB values.
    """
    dv = np.arange(inc_offset, 2**nbit - 1, inc)
    if (dv.max() < (2**nbit-1)) & (include_max): dv = np.hstack((dv,[2**nbit-1])) # ensure max values are present
    if (dv.min() > 0) & (include_min): dv = np.hstack((0, dv)) # ensure 0 values are present
    rgb =  np.array(list(itertools.product(*[dv]*3)))
        
    # Randomize order of training pairs:
    if randomize_order:
        np.random.seed(seed)
        idx = np.arange(rgb.shape[0]) # get indices
        np.random.shuffle(idx) # random order for indices
        rgb = rgb[idx] # re-arrange rgb in random order
                     
    return rgb 

def split_ramps_from_cube(rgb, xyz = None, rgb_only = False):
    """ 
    Split a cube data set in pure RGB (ramps) and non-pure (remainder of cube). 
    """
    # select pures from full re-sampled RGB cube:
    pure_rgb_bool = dc.find_pure_rgb(rgb)
    if rgb_only == False: 
        xyz_pure = np.array([xyz[pure_rgb_bool[i],:] for i in range(3)]).reshape(-1,3)
    rgb_pure = np.array([rgb[pure_rgb_bool[i],:] for i in range(3)]).reshape(-1,3)

    # ensure monotonically increasing rgbs for pures (delete extra blacks):
    black_rgb_bool = dc.find_index_in_rgb(rgb_pure, k = [0,0,0], as_bool = True)
    rgb_black = rgb_pure[black_rgb_bool].mean(axis=0,keepdims=True) # find blacks and get average value
    rgb_pure = np.vstack((rgb_black, rgb_pure[~black_rgb_bool])) # only keep non-blacks, except one.
    if rgb_only == False:
        xyz_black = xyz_pure[black_rgb_bool].mean(axis=0,keepdims=True) 
        xyz_pure = np.vstack((xyz_black, xyz_pure[~black_rgb_bool]))

    # select non-pure rgb:
    nonpure_rgb_bool = ~(pure_rgb_bool[0] | pure_rgb_bool[1] | pure_rgb_bool[2])
    rgb_nonpure = rgb[nonpure_rgb_bool,:]
    rgb = np.vstack((rgb_pure,rgb_nonpure))
    if rgb_only == False: 
        xyz_nonpure = xyz[nonpure_rgb_bool,:]
    if rgb_only: xyz_pure, xyz_nonpure = None, None
    return (xyz_pure, rgb_pure), (xyz_nonpure, rgb_nonpure)

def is_random_sampling_of_pure_rgbs(inc):
    """ 
    Return boolean indicating if the RGB cube axes (=single channel ramps) 
    are sampled (different increment) independently from the remainder of the cube.
    
    Note:
        1. Independent sampling is indicated when :inc: is a list with 2 different values.
    """
    if len(inc) == 1: 
        pure_rgb_sampled_separately = False
    else:
        if inc[0] == inc[1]:
            pure_rgb_sampled_separately = False
        else:
            pure_rgb_sampled_separately = True
    return pure_rgb_sampled_separately

def generate_training_data(inc = [10], inc_offset = 0, nbit = 8,
                           seed = 0, randomize_order = True,
                           verbosity = 0, fig = None):
    
    """
    Generate RGB training pairs by creating a cube of RGB values.
    
    Args:
        :inc:
            | [10], optional
            | Increment along each channel (=R,G,B) axes in the RGB cube.
            | If inc is a list with 2 different values the RGB cube axes 
            | are sampled independently from the remainder of the cube.
            | --> inc = [inc_remainder, inc_axes]
        :inc_offset:
            | 0, optional
            | The offset along each channel axes from which to start incrementing.
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :include_max:
            | True, optional
            | If True: ensure all combinations of max value (e.g. 255 for nbit = 8) are included in RGB cube. 
        :include_min:
            | True, optional
            | If True: ensure all combinations of min value 0 are included in RGB cube. 
        :seed:
            | 0, optional
            | Seed for setting the state of numpy's random number generator.
        :randomize_order:
            | True, optional
            | Randomize the order of the (xyz,rgb) pairs before output.
        :verbosity:
            | 0, optional
            | Level of output.
            
    Returns:
        :rgb:
            | ndarray with RGB values.
    """
        
    rgb_tr = _generate_training_data(inc = inc[0], inc_offset = inc_offset, nbit = nbit, randomize_order = True, seed = seed)
    
    pure_rgb_sampled_separately = is_random_sampling_of_pure_rgbs(inc)
            
    if pure_rgb_sampled_separately:
        # create extra training set to extract pure R,G,B data from:
        rgb_tr_pure = _generate_training_data(inc = inc[1], inc_offset = inc_offset, nbit = nbit, randomize_order = True, seed = seed)
        
        # select pures from full re-sampled RGB cube:
        pure_rgb_bool = dc.find_pure_rgb(rgb_tr_pure)
        rgb_tr_pure = np.array([rgb_tr_pure[pure_rgb_bool[i],:] for i in range(3)]).reshape(-1,3)
        
        # ensure monotonically increasing rgbs for pures (delete extra blacks):
        black_rgb_bool = dc.find_index_in_rgb(rgb_tr_pure, k = [0,0,0], as_bool = True)
        rgb_tr_black = rgb_tr_pure[black_rgb_bool].mean(axis=0,keepdims=True) # find blacks and get average value
        rgb_tr_pure = np.vstack((rgb_tr_black, rgb_tr_pure[~black_rgb_bool])) # only keep non-blacks, except one.
       
        # keep non-pure from original training data and add new pure rgb data:
        pure_rgb_bool = dc.find_pure_rgb(rgb_tr)
        nonpure_rgb_bool = ~(pure_rgb_bool[0] | pure_rgb_bool[1] | pure_rgb_bool[2])
        rgb_tr_nonpure = rgb_tr[nonpure_rgb_bool,:]
        rgb_tr = np.vstack((rgb_tr_pure,rgb_tr_nonpure))
    
        # Randomize order of training pairs:
        if randomize_order:
            np.random.seed(seed)
            idx = np.arange(rgb_tr.shape[0]) # get indices
            np.random.shuffle(idx) # random order for indices
            rgb_tr = rgb_tr[idx] # re-arrange rgb in random order                
                
    
    if verbosity > 0: print('Number of training points for inc = {} and inc_offset = {:1.0f}: {:1.0f}.'.format(inc, inc_offset, rgb_tr.shape[0]))
    if verbosity > 1: 
        data = rgb_tr 
        data_contains = ['rgb'] 
        fig, axs = plot_rgb_xyz_lab_of_set(data, data_contains = data_contains, subscript = '_train', nrows = 2, row = 1, fig = fig)
        if pure_rgb_sampled_separately:
            pure_rgb_bool = dc.find_pure_rgb(rgb_tr)
            pure_rgb_bool = (pure_rgb_bool[0] | pure_rgb_bool[1] | pure_rgb_bool[2])
            data = rgb_tr[pure_rgb_bool]
            plot_rgb_xyz_lab_of_set(data, data_contains = data_contains, subscript = '_train', nrows = 2, row = 1, marker = '+', fig = fig, axs = axs)

    return rgb_tr.astype(int)

def generate_test_data(dlab = [10,10,10], nbit = 8, seed = 0, xyzw = None, cieobs = _CIEOBS,
                       xyzrgb_hull = None, randomize_order = True, verbosity = 0, fig = None):
    """
    Generate XYZ test values by creating a cube of CIELAB L*a*b* values, then converting these to XYZ values.
    
    Args:
        :dlab:
            | [10,10,10], optional
            | Increment along each CIELAB (=L*,a*,b*) axes in the Lab cube.
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :seed:
            | 0, optional
            | Seed for setting the state of numpy's random number generator.
        :xyzw:
            | None, optional
            | White point xyz to convert from lab to xyz
            | If None: use the white in xyzrgb_hull. If this is also None: use _CIE_D65 white.
        :cieobs:
            | _CIEOBS, optional
            | CIE standard observer used to convert _CIE_D65 to XYZ when xyzw 
            |   needs to be determined from the illuminant spectrum.
        :xyzrgb_hull:
            | None, optional
            | ndarray with (XYZ,RGB) pairs from which the hull (= display gamut) can be determined.
            | If None: test XYZ might fall outside of display gamut !
        :randomize_order:
            | True, optional
            | Randomize the order of the test xyz before output.
        :verbosity:
            | 0, optional
            | Level of output.
            
    Returns:
        :xyz:
            | ndarray with XYZ values.
    """
    if xyzrgb_hull is not None:
        xyz_hull, rgb_hull = xyzrgb_hull # if pre-calculated or measured: must be normalized to white Yw = 100 !
    
        # find white xyz:
        if xyzw is None: xyzw = xyz_hull[dc.find_index_in_rgb(rgb_hull, k = [2**nbit - 1]*3),:].mean(axis = 0, keepdims = True)
    else: 
        if xyzw is None: xyzw = spd_to_xyz(_CIE_D65, cieobs = cieobs, relative = True)
        
    # generate grid of points in lab space and convert to xyz:
    l = np.arange(0,100+dlab[1],dlab[0])
    a = np.hstack((np.arange(-200,0,dlab[1]),np.arange(0,200+dlab[1],dlab[1])))
    b = np.hstack((np.arange(-200,0,dlab[2]),np.arange(0,200+dlab[2],dlab[2])))
    lab = np.array(list(itertools.product(l,a,b)))
    xyz = lab_to_xyz(lab, xyzw = xyzw)
    
    # get rid of those that result in negative xyz values:
    xyz = xyz[~((xyz<0).sum(-1)>0),:]
    
    if xyzrgb_hull is not None:
        # lab_hull = xyz_to_lab(xyz_hull, xyzw = xyzw)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection ='3d')
        # ax.plot(lab_hull[:,1],lab_hull[:,2],lab_hull[:,0],'.')
        # lab_hull = xyz_to_lab(xyz_hull, xyzw = xyzw)
        # print(xyz_hull.shape, xyz_hull.max(),xyz_hull.min(),xyzw)
        # raise Exception('')
        # in_gamut = math.in_hull(lab, lab_hull)
        
        # Only keep those that are within the XYZ-gamut of the display (use XYZ gamut as lab-gamut has concave surfaces !):
        in_gamut = math.in_hull(xyz, xyz_hull)
        xyz = xyz[in_gamut,:]
        
    # Randomize order of training pairs:
    if randomize_order:
        np.random.seed(seed)
        idx = np.arange(xyz.shape[0]) # get indices
        np.random.shuffle(idx) # random order for indices
        xyz = xyz[idx] # re-arrange xyz in random order

    # Make plots:
    #convert xyz to lab for plots:
    lab = xyz_to_lab(xyz, xyzw = xyzw)
    
    if verbosity > 0: print('Number of in-gamut test points for dlab = [{:1.0f},{:1.0f},{:1.0f}]:  {:1.0f}'.format(*dlab, xyz.shape[0]))
    data = np.dstack((xyz,lab)) 
    data_contains = ['xyz','lab'] 
    if verbosity > 1: plot_rgb_xyz_lab_of_set(data, data_contains = data_contains, subscript = '_test', nrows = 2, row = 2, fig = fig)

    return xyz
    
def plot_rgb_xyz_lab_of_set(rgb_xyz_lab, subscript = '', data_contains = ['rgb','xyz','lab'],
                            nrows = 1, row = 1, fig = None, axs = None, figsize = (14,7),
                            marker = '.'):
    """
    Make 3d-plots of the RGB, XYZ and L*a*b* cubes of the data in rgb_xyz_lab.
    
    Args:
        :rgb_xyz_lab:
            | ndarray with RGB, XYZ, Lab data.
        :subscript:
            | '', optional
            |subscript to add to the axis labels.
        :data_contains:
            | ['rgb','xyz','lab'], optional
            | specifies what is in rgb_xyz_lab
        :nrows:
            | 1, optional
            | Number of rows in (nx3) figure.
        :row:
            | 1, optional
            | Current row number to plot to (when using the function to plot nx3 figures)
        :fig:
            | None, optional
            | Figure handle.
            | If None: generate new figure.
        :axs:
            | None, optional
            | Axes handles: (3,) or None
            | If None: add new axes for each of the RGB, XYZ, Lab subplots.
        :figsize:
            | (14,7), optional
            | Figure size.
        :marker:
            | '.', optional
            | Marker symbol used for plotting.
            
    Return:
        :fig, axs:
            | Handles to the figure and the three axes in that figure.
        
    """
    if fig is None:
        fig = plt.figure(figsize = figsize)
    
        
    axis_labels = {'rgb' : ['G','B','R'],
                  'xyz' : ['Y','Z','X'],
                  'lab' : ['a*','b*','L*']}
    axs_ = []
    if rgb_xyz_lab.ndim == 2: rgb_xyz_lab = rgb_xyz_lab[...,None]
    for i in range(rgb_xyz_lab.shape[-1]):
        ax_i = axs[i] if axs is not None else fig.add_subplot(nrows, len(data_contains), 1 + i + (row-1)*len(data_contains), projection='3d')
        ax_i.plot(rgb_xyz_lab[:,1,i], rgb_xyz_lab[:,2,i], rgb_xyz_lab[:,0,i], marker = marker, linestyle = 'none')
        ax_i.set_xlabel(axis_labels[data_contains[i]][0]+subscript)
        ax_i.set_ylabel(axis_labels[data_contains[i]][1]+subscript)
        ax_i.set_zlabel(axis_labels[data_contains[i]][2]+subscript)
        axs_.append(ax_i)
    
    return fig, axs_
    
if __name__ == '__main__':
    
    # generate some training RGB with axes spacing of 10, inner cube spacing 30:
    rgb_tr = generate_training_data(inc = [30, 10], verbosity = 2)
    print('rgb_tr.shape: ', rgb_tr.shape)
    
    # generate some XYZ test data:
    xyz_t = generate_test_data(dlab = [10,10,10], xyzrgb_hull = None, 
                               verbosity = 2, fig = None)
    print('xyz_t.shape: ', xyz_t.shape)
    
    # To ensure XYZ test data is within hull of display, provide xyzrgb_hull data:
    # simulate some measurements of the training data using a virtual display:
    from virtualdisplay import VirtualDisplay
    vd = VirtualDisplay(model = 'virtualdisplay_kwak2000_SII', channel_dependence = True)
    xyz_tr = vd.to_xyz(rgb_tr) # 'measure xyz for each of the displayed rgb_tr
    print('simulated xyz_tr.shape: ', xyz_tr.shape)
    
    # generate some XYZ test data within the display gamut 
    # (make sure black and white are present in xyzrgb_hull!):
    xyz_t_ingamut = generate_test_data(dlab = [10,10,10], xyzrgb_hull = (xyz_tr,rgb_tr), 
                                       verbosity = 2, fig = None)
    print('xyz_t_ingamut.shape: ', xyz_t_ingamut.shape)
    
    
    