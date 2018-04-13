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
#################################################################################
# Module with functions related to color rendering Vector Field and Pixel models
#################################################################################
#--VECTOR FIELD MODEL-------------------------------------------------------------
#
# _VF_CRI_DEFAULT: default cri_type parameters for VF model
#
# _VF_CSPACE: default dict with color space parameters.
#
# _VF_MAXR: maximum C to use in calculations and plotting of vector fields
#
# _VF_DELTAR:  grid spacing, pixel size
#
# _VF_MODEL_TYPE: type of polynomial model for base color shifts
#
# _DETERMINE_HUE_ANGLES: Bool, determines whether to calculate hue_angles for 5 or 6 'informative' model parameters
#
# _VF_PCOLORSHIFT: Default dict with hue_angle parameters for VF model
#
# _VF_SIG = 0.3 #  Determines smoothness of the transition between hue-bin-boundaries (no hard cutoff at boundary).
# 
# get_poly_model(): Setup base color shift model (delta_a, delta_b), determine model parameters and accuracy.
#
# apply_poly_model_at_x(): Applies base color shift model at cartesian coordinates axr, bxr.
#
# generate_vector_field(): Generates a field of vectors using the base color shift model.
#
# VF_colorshift_model(): Applies full vector field model calculations to spectral data.
#
# generate_grid():  Generate a grid of color coordinates.
#
# calculate_shiftvectors(): Calculate color shift vectors.
#
# plot_shift_data(): Plots vector or circle fields.
#
# plotcircle(): Plot one or more concentric circles.
#
# initialize_VF_hue_angles(): Initialize the hue angles that will be used to 'summarize' the VF model fitting parameters.
#
#
#--PIXEL MODEL-------------------------------------------------------------------
# 
# get_pixel_coordinates(): Get pixel coordinates corresponding to color coordinates in jab.
# 
# PX_colorshift_model(): Pixelates the color space and calculates the color shifts in each pixel.
#
#
#--VECTOR FIELD & PIXEL MODEL----------------------------------------------------
#
# calculate_VF_PX_models(): Calculate Vector Field and Pixel color shift models.
#
# subsample_RFL_set(): Sub samples a set of spectral reflectance functions by pixelization of color space.
#
# plot_VF_PX_models(): Plot the VF and PX model color shift vectors.
#
#
#--------------------------------------------------------------------------------

Created on Wed Mar 28 18:30:30 2018


@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .. import np, pd, _CIE_ILLUMINANTS, spd_to_xyz, colortf
from .vectorshiftmodel import *
from .pixelshiftmodel import *

# .colorrendition_vectorshiftmodel:
__all__ = ['_VF_CRI_DEFAULT','_VF_CSPACE','_VF_CSPACE_EXAMPLE','_VF_CIEOBS','_VF_MAXR','_VF_DELTAR','_VF_MODEL_TYPE','_VF_SIG','_VF_PCOLORSHIFT']
__all__ += ['get_poly_model','apply_poly_model_at_x','generate_vector_field','VF_colorshift_model','initialize_VF_hue_angles']
__all__ += ['generate_grid','calculate_shiftvectors','plot_shift_data','plotcircle']

# .colorrendition_pixelshiftmodel:
__all__ += ['get_pixel_coordinates','PX_colorshift_model']

# local:
__all__ +=['calculate_VF_PX_models','subsample_RFL_set','plot_VF_PX_models']


#--VECTOR FIELD & PIXEL MODEL functions------------------------------------------

def calculate_VF_PX_models(S, cri_type = _VF_CRI_DEFAULT, sampleset = None, pool = False, \
                           pcolorshift = {'href': np.arange(np.pi/10,2*np.pi,2*np.pi/10),'Cref' : _VF_MAXR, 'sig' : _VF_SIG, 'labels' : '#'},\
                           vfcolor = 'k', verbosity = 0):
    """
    Calculate Vector Field and Pixel color shift models.
    
    Args:
        :cri_type: _VF_CRI_DEFAULT or str or dict, optional
            Specifies type of color fidelity model to use. 
            Controls choice of reference illuminant, sample set, averaging, scaling, etc.
            See luxpy.cri.spd_to_cri for more info.
        :sampleset:  None or str or numpy.ndarray, optional
            Sampleset to be used when calculating vector field model.
        :pool: False, optional
            If :S: contains multiple spectra, True pools all jab data before modeling the vector field, False models a different field for each spectrum.
        :pcolorshift:  {'href': np.arange(np.pi/10,2*np.pi,2*np.pi/10),'Cref' : _VF_MAXR, 'sig' : _VF_SIG} or user defined dict, optional
            Dict containing the specifications input for apply_poly_model_at_hue_x().
            The polynomial models of degree 5 and 6 can be fully specified or summarized 
            by the model parameters themselved OR by calculating the dCoverC and dH at resp. 5 and 6 hues.
        :vfcolor: 'k', optional
            For plotting the vector fields.
        :verbosity:= 0, optional
            Report warnings or not.
    
    Returns:
        :dataVF:, :dataPX: dicts 
            For more info see output description of resp. luxpy.cri.VF_colorshift_model() and luxpy.cri.PX_colorshift_model()
    """
    # calculate VectorField cri_color_shift model:
    dataVF = VF_colorshift_model(S, cri_type = cri_type, sampleset = sampleset, vfcolor = vfcolor, pcolorshift = pcolorshift, pool = pool, verbosity = verbosity)
    
    # Set jab_ranges and _deltas for PX-model pixel calculations:
    PX_jab_deltas = np.array([_VF_DELTAR,_VF_DELTAR,_VF_DELTAR]) #set same as for vectorfield generation
    PX_jab_ranges = np.vstack(([0,100,_VF_DELTAR],[-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR], [-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR]))#IES4880 gamut
   
    # Calculate shift vectors using vectorfield and pixel methods:
    delta_SvsVF_vshift_ab_mean = np.nan*np.ones((len(dataVF),1))
    delta_SvsVF_vshift_ab_mean_normalized = delta_SvsVF_vshift_ab_mean.copy()
    delta_PXvsVF_vshift_ab_mean = np.nan*np.ones((len(dataVF),1))
    delta_PXvsVF_vshift_ab_mean_normalized = delta_PXvsVF_vshift_ab_mean.copy()
    dataPX = [[] for k in range(len(dataVF))]
    for Snr in range(len(dataVF)):

        # Calculate shifts using pixel method, PX:
        dataPX[Snr] = PX_colorshift_model(dataVF[Snr]['Jab']['Jabt'][:,0,:],dataVF[Snr]['Jab']['Jabr'][:,0,:], jab_ranges = PX_jab_ranges, jab_deltas = PX_jab_deltas,limit_grid_radius = _VF_MAXR)
        
        # Calculate shift difference between Samples (S) and VectorField model predictions (VF):
        delta_SvsVF_vshift_ab = dataVF[Snr]['vshifts']['vshift_ab_s'] - dataVF[Snr]['vshifts']['vshift_ab_s_vf']
        delta_SvsVF_vshift_ab_mean[Snr] = np.nanmean(np.sqrt((delta_SvsVF_vshift_ab[...,1:3]**2).sum(axis = delta_SvsVF_vshift_ab[...,1:3].ndim-1)), axis=0)
        delta_SvsVF_vshift_ab_mean_normalized[Snr] = delta_SvsVF_vshift_ab_mean[Snr]/dataVF[Snr]['Jab']['DEi'].mean(axis=0)
        
        # Calculate shift difference between PiXel method (PX) and VectorField (VF):
        delta_PXvsVF_vshift_ab = dataPX[Snr]['vshifts']['vectorshift_ab_J0'] - dataVF[Snr]['vshifts']['vshift_ab_vf']
        delta_PXvsVF_vshift_ab_mean[Snr] = np.nanmean(np.sqrt((delta_PXvsVF_vshift_ab[...,1:3]**2).sum(axis = delta_PXvsVF_vshift_ab[...,1:3].ndim-1)), axis=0)
        delta_PXvsVF_vshift_ab_mean_normalized[Snr] = delta_PXvsVF_vshift_ab_mean[Snr]/dataVF[Snr]['Jab']['DEi'].mean(axis=0)

        dataVF[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean'] = delta_PXvsVF_vshift_ab_mean[Snr]
        dataVF[Snr]['vshifts']['delta_SvsVF_vshift_ab_mean'] = delta_SvsVF_vshift_ab_mean[Snr]
        dataVF[Snr]['vshifts']['delta_SvsVF_vshift_ab_mean_normalized'] = delta_SvsVF_vshift_ab_mean_normalized[Snr]
        dataVF[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean_normalized'] = delta_PXvsVF_vshift_ab_mean_normalized[Snr]
        dataPX[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean'] = dataVF[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean']
        dataPX[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean_normalized'] = dataVF[Snr]['vshifts']['delta_PXvsVF_vshift_ab_mean_normalized']

    return dataVF, dataPX

#------------------------------------------------------------------------------
def subsample_RFL_set(rfl, rflpath = '', samplefcn = 'rand', S = _CIE_ILLUMINANTS['E'], \
                      jab_ranges = None, jab_deltas = None, cieobs = _VF_CIEOBS, cspace = _VF_CSPACE, \
                      ax = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), \
                      bx = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), \
                      jx = None, limit_grid_radius = 0):
    """
    Sub samples a set of spectral reflectance functions by pixelization of color space.
    
    Args:
        :rfl: numpy.ndarray or str
            Array with of str refering to a set of spectral reflectance functions to be subsampled.
            If str to file: file must contain data as columns, with first column the wavelengths.
        :rflpath: '' or str, optional
            Path to folder with rfl-set specified in a str :rfl: filename.
        :samplefcn: 'rand' or 'mean', optional
            'rand' selects a random sample from the samples present in each pixel,
            while 'mean' returns the mean spectral reflectance in each pixel.
        :S: _CIE_ILLUMINANTS['E'], optional
            Illuminant used to calculate the color coordinates of the spectral reflectance samples.
        :jab_ranges: None or numpy.ndarray (.shape =(3,3), first axis: J,a,b, second axis: min, max, delta), optional
            Specifies the pixelization of color space.
        :jab_deltas: float or numpy.ndarray, optional
            Specifies the sampling range. 
            A float uses jab_deltas as the maximum Euclidean distance to select
            samples around each pixel center. A numpy.ndarray of 3 deltas, uses
            a city block sampling around each pixel center.
        :cspace: _VF_CSPACE or dict, optional
            Specifies color space. See _VF_CSPACE_EXAMPLE for example structure.
        :cieobs: _VF_CIEOBS or str, optional
            Specifies CMF set used to calculate color coordinates.
                :ax: np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR) or numpy.ndarray, optional
        :ax: np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR) or numpy.ndarray, optional
        :jx: None, optional
            Note that a not-None :jab_ranges: overrides :ax:, :bx: and :jx input.
        :limit_grid_radius: 0, optional
            A value of zeros keeps grid as specified  by axr,bxr.
            A value > 0 only keeps (a,b) coordinates within a radius of :limit_grid_radius:.
   
    Returns:
        :returns: rflsampled, jabp
            numpy.ndarrays with resp. the subsampled set of spectral reflectance functions and the pixel coordinate centers
    """
    # Testing effects of sample set, pixel size and gamut size:
    if type(rfl) == str:
        rfl = pd.read_csv(os.path.join(rflpath,rfl),header = None).get_values().T
  
    # Calculate Jab coordinates of samples:
    xyz,xyzw = spd_to_xyz(S, cieobs = cieobs, rfl = rfl.copy(), out = 2)
    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    cspace_pars['xyzw'] = xyzw
    jab = colortf(xyz,tf = cspace['type'],fwtf = cspace_pars)

    # Generate grid and get samples in each grid:
    gridp,idxp, jabp, pixelsamplenrs, pixelIDs = get_pixel_coordinates(jab, jab_ranges = jab_ranges, jab_deltas = jab_deltas, limit_grid_radius = limit_grid_radius)

    # Get rfls from set using sampling function (mean or rand):
    W = rfl[:1]
    R = rfl[1:]
    rflsampled = np.nan*np.ones((len(idxp),R.shape[1]))
    for i in range(len(idxp)):
        if samplefcn == 'mean':
            rfl_i = np.nanmean(rfl[pixelsamplenrs[i],:],axis = 0)
        else:
            samplenr_i = np.random.randint(len(pixelsamplenrs[i]))
            rfl_i = rfl[pixelsamplenrs[i][samplenr_i],:]
        rflsampled[i,:] = rfl_i        
    rflsampled = np.vstack((W,rflsampled))
    return rflsampled, jabp


#------------------------------------------------------------------------------
def plot_VF_PX_models(dataVF = None, dataPX = None, plot_VF = True, plot_PX = True, axtype='polar', ax = 'new', \
                      plot_circle_field = True, plot_sample_shifts = False, plot_samples_shifts_at_pixel_center = False, \
                      jabp_sampled = None, plot_VF_colors = ['g'], plot_PX_colors = ['r'], hbin_cmap = None, \
                      bin_labels = None, plot_bin_colors = True, force_CVG_layout = False):
    """
    Plot the VF and PX model color shift vectors.
    
    Args:
        :dataVF: None or list[dict] with data obtained with VF_colorshift_model(), optional
            None plots nothing related to VF model.
            Each list element refers to a different test SPD.
        :dataPX: None or list[dict] with data obtained with PX_colorshift_model(), optional
            None plots nothing related to PX model.
            Each list element refers to a different test SPD.
        :plot_VF: True, optional
            Plot VF model (if :dataVF: is not None).
        :plot_PX: True, optional
            Plot PX model (if :dataPX: is not None).  
        :axtype: 'polar' or 'cart', optional
            Make polar or Cartesian plot.
        :ax: None or 'new' or 'same', optional
            - None or 'new' creates new plot
            - 'same': continue plot on same axes.
            - axes handle: plot on specified axes. 
        :plot_circle_field: True or False, optional
            Plot lines showing how a series of circles of color coordinates is distorted by the test SPD.
            The width (wider means more) and color (red means more) of the lines specify the intensity of the hue part of the color shift.
        :plot_sample_shifts: False or True, optional
            Plots the shifts of the individual samples of the rfl-set used to calculated the VF model.
        :plot_samples_shifts_at_pixel_center: False, optional
            Offers the possibility of shifting the vector shifts of subsampled sets from the reference illuminant positions to the pixel centers.
            Note that the pixel centers must be supplied in :jabp_sampled:.
        :jabp_sampled: None, numpy.ndarray, optional
            Corresponding pixel center for each sample in a subsampled set.
        :plot_VF_colors: ['g'] or list[str], optional
            Specifies the plot color the color shift vectors of the VF model. 
            If len(:plot_VF_colors:) == 1: same color for each list element of :dataVF:.
        :plot_VF_colors: ['g'] or list[str], optional
            Specifies the plot color the color shift vectors of the VF model. 
            If len(:plot_VF_colors:) == 1: same color for each list element of :dataVF:.
        :hbin_cmap: None or colormap, optional
            Color map with RGB entries for each of the hue bins specified by the hues in _VF_PCOLORSHIFT.
            If None: will be obtained on first run by luxpy.cri.plot_shift_data() and returned as :cmap: for use in other functions.
        :plot_bin_colors: True, optional
            Colorize hue-bins.
        :bin_labels: None or list[str] or '#', optional
            Plots labels at the bin center hues.
            - None: don't plot.
            - list[str]: list with str for each bin. (len(:bin_labels:) = :nhbins:)
            - '#': plots number.
            - '_VF_PCOLORSHIFT': uses labels in _VF_PCOLORSHIFT['labels']
            - 'pcolorshift': uses the labels in dataVF['modeldata']['pcolorshift']['labels']
        :force_CVG_layout: False or True, optional
            True: Force plot of basis of CVG.
    Returns:
        :returns: ax (handle to current axes), cmap (hbin_cmap)
    """
    if dataVF is not None:
        if len(plot_VF_colors) < len(dataVF):
            plot_VF_colors = [plot_VF_colors.append('g') for i in range(len(dataVF))]
    if dataPX is not None:
        if len(plot_PX_colors) < len(dataPX):
            plot_PX_colors = [plot_PX_colors.append('r') for i in range(len(dataPX))]       
    
    cmap = hbin_cmap
    
    for Snr in range(len(dataVF)):  
        if bin_labels is not None:
            if (bin_labels is 'pcolorshift') & (dataVF is not None):
                hbins = dataVF[Snr]['modeldata']['pcolorshift']['href']*180/np.pi
                start_hue = 0
                scalef0 = dataVF[Snr]['modeldata']['pcolorshift']['Cref']
                bin_labels = dataVF[Snr]['modeldata']['pcolorshift']['labels']
            else: 
                hbins = _VF_PCOLORSHIFT['href']*180/np.pi
                start_hue = 0
                scalef0 = _VF_PCOLORSHIFT['Cref']
                bin_labels = _VF_PCOLORSHIFT['labels']
        else:
            scalef0 = 100
        if plot_circle_field == True:
            scalef = scalef0*1.35
        else:
            scalef = scalef0
        # Plot shift vectors obtained using VF method:    
        if (dataVF is not None) & (plot_VF == True):
            if ((Snr==0) & (ax == 'new')):
                figCVG, ax, cmap = plot_shift_data(dataVF[Snr], fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef, color = plot_VF_colors[Snr],axtype = axtype,  ax = ax, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
            else:
                plot_shift_data(dataVF[Snr], fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef, color = plot_VF_colors[Snr], axtype = axtype, ax = ax,  force_CVG_layout = force_CVG_layout, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
                force_CVG_layout = False

        # Plot shift vectors obtined using PX method:  
        if ((dataPX is not None) & (plot_PX == True)):
            if (Snr==0) & (ax == 'new') & (plot_VF == False):
                figCVG, ax, cmap = plot_shift_data(dataPX[Snr], fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef, color = plot_PX_colors[Snr], ax = ax, axtype = axtype,  bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
            else:
                plot_shift_data(dataPX[Snr], fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef, color = plot_PX_colors[Snr], ax = ax, axtype = axtype, force_CVG_layout = force_CVG_layout, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
                force_CVG_layout = False

        # Plot sample data to check vector field shifts::
        if (plot_sample_shifts == True) & (dataVF is not None):

            dataS = dataVF[Snr].copy()
            dataS['fielddata']['vectorfield']['axr'] = dataVF[Snr]['Jab']['Jabr'][...,1][:,0]
            dataS['fielddata']['vectorfield']['bxr'] = dataVF[Snr]['Jab']['Jabr'][...,2][:,0]
            dataS['fielddata']['vectorfield']['axt'] = dataVF[Snr]['Jab']['Jabt'][...,1][:,0]
            dataS['fielddata']['vectorfield']['bxt'] = dataVF[Snr]['Jab']['Jabt'][...,2][:,0]
            
            if (plot_samples_shifts_at_pixel_center == True):
                if (jabp_sampled is not None): # set vector shifts to center of 'pixel' used in subsampling the rfl-set
                    dataS['fielddata']['vectorfield']['axr'] = jabp_sampled[:,1].copy()
                    dataS['fielddata']['vectorfield']['bxr'] = jabp_sampled[:,2].copy()
                    dataS['fielddata']['vectorfield']['axt'] = jabp_sampled[:,1] + dataVF[Snr]['Jab']['Jabt'][...,1][:,0] - dataVF[Snr]['Jab']['Jabr'][...,1][:,0]
                    dataS['fielddata']['vectorfield']['bxt'] = jabp_sampled[:,2] + dataVF[Snr]['Jab']['Jabt'][...,2][:,0] - dataVF[Snr]['Jab']['Jabr'][...,2][:,0]
                   
            if (Snr==0) & (ax == 'new') &  (plot_VF is False):
                figCVG, ax, cmap = plot_shift_data(dataS, fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef,color = 'k', ax = ax, axtype = axtype,force_CVG_layout = True, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
            else:
                plot_shift_data(dataS, fieldtype = 'vectorfield', hbins = hbins, start_hue = start_hue, scalef = scalef,color = 'k', ax = ax, axtype = axtype, force_CVG_layout = force_CVG_layout, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
                force_CVG_layout = False
        
        if (plot_circle_field == True) & (dataVF is not None):
            if  (cmap is None):
                figCVG, ax, cmap = plot_shift_data(dataVF[Snr], fieldtype = 'circlefield', hbins = hbins, start_hue = start_hue, scalef = scalef,color = 'darkgrey', ax = ax, axtype = axtype, force_CVG_layout = True, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
            else:
                plot_shift_data(dataVF[Snr], fieldtype = 'circlefield', hbins = hbins, start_hue = start_hue, scalef = scalef,color = 'darkgrey', ax = ax, axtype = axtype, force_CVG_layout = force_CVG_layout, bin_labels = bin_labels, plot_bin_colors = plot_bin_colors)
                force_CVG_layout = False
                
        if axtype == 'cart':
            plotcircle(color = 'grey')

    return ax, cmap