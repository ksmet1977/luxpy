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
# Module with functions related to color rendering Pixel models
#################################################################################
# 
# get_pixel_coordinates(): Get pixel coordinates corresponding to color coordinates in jab.
# 
# PX_colorshift_model(): Pixelates the color space and calculates the color shifts in each pixel.
#
#------------------------------------------------------------------------------
Created on Wed Mar 28 18:57:50 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

#from .colorrendition_indices import *
#from .colorrendition_graphics import *
from .. import np
from .colorrendition_vectorshiftmodel import _VF_DELTAR,_VF_DELTAR, generate_grid

__all__ = ['get_pixel_coordinates','PX_colorshift_model']

def get_pixel_coordinates(jab, jab_ranges = None, jab_deltas = None, limit_grid_radius = 0):
    """
    Get pixel coordinates corresponding to color coordinates in jab.
    
    Args:
        :jab: numpy.ndarray of color coordinates
        :jab_ranges: None or numpy.ndarray (.shape =(3,3), first axis: J,a,b, second axis: min, max, delta), optional
            Specifies the pixelization of color space.
        :jab_deltas: float or numpy.ndarray, optional
            Specifies the sampling range. 
            A float uses jab_deltas as the maximum Euclidean distance to select
            samples around each pixel center. A numpy.ndarray of 3 deltas, uses
            a city block sampling around each pixel center.
        :limit_grid_radius: 0, optional
            A value of zeros keeps grid as specified  by axr,bxr.
            A value > 0 only keeps (a,b) coordinates within a radius of :limit_grid_radius:.
    
    Returns:
        :returns: gridp, idxp, jabp, samplenrs, samplesIDs
            - :gridp: numpy.ndarray with coordinates of all pixel centers.
            - :idxp: list[int] with pixel index for each non-empty pixel
            - :jabp: numpy.ndarray with center color coordinates of non-empty pixels
            - :samplenrs: list[list[int]] with sample numbers belong to each non-empty pixel
            - :sampleIDs: summarizing list, with column order: 'idxp, jabp, samplenrs'
    """
    if jab_deltas is None:
        jab_deltas = np.array([_VF_DELTAR,_VF_DELTAR,_VF_DELTAR])
    if jab_ranges is None:
        jab_ranges = np.vstack(([0,100,jab_deltas[0]],[-_VF_MAXR,_VF_MAXR+jab_deltas[1],jab_deltas[1]], [-_VF_MAXR,_VF_MAXR+jab_deltas[2],jab_deltas[2]]))
    
    # Get pixel grid:
    gridp = generate_grid(jab_ranges = jab_ranges, limit_grid_radius = limit_grid_radius) 

    # determine pixel coordinates of each sample in jab:
    samplesIDs = []
    for idx in np.arange(gridp.shape[0]):
        
        # get pixel coordinates:
        jp = gridp[idx,0]
        ap = gridp[idx,1]
        bp = gridp[idx,2]
        Cp = np.sqrt(ap**2+bp**2)
                      
        if type(jab_deltas) == np.ndarray:
            sampleID = np.where(((np.abs(jab[...,0]-jp) <= jab_deltas[0]/2) & (np.abs(jab[...,1]-ap) <= jab_deltas[1]/2) & (np.abs(jab[...,2]-bp) <= jab_deltas[2]/2)))
        else:
            sampleID = np.where((np.sqrt((jab[...,0]-jp)**2 + (jab[...,1]-ap)**2 + (jab[...,2]-bp)**2) <= jab_deltas/2))

        if (sampleID[0].shape[0] > 0):
            samplesIDs.append(np.hstack((idx,np.array([jp,ap,bp]),sampleID[0])))
       
    idxp = [np.int(samplesIDs[i][0]) for i in range(len(samplesIDs))]
    jabp = np.vstack([samplesIDs[i][1:4] for i in range(len(samplesIDs))])
    samplenrs = [np.array(samplesIDs[i][4:],dtype = int).tolist() for i in range(len(samplesIDs))]
    
    return gridp, idxp,jabp,samplenrs, samplesIDs


def PX_colorshift_model(Jabt,Jabr, jab_ranges = None, jab_deltas = None,limit_grid_radius = 0):
    """
    Pixelates the color space and calculates the color shifts in each pixel.
    
    Args:
        :Jabt: numpy.ndarray with color coordinates under the (single) test SPD.
        :Jabr: numpy.ndarray with color coordinates under the (single) reference SPD.  
        :jab_ranges: None or numpy.ndarray (.shape =(3,3), first axis: J,a,b, second axis: min, max, delta), optional
            Specifies the pixelization of color space.
        :jab_deltas: float or numpy.ndarray, optional
            Specifies the sampling range. 
            A float uses jab_deltas as the maximum Euclidean distance to select
            samples around each pixel center. A numpy.ndarray of 3 deltas, uses
            a city block sampling around each pixel center.
        :limit_grid_radius: 0, optional
            A value of zeros keeps grid as specified  by axr,bxr.
            A value > 0 only keeps (a,b) coordinates within a radius of :limit_grid_radius:.
            
    Returns:
        :returns: dict with the following keys:
            - 'Jab': dict with with numpy.ndarrays for Jabt, Jabr and DEi, DEi_ab (only ab-coordinates), DEa (mean) and DEa_ab
            - 'vshifts': dict with
                    * 'vectorshift': numpy.ndarray with vector shifts between average Jabt and Jabr for each pixel
                    * 'vectorshift_ab': numpy.ndarray with vector shifts averaged over J for each pixel
                    * 'vectorshift_ab_J0': numpy.ndarray with vector shifts averaged over J for each pixel of J=0 plane.
                    * 'vectorshift_len': length of 'vectorshift'
                    * 'vectorshift_ab_len': length of 'vectorshift_ab'
                    * 'vectorshift_ab_J0_len': length of 'vectorshift_ab_J0'
                    * 'vectorshift_len_DEnormed': length of 'vectorshift' normalized to 'DEa'
                    * 'vectorshift_ab_len_DEnormed': length of 'vectorshift_ab' to 'DEa_ab'
                    * 'vectorshift_ab_J0_len_DEnormed': length of 'vectorshift_ab_J0' to 'DEa_ab'
            - 'pixeldata': dict with pixel info:
                    * 'grid' numpy.ndarray with coordinates of all pixel centers.
                    * 'idx': list[int] with pixel index for each non-empty pixel
                    * 'Jab': numpy.ndarray with center color coordinates of non-empty pixels
                    * 'samplenrs': list[list[int]] with sample numbers belong to each non-empty pixel
                    * 'IDs: summarizing list, with column order: 'idxp, jabp, samplenrs'
            'fielddata' : dict with dicts containing data on the calculated vector-field and circle-fields 
                    * 'vectorfield': dict with numpy.ndarrays for the ab color coordinates 
                        under the reference (axr, bxr) and test illuminant (axt, bxt) centered at 
                        the pixel centers corresponding to the ab coordinates of the reference illuminant.
     """
    
    
    
    # get pixelIDs of all samples under ref. conditions:
    gridp,idxp, jabp, pixelsamplenrs, pixelIDs = get_pixel_coordinates(Jabr, jab_ranges = jab_ranges, jab_deltas = jab_deltas, limit_grid_radius = limit_grid_radius)

    # get average Jab coordinates for each pixel:
    Npixels = len(idxp) # number of non-empty pixels
    Jabr_avg = np.nan*np.ones((gridp.shape[0],3))
    Jabt_avg = Jabr_avg.copy()
    for i in np.arange(Npixels):
        Jabr_avg[idxp[i],:] = Jabr[pixelsamplenrs[i],:].mean(axis=0)
        Jabt_avg[idxp[i],:] = Jabt[pixelsamplenrs[i],:].mean(axis=0)
        jabtemp = Jabr[pixelsamplenrs[i],:]
        jabtempm = Jabr_avg[idxp[i],:]
            
    # calculate Jab vector shift:    
    vectorshift = Jabt_avg - Jabr_avg
    
    # calculate ab vector shift:
    uabs = gridp[gridp[:,0]==0,1:3] #np.unique(gridp[:,1:3],axis=0)
    vectorshift_ab_J0 = np.ones((uabs.shape[0],2))*np.nan
    vectorshift_ab = np.ones((vectorshift.shape[0],2))*np.nan
    for i in range(uabs.shape[0]):
        cond = (gridp[:,1:3] == uabs[i,:]).all(axis = 1)
        if cond.any() & np.logical_not(np.isnan(vectorshift[cond,1:3]).all()): #last condition is to avoid warning of taking nanmean of empty slice when all are NaNs
            vectorshift_ab_J0[i,:] = np.nanmean(vectorshift[cond,1:3], axis = 0)
            vectorshift_ab[cond,:] = np.nanmean(vectorshift[cond,1:3],axis = 0)
   
    # Calculate length of shift vectors:
    vectorshift_len = np.sqrt((vectorshift**2).sum(axis = vectorshift.ndim-1))
    vectorshift_ab_len = np.sqrt((vectorshift_ab**2).sum(axis = vectorshift_ab.ndim-1))
    vectorshift_ab_J0_len = np.sqrt((vectorshift_ab_J0**2).sum(axis = vectorshift_ab_J0.ndim-1))
    
    # Calculate average DE for normalization of vectorshifts
    DEi_Jab_avg = np.sqrt(((Jabt-Jabr)**2).sum(axis = Jabr.ndim-1))
    DE_Jab_avg = DEi_Jab_avg.mean(axis=0) 
    DEi_ab_avg = np.sqrt(((Jabt[...,1:3]-Jabr[...,1:3])**2).sum(axis = Jabr[...,1:3].ndim-1))
    DE_ab_avg = DEi_ab_avg.mean(axis=0) 

    # calculate vectorfield:
    axr = uabs[:,0,None]
    bxr = uabs[:,1,None]
    axt = axr + vectorshift_ab_J0[:,0,None]
    bxt = bxr + vectorshift_ab_J0[:,1,None]

    data = {'Jab' : {'Jabr': Jabr_avg, 'Jabt': Jabt_avg, 
                    'DEi' : DEi_Jab_avg, 'DEi_ab': DEi_ab_avg,
                    'DEa' : DE_Jab_avg, 'DEa_ab' : DE_ab_avg}, 
           'vshifts' : {'vectorshift' : vectorshift, 'vectorshift_ab' : vectorshift_ab, 
                        'vectorshift_ab_J0' : vectorshift_ab_J0,
                        'vectorshift_len' : vectorshift_len, 'vectorshift_ab_len' : vectorshift_ab_len,
                        'vectorshift_ab_J0_len' : vectorshift_ab_J0_len,
                         'vectorshift_len_DEnormed' : vectorshift_len/DE_Jab_avg, 
                         'vectorshift_ab_len_DEnormed' : vectorshift_ab_len/DE_ab_avg, 
                         'vectorshift_ab_J0_len_DEnormed' : vectorshift_ab_J0_len/DE_ab_avg},
           'pixeldata' : {'grid' : gridp, 'idx' : idxp,'Jab': jabp, 
                          'samplenrs' : pixelsamplenrs,'IDs': pixelIDs},
           'fielddata' : {'vectorfield' : {'axr' : axr, 'bxr' : bxr, 'axt' : axt, 'bxt' : bxt}}
           }
    return data

