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
Module with color rendition, fidelity and gamut area helper functions
=====================================================================

 :_get_hue_bin_data(): Slice gamut spanned by the sample jabt, jabr and calculate hue-bin data.

 :_hue_bin_data_to_rxhj(): Calculate hue bin measures: Rcshj, Rhshj, Rfhj, DEhj
     
 :_hue_bin_data_to_rfi(): Get sample color differences DEi and calculate color fidelity values Rfi.

 :_hue_bin_data_to_rg():  Calculates gamut area index, Rg.

 :spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated
                    with test source and its reference illuminant.

 :spd_to_rg(): Calculates the color gamut index of spectral data 
               for a sample set illuminated with test source (data) 
               with respect to some reference illuminant.

 :spd_to_DEi(): Calculates color difference (~fidelity) of spectral data 
                between sample set illuminated with test source (data) 
                and some reference illuminant.

 :optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type 
                           such that average Rf for a set of light sources is 
                           the same as that of a target-cri (default: 'ciera')

 :spd_to_cri(): Calculates the color rendering fidelity index 
                (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
                Can also output Rg, Rfhi, Rcshi, Rhshi, cct, duv, ...

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import copy
from luxpy import (_S_INTERP_TYPE, _CRI_RFL, _IESTM3015, math, cam, cat,
                   spd, colortf, spd_to_xyz, cie_interp, cri_ref, xyz_to_cct)
from luxpy.utils import np, sp,plt,asplit, np2d, put_args_in_db 
from luxpy.color.cri.utils.DE_scalers import linear_scale, log_scale, psy_scale

from luxpy.color.cri.utils.init_cri_defaults_database import _CRI_TYPE_DEFAULT, _CRI_DEFAULTS, process_cri_type_input

__all__ = ['_get_hue_bin_data','spd_to_jab_t_r','spd_to_rg', 'spd_to_DEi', 
           'optimize_scale_factor','spd_to_cri',
           '_hue_bin_data_to_rxhj', '_hue_bin_data_to_rfi', '_hue_bin_data_to_rg']

#------------------------------------------------------------------------------
def _get_hue_bin_data_individual_samples(jabt,jabr, normalized_chroma_ref = 100):
    """ Helper function to return dict with required keys when nhbins = None in call to _get_hue_bin_data"""
    
    # get hues of jabt, jabr:
    ht = cam.hue_angle(jabt[...,1], jabt[...,2], htype = 'rad')
    hr = cam.hue_angle(jabr[...,1], jabr[...,2], htype = 'rad')
    
    # Get chroma of jabt, jabr:
    Ct = ((jabt[...,1]**2 + jabt[...,2]**2))**0.5
    Cr = ((jabr[...,1]**2 + jabr[...,2]**2))**0.5
    
    
    # Calculate DEi between jabt, jabr:
    DEi = ((jabt - jabr)**2).sum(axis = -1, keepdims = True)**0.5
    
    jabt_hj, jabr_hj, DE_hj = jabt, jabr, DEi
    
    # some dummy variables:
    start_hue = 0
    dh = None
    hue_bin_edges = None
    nhbins = jabt.shape[0]
    ht_idx, hr_idx = np.arange(nhbins)[:,None], np.arange(nhbins)[:,None]
    
    # calculate normalized hue-bin averages for jabt, jabr:
    ht_hj = cam.hue_angle(jabt_hj[...,1],jabt_hj[...,2],htype='rad')
    hr_hj = cam.hue_angle(jabr_hj[...,1],jabr_hj[...,2],htype='rad')
    Ct_hj = ((jabt_hj[...,1]**2 + jabt_hj[...,2]**2))**0.5
    Cr_hj = ((jabr_hj[...,1]**2 + jabr_hj[...,2]**2))**0.5
    Ctn_hj = normalized_chroma_ref*Ct_hj/(Cr_hj + 1e-308) # calculate normalized chroma for samples under test
    Ctn_hj[Cr_hj == 0.0] = np.inf
    jabtn_hj = jabt_hj.copy()
    jabrn_hj = jabr_hj.copy()
    jabtn_hj[...,1], jabtn_hj[...,2] = Ctn_hj*np.cos(ht_hj), Ctn_hj*np.sin(ht_hj)
    jabrn_hj[...,1], jabrn_hj[...,2] = normalized_chroma_ref*np.cos(hr_hj), normalized_chroma_ref*np.sin(hr_hj)
    
    # calculate normalized versions of jabt, jabr:
    jabtn = jabt.copy()
    jabrn = jabr.copy()
    Ctn = np.zeros((jabt.shape[0],jabt.shape[1]))

    Crn = Ctn.copy()
    for j in range(nhbins):
        Ctn = Ctn + (Ct/Cr_hj[j,...])*(hr_idx==j)
        Crn = Crn + (Cr/Cr_hj[j,...])*(hr_idx==j)
    Ctn*=normalized_chroma_ref
    Crn*=normalized_chroma_ref

    jabtn[...,1] = (Ctn*np.cos(ht))
    jabtn[...,2] = (Ctn*np.sin(ht))
    jabrn[...,1] = (Crn*np.cos(hr))
    jabrn[...,2] = (Crn*np.sin(hr))

    # closed jabt_hj, jabr_hj for Rg:
    jabt_hj_closed = np.vstack((jabt_hj,jabt_hj[:1,...]))
    jabr_hj_closed = np.vstack((jabr_hj,jabr_hj[:1,...]))
    
    # closed jabtn_hj, jabrn_hj for plotting:
    jabtn_hj_closed = np.vstack((jabtn_hj,jabtn_hj[:1,...]))
    jabrn_hj_closed = np.vstack((jabrn_hj,jabrn_hj[:1,...]))
    
    return {'jabt' : jabt, 'jabr' : jabr, 
            'jabtn' : jabtn, 'jabrn' : jabrn,
            'DEi' : DEi[...,0], 
            'Ct' : Ct, 'Cr': Cr, 'ht' : ht, 'hr' : hr, 
            'ht_idx' : ht_idx, 'hr_idx' : hr_idx,
            'jabt_hj' : jabt_hj, 'jabr_hj' : jabr_hj, 'DE_hj' : DE_hj,
            'jabt_hj_closed' : jabt_hj_closed, 'jabr_hj_closed' : jabr_hj_closed,
            'jabtn_hj' : jabtn_hj, 'jabrn_hj' : jabrn_hj,
            'jabtn_hj_closed' : jabtn_hj_closed, 'jabrn_hj_closed' : jabrn_hj_closed,
            'ht_hj' : ht_hj, 'hr_hj' : hr_hj, 
            'Ct_hj': Ct_hj, 'Cr_hj' : Cr_hj, 'Ctn_hj': Ctn_hj,
            'nhbins' : nhbins, 'start_hue' : start_hue, 
            'normalized_chroma_ref' : normalized_chroma_ref, 
            'dh' : dh, 'hue_bin_edges' : hue_bin_edges, 
            'hbinnrs' : hr_idx}
    
def _get_hue_bin_data(jabt, jabr, start_hue = 0, nhbins = 16,
                      normalized_chroma_ref = 100):
    """
    Slice gamut spanned by the sample jabt, jabr and calculate hue-bin data.
    
    Args:
        :jabt: 
            | ndarray with jab sample data under test illuminant
        :jabr: 
            | ndarray with jab sample data under reference illuminant
        :start_hue:
            | 0.0 or float, optional
            | Hue angle to start bin slicing
        :nhbins:
            | None or int, optional
            |   - None: defaults to using the sample hues themselves as 'bins'. 
            |           In other words, the number of bins will be equal to the 
            |           number of samples.
            |   - float: number of bins to slice the sample gamut in.
        :normalized_chroma_ref:
            | 100.0 or float, optional
            | Controls the size (chroma/radius) of the normalization circle/gamut.
    
    Returns:
        :dict:
            | Dictionary with keys:
            | 
            | - 'jabt', 'jabr': ndarrays with jab sample data under test & ref. illuminants
            | - 'DEi': ndarray with sample jab color difference between test and ref.
            | - 'Ct', 'Cr': chroma for each sample under test and ref.
            | - 'ht', 'hr': hue angles (rad.) for each sample under test and ref.
            | - 'ht_idx', 'hr_idx': hue bin indices for each sample under test and ref.
            | - 'jabt_hj', 'jabr_hj': ndarrays with hue-bin averaged jab's under test & ref. illuminants
            | - 'DE_hj' : ndarray with average  sample DE in each hue bin
            | - 'jabt_hj_closed', 'jabr_hj_closed': ndarrays with hue-bin averaged jab's under test & ref. illuminants (closed gamut: 1st == last)
            | - 'jabtn_hj', 'jabrn_hj': ndarrays with hue-bin averaged and normalized jab's under test & ref. illuminants 
            | - 'jabtn_hj_closed', 'jabrn_hj_closed': ndarrays with hue-bin and normalized averaged jab's under test & ref. illuminants (closed gamut: 1st == last)
            | - 'ht_hj', 'hr_hj': hues (rad.) for each hue bin for test and ref.
            | - 'Ct_hj', 'Cr_hj': chroma for each hue bin for test and ref.
            | - 'Ctn_hj' : normalized chroma for each hue bin for test (ref = normalized_chroma_ref)
            | - 'nhbins': number of hue bins
            | - 'start_hue' : start hue for bin slicing
            | - 'normalized_chroma_ref': normalized chroma value for ref.
            | - 'dh': hue-angle arcs (°)
            | - 'hue_bin_edges': hue bin edge (rad)
            | - 'hbinnrs':  hue bin indices for each sample under ref. (= hr_idx)
    """
    
    if nhbins is None:
        return _get_hue_bin_data_individual_samples(jabt,jabr,normalized_chroma_ref = normalized_chroma_ref)
    else:
        # calculate hue-bin width, edges:
        dh = 360/nhbins 
        hue_bin_edges = np.arange(start_hue, 360 + 1, dh)*np.pi/180
        
    # get hues of jabt, jabr:
    ht = cam.hue_angle(jabt[...,1], jabt[...,2], htype = 'rad')
    hr = cam.hue_angle(jabr[...,1], jabr[...,2], htype = 'rad')
    
    # Get chroma of jabt, jabr:
    Ct = ((jabt[...,1]**2 + jabt[...,2]**2))**0.5
    Cr = ((jabr[...,1]**2 + jabr[...,2]**2))**0.5
    
    
    # Calculate DEi between jabt, jabr:
    DEi = ((jabt - jabr)**2).sum(axis = -1, keepdims = True)**0.5

    # calculate hue-bin averages for jabt, jabr:
    jabt_hj = np.ones((nhbins,ht.shape[1],3))*np.nan
    jabr_hj = np.ones((nhbins,hr.shape[1],3))*np.nan
    DE_hj = np.ones((nhbins,hr.shape[1]))*np.nan
    ht_idx = np.ones_like((ht))*np.nan
    hr_idx = np.ones_like((hr))*np.nan
    n = hr_idx.shape[-1]

    for j in range(nhbins):
        cndt_hj = (ht>=hue_bin_edges[j]) & (ht<hue_bin_edges[j+1])
        cndr_hj = (hr>=hue_bin_edges[j]) & (hr<hue_bin_edges[j+1])

        ht_idx[cndt_hj] = j # store hue bin indices for all samples
        hr_idx[cndr_hj] = j
        #wt = np.sum(cndt_hj,axis=0,keepdims=True).astype(np.float)
        wr = np.nansum(cndr_hj,axis=0,keepdims=True).astype(np.float)

        #wt[wt==0] = np.nan
        wr[wr==0] = np.nan

        jabt_hj[j,...] = np.sum((jabt * cndr_hj[...,None]), axis=0)/wr.T # must use ref. bins !!!
        jabr_hj[j,...] = np.sum((jabr * cndr_hj[...,None]), axis=0)/wr.T
        DE_hj[j,...] = np.nansum((DEi * cndr_hj[...,None])/wr.T, axis = 0).T # local color difference is average of DEi per hue bin !!
        DE_hj[j,np.isnan(wr[0])] = np.nan # signal empty hue bins with a NaN
        
    # calculate normalized hue-bin averages for jabt, jabr:
    ht_hj = cam.hue_angle(jabt_hj[...,1],jabt_hj[...,2],htype='rad')
    hr_hj = cam.hue_angle(jabr_hj[...,1],jabr_hj[...,2],htype='rad')
    Ct_hj = ((jabt_hj[...,1]**2 + jabt_hj[...,2]**2))**0.5
    Cr_hj = ((jabr_hj[...,1]**2 + jabr_hj[...,2]**2))**0.5
    Ctn_hj = normalized_chroma_ref*Ct_hj/(Cr_hj + 1e-308) # calculate normalized chroma for samples under test
    Ctn_hj[Cr_hj == 0.0] = np.inf
    jabtn_hj = jabt_hj.copy()
    jabrn_hj = jabr_hj.copy()
    jabtn_hj[...,1], jabtn_hj[...,2] = Ctn_hj*np.cos(ht_hj), Ctn_hj*np.sin(ht_hj)
    jabrn_hj[...,1], jabrn_hj[...,2] = normalized_chroma_ref*np.cos(hr_hj), normalized_chroma_ref*np.sin(hr_hj)
    
    # calculate normalized versions of jabt, jabr:
    jabtn = jabt.copy()
    jabrn = jabr.copy()
    Ctn = np.zeros((jabt.shape[0],jabt.shape[1]))
    Crn = Ctn.copy()
    for j in range(nhbins):
        Ctn = Ctn + (Ct/Cr_hj[j,...])*(hr_idx==j)
        Crn = Crn + (Cr/Cr_hj[j,...])*(hr_idx==j)
    Ctn*=normalized_chroma_ref
    Crn*=normalized_chroma_ref
    jabtn[...,1] = (Ctn*np.cos(ht))
    jabtn[...,2] = (Ctn*np.sin(ht))
    jabrn[...,1] = (Crn*np.cos(hr))
    jabrn[...,2] = (Crn*np.sin(hr))
    # plt.plot(jabtn[:,0,1],jabtn[:,0,2],'b+')
    # plt.plot(jabrn[:,0,1],jabrn[:,0,2],'rx')
    # plt.plot(jabtn_hj[:,0,1],jabtn_hj[:,0,2],'bo-')
    # plt.plot(jabrn_hj[:,0,1],jabrn_hj[:,0,2],'ro-')
    # plt.axis('equal')

    # closed jabt_hj, jabr_hj for Rg:
    jabt_hj_closed = np.vstack((jabt_hj,jabt_hj[:1,...]))
    jabr_hj_closed = np.vstack((jabr_hj,jabr_hj[:1,...]))
    
    # closed jabtn_hj, jabrn_hj for plotting:
    jabtn_hj_closed = np.vstack((jabtn_hj,jabtn_hj[:1,...]))
    jabrn_hj_closed = np.vstack((jabrn_hj,jabrn_hj[:1,...]))

    return {'jabt' : jabt, 'jabr' : jabr, 
            'jabtn' : jabtn, 'jabrn' : jabrn,
            'DEi' : DEi[...,0], 
            'Ct' : Ct, 'Cr': Cr, 'ht' : ht, 'hr' : hr, 
            'ht_idx' : ht_idx, 'hr_idx' : hr_idx,
            'jabt_hj' : jabt_hj, 'jabr_hj' : jabr_hj, 'DE_hj' : DE_hj,
            'jabt_hj_closed' : jabt_hj_closed, 'jabr_hj_closed' : jabr_hj_closed,
            'jabtn_hj' : jabtn_hj, 'jabrn_hj' : jabrn_hj,
            'jabtn_hj_closed' : jabtn_hj_closed, 'jabrn_hj_closed' : jabrn_hj_closed,
            'ht_hj' : ht_hj, 'hr_hj' : hr_hj, 
            'Ct_hj': Ct_hj, 'Cr_hj' : Cr_hj, 'Ctn_hj': Ctn_hj,
            'nhbins' : nhbins, 'start_hue' : start_hue, 
            'normalized_chroma_ref' : normalized_chroma_ref, 
            'dh' : dh, 'hue_bin_edges' : hue_bin_edges, 
            'hbinnrs' : hr_idx}

#------------------------------------------------------------------------------
def _polyarea(x,y):
    """
    Calculate area of polygon with coordinates (x,y).
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1,axis=0))-np.dot(y,np.roll(x,1,axis=0)))

#------------------------------------------------------------------------------
def _hue_bin_data_to_rg(hue_bin_data, max_scale = 100, normalize_gamut = False):
    """
    Calculates gamut area index, Rg.
    
    Args:
        :hue_bin_data:
            | Dict with hue bin data obtained with _get_hue_bin_data().
        :max_scale:
            | 100.0, optional
            | Value of Rg when Rf = max_scale (i.e. DEavg = 0)
        :normalize_gamut:
            | False, optional
            | True normalizes the gamut of test to that of ref.
            | (perfect agreement results in circle).
        :out: 
            | 'Rg', optional
            | Specifies which variables to output as ndarray

    Returns: 
        :Rg: 
            | float or ndarray with gamut area indices Rg.
    """ 
    if normalize_gamut == False:
        jabt_hj, jabr_hj = hue_bin_data['jabt_hj_closed'], hue_bin_data['jabr_hj_closed']
    else:
        jabt_hj, jabr_hj = hue_bin_data['jabtn_hj_closed'], hue_bin_data['jabrn_hj_closed']

    notnan_t = np.logical_not(np.isnan(jabt_hj[...,1])) # avoid NaN's (i.e. empty hue-bins)
    notnan_r = np.logical_not(np.isnan(jabr_hj[...,1]))
    
    Rg = np.array([[max_scale*_polyarea(jabt_hj[notnan_t[:,i],i,1],jabt_hj[notnan_t[:,i],i,2]) / _polyarea(jabr_hj[notnan_r[:,i],i,1],jabr_hj[notnan_r[:,i],i,2]) for i in range(notnan_r.shape[-1])]])
    
    return Rg

#------------------------------------------------------------------------------
def _hue_bin_data_to_rxhj(hue_bin_data, cri_type = _CRI_TYPE_DEFAULT,
                             scale_factor = None, scale_fcn = None,
                             use_bin_avg_DEi = True):
    """
    Calculate hue bin measures: Rcshj, Rhshj, Rfhj, DEhj.
     
    |   Rcshj: local chroma shift
    |   Rhshj: local hue shift
    |   Rfhj: local (hue bin) color fidelity  
    |   DEhj: local (hue bin) color differences 
    |
    |   (See IES TM30)
    
    Args:
        :hue_bin_data:
            | Dict with hue bin data obtained with _get_hue_bin_data().
        :use_bin_avg_DEi: 
            | True, optional
            | Note that following IES-TM30 DEhj from gamut_slicer() is obtained by
            | averaging the DEi per hue bin (True), and NOT by averaging the 
            | jabt and jabr per hue  bin and then calculating the DEhj (False).
            | If None: use value in rg_pars dict in cri_type dict!
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :scale_factor:
            | factors used in scaling function
        
    Returns:
        :returns: 
            | ndarrays of Rcshj, Rhshj, Rfhj, DEhj 
        
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    """
    
    if (scale_factor is None) | (scale_fcn is None) | (use_bin_avg_DEi is None):
        if isinstance(cri_type, str): 
           args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
           cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri._hue_bin_data_to_Ri')
        
        # Get scale factor and function:
        if (scale_factor is None):
            scale_factor = cri_type['scale']['cfactor']
        
        if (scale_fcn is None):
            scale_fcn = cri_type['scale']['fcn']    
        
        if (use_bin_avg_DEi is None) & ('rg_pars' in cri_type): 
            use_bin_avg_DEi = cri_type['rg_pars']['use_bin_avg_DEi']
        else:
            raise Exception('Define use_bin_avg_DEi in rg_pars dict in cri_type dict or set use_bin_avg_DEi kwarg to not None!')

    nhbins = hue_bin_data['nhbins']
    start_hue = hue_bin_data['start_hue']
    
    # A. Local color fidelity, Rfhj:
    if use_bin_avg_DEi:
        DEhj = hue_bin_data['DE_hj']
    else:
        DEhj = ((hue_bin_data['jabt_hj']-hue_bin_data['jabr_hj'])**2).sum(axis=-1)**0.5
    Rfhj = scale_fcn(DEhj, scale_factor = scale_factor)
    
    # B.Local chroma shift and hue shift, [Rcshi, Rhshi]:
    # B.1 relative paths:
    dab = (hue_bin_data['jabt_hj']- hue_bin_data['jabr_hj'])[...,1:]/(hue_bin_data['Cr_hj'][...,None] + 1e-308)

    # B.2 Reference unit circle:
    hbincenters = np.arange(start_hue + np.pi/nhbins, 2*np.pi, 2*np.pi/nhbins)[...,None]
    arc = np.cos(hbincenters)
    brc = np.sin(hbincenters)

    # B.3 calculate local chroma shift, Rcshi:
    Rcshj = dab[...,0] * arc + dab[...,1] * brc
    
    # B.4 calculate local hue shift, Rcshi:
    Rhshj = dab[...,1] * arc - dab[...,0] * brc
    
    return Rcshj, Rhshj, Rfhj, DEhj 

#------------------------------------------------------------------------------
def _hue_bin_data_to_ellipsefit(hue_bin_data):
    """
    Fit ellipse to normalized color gamut,
    and calculate orientation angle (°) and eccentricity (ellipse a-axis / ellipse b-axis)
    
    Args:
        :hue_bin_data:
            | Dict with hue bin data obtained with _get_hue_bin_data().
    
    Returns:
        dict:
            | {'v':v, 'a/b': ecc,'thetad': theta}
            | v is an ndarray with [a,b, xc, yc, theta(rad)] describing the ellipse
            | 'a/b' is the eccentricity
            | 'thetad' is the angle in degrees, [0°,180°]
    """
    # use get chroma-normalized jabtn_hj:
    jabt = hue_bin_data['jabtn_hj']
    ecc = np.ones((1,jabt.shape[1]))*np.nan
    theta = np.ones((1,jabt.shape[1]))*np.nan
    v = np.ones((jabt.shape[1],5))*np.nan
    for i in range(jabt.shape[1]):
        try:
            v[i,:] = math.fit_ellipse(jabt[:,i,1:])
            a,b = v[i,0], v[i,1] # major and minor ellipse axes
            ecc[0,i] = a/b
            theta[0,i] = np.rad2deg(v[i,4]) # orientation angle
            if theta[0,i]>180: theta[0,i] = theta[0,i] - 180
        except:
            v[i,:] = np.nan*np.ones((1,5))
            ecc[0,i] = np.nan
            theta[0,i] = np.nan # orientation angle
    return {'v':v, 'a/b':ecc,'thetad': theta}


#------------------------------------------------------------------------------
def _hue_bin_data_to_rfi(hue_bin_data = None, cri_type = _CRI_TYPE_DEFAULT,
                        scale_factor = None, scale_fcn = None):
    """
    Get sample color differences DEi and calculate color fidelity values Rfi.
     
    |   Rfi: Sample color fidelity  
    |   DEi: Sample color differences 
    |
    |   (See IES TM30)
    
    Args:
        :hue_bin_data:
            | Dict with hue bin data obtained with _get_hue_bin_data().
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :scale_factor:
            | factors used in scaling function
        
    Returns:
        :returns: 
            | ndarrays of Rfi, DEi 
        
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    """
    if (scale_factor is None) | (scale_fcn is None):
        if isinstance(cri_type, str): 
           args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
           cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri._hue_bin_data_to_Rfi')
        
        # Get scale factor and function:
        if (scale_factor is None):
            scale_factor = cri_type['scale']['cfactor']
        
        if (scale_fcn is None):
            scale_fcn = cri_type['scale']['fcn']   
    
    # Color fidelity, Rfi:
    DEi = hue_bin_data['DEi']
    Rfi = scale_fcn(DEi, scale_factor = scale_factor)
    
    return Rfi, DEi

#------------------------------------------------------------------------------
def _hue_bin_data_to_rf(hue_bin_data = None, cri_type = _CRI_TYPE_DEFAULT,
                        scale_factor = None, scale_fcn = None, avg = None,
                        out = 'Rf,DEa'):
    """
    Get average sample color difference DEa and calculate color fidelity index Rf.
     
    |   Rf: color fidelity index
    |   DEa: average color difference 
    |
    |   (See IES TM30)
    
    Args:
        :hue_bin_data:
            | Dict with hue bin data obtained with _get_hue_bin_data().
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :scale_factor:
            | factors used in scaling function
        :avg:
            | Averaging function for DEi -> DEa.
        :out: 
            | 'Rf,DEa' or str, optional
            | Specifies requested output  
        
    Returns:
        :returns: 
            | ndarrays of Rf, DEa
        
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    """
    if (scale_factor is None) | (scale_fcn is None) | (avg is None):
        if isinstance(cri_type, str): 
           args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
           cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri._hue_bin_data_to_Rfi')
        
        # Get scale factor and function:
        if (scale_factor is None):
            scale_factor = cri_type['scale']['cfactor']
        
        if (scale_fcn is None):
            scale_fcn = cri_type['scale']['fcn'] 
            
        # Get averaging function:
            avg = cri_type['avg'] 
    
    # Color fidelity, Rfi:
    DEa = np2d(avg(hue_bin_data['DEi'], axis = 0))
    Rf = np2d(scale_fcn(DEa, scale_factor = scale_factor))
  
    # output:
    if out == 'Rf,DEa':
        return Rf, DEa
    elif out == 'Rf':
        return  Rf
    elif out == 'DEa':
        return DEa

def spd_to_jab_t_r(St, cri_type = _CRI_TYPE_DEFAULT, out = 'jabt,jabr', 
                   wl = None, sampleset = None, ref_type = None, 
                   cieobs  = None, cspace = None, catf = None, 
                   cri_specific_pars = None):
    """
    Calculates jab color values for a sample set illuminated with test source 
    SPD and its reference illuminant.
        
    Args:
        :St: 
            | ndarray with spectral data 
            | (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'jabt,jabr' or str, optional
            | Specifies requested output (e.g.'jabt,jabr' or 'jabt,jabr,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the spds in St to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
            |  override default values in cri_type dict.
            
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
    
    Returns:
        :returns: 
            | (ndarray, ndarray) 
            | with jabt and jabr data for :out: 'jabt,jabr'
            | 
            | Other output is also possible by changing the :out: str value.
    """
   
    #Override input parameters with data specified in cri_type:
    args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_jab_t_r')
    
    # unpack and update dict with parameters:
    (avg, catf, cieobs,
     cri_specific_pars, cspace, 
     ref_type, rg_pars, sampleset, scale) = [cri_type[x] for x in sorted(cri_type.keys())] 

    # pre-interpolate SPD:
    if wl is not None: 
        St = cie_interp(St, wl_new = wl, kind = _S_INTERP_TYPE)
      
    # obtain sampleset:
    if isinstance(sampleset,str):
        sampleset = eval(sampleset)
    
    # A. calculate reference illuminant:
    # A.a. get xyzw:
    xyztw_cct = spd_to_xyz(St, cieobs = cieobs['cct'], rfl = None, out = 1)

    # A.b. get cct:
    cct, duv = xyz_to_cct(xyztw_cct, cieobs = cieobs['cct'], out = 'cct,duv',mode = 'lut')
    
    # A.c. get reference ill.:
    if isinstance(ref_type,np.ndarray):
        Sr = cri_ref(ref_type, ref_type = 'spd', cieobs = cieobs['cct'], wl3 = St[0])
    else:
        Sr = cri_ref(cct, ref_type = ref_type, cieobs = cieobs['cct'], wl3 = St[0])

    # B. calculate xyz and xyzw of SPD and Sr (stack for speed):
    xyzi, xyzw = spd_to_xyz(np.vstack((St,Sr[1:])), cieobs = cieobs['xyz'], rfl = sampleset, out = 2)
    xyzri, xyzrw = spd_to_xyz(Sr, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)
    N = St.shape[0]-1
    xyzti, xyzri =  xyzi[:,:N,:], xyzi[:,N:,:]
    xyztw, xyzrw =  xyzw[:N,:], xyzw[N:,:]
    
    # C. apply chromatic adaptation for non-cam/lab cspaces:
    if catf is not None:
        D_cat, Dtype_cat, La_cat, catmode_cat, cattype_cat, mcat_cat, xyzw_cat = [catf[x] for x in sorted(catf.keys())]
        
        #if not isinstance(D_cat,list): D_cat = [D_cat]
        if xyzw_cat is None: #transform from xyzwt --> xyzwr
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
        
        else: # transform both xyzwr and xyzwt to xyzw_cat
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)

    # D. convert xyz to colorspace, cam or chromaticity co. lab (i.e. lab, ipt, Yuv, jab, wuv,..):
    # D.a. broadcast xyzw to shape of xyzi:
    # xyztw = xyztw[None] 
    # xyzrw = xyzrw[None] 

    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyztw # enter test whitepoint
    jabt = colortf(xyzti, tf = cspace['type'], fwtf = cspace_pars)
    
    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyzrw # enter ref. whitepoint
    jabr = colortf(xyzri, tf = cspace['type'], fwtf = cspace_pars)    
    del cspace_pars


    # E. Regulate output:
    if out == 'jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr':
        return jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr
    elif out == 'jabt,jabr,cct,duv,Sr':
        return jabt,jabr,cct,duv,Sr
    elif out == 'jabt,jabr,cct,duv':
        return jabt,jabr,cct,duv
    elif out == 'jabt,jabr':
        return jabt, jabr
    else:
        eval(out)


#------------------------------------------------------------------------------
def spd_to_DEi(St, cri_type = _CRI_TYPE_DEFAULT, out = 'DEi', wl = None, \
               sampleset = None, ref_type = None, cieobs = None, avg = None, \
               cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates color differences (~fidelity), DEi, of spectral data.
    
    Args:
        :St: 
            | ndarray with spectral data 
            | (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'DEi' or str, optional
            | Specifies requested output (e.g. 'DEi,DEa,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the spds in St to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
              override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
    
    Returns:
        :returns: 
            | float or ndarray with DEi for :out: 'DEi'
            | 
            | Other output is also possible by changing the :out: str value.

    """
    #Override input parameters with data specified in cri_type:
    args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_DEi')

    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    (jabt, jabr, 
     xyzti, xyztw, 
     xyzri, xyzrw,
     xyztw_cct,
     cct, duv, St, Sr) = spd_to_jab_t_r(St, wl = wl, cri_type = cri_type, 
                                        out = 'jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr')
      
    # E. calculate DEi, DEa:
    DEi = ((jabt - jabr)**2).sum(axis = -1)**0.5
    DEa = np2d(cri_type['avg'](DEi, axis = 0))
  
     # output:
    if out == 'DEi,DEa,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr':
        return DEi,DEa,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr
    elif out == 'DEa':
        return DEa
    elif out == 'DEi':
        return DEi
    elif out == 'DEi,DEa':
        return DEi,DEa
    elif out == 'DEa,DEi':
        return DEa,DEi
    elif out == 'DEi,DEa,jabt,jabr,cct,duv,Sr':
        return DEi,DEa,jabt,jabr,cct,duv,Sr
    else:
        return  eval(out)


#------------------------------------------------------------------------------
def optimize_scale_factor(cri_type, opt_scale_factor, scale_fcn, avg) :
    """
    Optimize scale_factor of cri-model in cri_type 
    such that average Rf for a set of light sources is the same as that 
    of a target-cri (default: 'ciera').
    
    Args:
        :cri_type: 
            | str or dict
            |   -'str: specifies dict with default cri model parameters 
            |       (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |       (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |       for required structure)
        :opt_scale:
            | True or False
            | True: optimize scaling-factor, else do nothing and use value of 
            | scaling-factor in :scale: dict.   
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :avg: 
            | None or fcn handle
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.

    Returns:
        :scaling_factor: 
            | ndarray

    """

    if np.any(opt_scale_factor):
        if 'opt_cri_type' not in cri_type['scale'].keys(): 
            opt_cri_type = _CRI_DEFAULTS['ciera'] # use CIE Ra-13.3-1995 as target
        else:
            if isinstance(cri_type['scale']['opt_cri_type'],str):
                opt_cri_type = _CRI_DEFAULTS[cri_type['scale']['opt_cri_type']]
            else: #should be dict !!
                opt_cri_type = cri_type['scale']['opt_cri_type']
        if 'opt_spd_set' not in cri_type['scale'].keys(): 
            opt_spd_set = _IESTM3015['S']['data'][0:13] # use CIE F1-F12
        else:
            opt_spd_set = cri_type['scale']['opt_spd_set']
        
        scale_fcn_opt = opt_cri_type ['scale']['fcn']
        scale_factor_opt = opt_cri_type ['scale']['cfactor']
        avg_opt = opt_cri_type ['avg']
        DEa_opt = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = opt_cri_type) # DEa using target cri
        Rf_opt = avg(scale_fcn_opt(DEa_opt,scale_factor_opt))
        
        DEa = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = cri_type) # DEa using current cri

        
        # optimize scale_factor to minimize rms difference:
        sf = cri_type['scale']['cfactor'] # get scale_factor of cri_type to determine len and non-optimized factors
        
        if sf is None: sf = [None]
        
        if (isinstance(sf,float)): #(isinstance(1.0*sf,float))
            sf = [sf]
        if isinstance(opt_scale_factor, bool):
            opt_scale_factor = [opt_scale_factor] 
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,x)) - Rf_opt,axis=1) # optimize the only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[1:]) ))) - Rf_opt,axis=1) # optimize the first cfactor (for scale_factor input of len = 1)
        else:
            x0 = np.ones(np.sum(opt_scale_factor))
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[np.invert(opt_scale_factor)]) ))) - Rf_opt,axis=1) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)
        
        optresult = sp.optimize.minimize(fun = optfcn, x0 = x0, args=(), method = 'Nelder-Mead')
        scale_factor = optresult['x']

        #Reconstruct 'scale_factor' from optimized and fixed parts:
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            pass #only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            scale_factor = np.hstack( (scale_factor,sf[1:]) )
        else:
          scale_factor = np.hstack( (scale_factor,sf[np.invert(opt_scale_factor)]) ) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)

    else:
        scale_factor = cri_type['scale']['cfactor']
    return scale_factor

#------------------------------------------------------------------------------
def spd_to_rg(St, cri_type = _CRI_TYPE_DEFAULT, out = 'Rg', wl = None, \
              sampleset = None, ref_type = None, cieobs  = None, avg = None, \
              cspace = None, catf = None, cri_specific_pars = None, rg_pars = None,
              fit_gamut_ellipse = False):
    """
    Calculates the color gamut index, Rg, of spectral data. 
    
    Args:
        :St: 
            | ndarray with spectral data 
            | (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'Rg' or str, optional
            | Specifies requested output (e.g. 'Rg,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
            | override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
        :rg_pars: 
            | None or dict, optional
            | Dict containing specifying parameters for slicing the gamut.
            | Dict structure: 
            |     {'nhbins' : None, 'start_hue' : 0, 
            |       'normalize_gamut' : False, 'normalized_chroma_ref': 100.0}
            |    - key: 'nhbins': int, number of hue bins to slice gamut 
            |                 (None use the one specified in :cri_type: dict).
            |    - key: 'start_hue': float (°), hue at which to start slicing
            |    - key: 'normalize_gamut': True or False: 
            |                normalize gamut or not before calculating a gamut 
            |                area index Rg. 
            |    - key: 'normalized_chroma_ref': 100.0 or float, optional
            |                Controls the size (chroma/radius) 
            |                of the normalization circle/gamut.
        :avg: 
            | None or fcn handle, optional
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.
        :scale:
            | None or dict, optional
            | Specifies scaling of color differences to obtain CRI.
            |     - None use the one specified in :cri_type: dict.
            |     - dict: user specified dict with scaling parameters.
            |         - key: 'fcn': function handle to type of cri scale, 
            |                 e.g. 
            |                 * linear()_scale --> (100 - scale_factor*DEi), 
            |                 * log_scale --> (cfr. Ohno's CQS), 
            |                 * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
            |        - key: 'cfactor': factors used in scaling function, 
            |              If None: 
            |                     Scaling factor value(s) will be optimized to 
            |                     minimize the rms between the Rf's of the 
            |                     requested metric and the target metric specified
            |                     in:
            |
            |                  - key: 'opt_cri_type':  str 
            |                      * str: one of the preset _CRI_DEFAULTS
            |                      * dict: user speciied 
            |                      (dict must contain all keys as normal)
            |                     Note that if key not in :scale: dict, 
            |                     then 'opt_cri_type' is added with default 
            |                     setting = 'ciera'.
            |                 - key: 'opt_spd_set': ndarray with set of light 
            |                     source spds used to optimize cfactor. 
            |                     Note that if key not in :scale: dict, 
            |                     then default = 'F1-F12'.
        :fit_gamut_ellipse:
            | fit ellipse to normalized color gamut 
            | (extract from function using out; also stored in hue_bin_data['gamut_ellipse_fit'])

    Returns:
        :returns:
            | float or ndarray with Rg for :out: 'Rg'
            | Other output is also possible by changing the :out: str value.
            | E.g. out == 'Rg,data' would output an ndarray with Rg values 
            |               and a dictionary :data: with keys:
            |                   'St', 'Sr', 'cct', 'duv', 'hue_bin_data' 
            |                   'xyzti', xyzti, 'xyztw', 'xyzri', 'xyzrw'
            
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
    """
    #Override input parameters with data specified in cri_type:
    args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_rg')

    #avg, catf, cieobs, cieobs_cct, cri_specific_pars, cspace, cspace_pars, ref_type, rg_pars, sampleset, scale_factor, scale_fcn = [cri_type[x] for x in sorted(cri_type.keys())] 

       
    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    (jabt,jabr,
     xyzti,xyztw,
     xyzri,xyzrw,
     xyztw_cct,
     cct,duv,St,Sr) = spd_to_jab_t_r(St, wl = wl, cri_type = cri_type, 
                                          out = 'jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr') 

    # calculate gamut area index:
    rg_pars = cri_type['rg_pars']
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue  = [rg_pars[x] for x in sorted(rg_pars.keys())]
    
    # get hue_bin_data:
    hue_bin_data = _get_hue_bin_data(jabt, jabr, 
                                    start_hue = start_hue, nhbins = nhbins,
                                    normalized_chroma_ref = normalized_chroma_ref)
    
    Rg = _hue_bin_data_to_rg(hue_bin_data, normalize_gamut = normalize_gamut)
    
    if fit_gamut_ellipse:
        gamut_ellipse_fit = _hue_bin_data_to_ellipsefit(hue_bin_data)
        hue_bin_data['gamut_ellipse_fit'] = gamut_ellipse_fit
    else:
        gamut_ellipse_fit = {}
    
    if 'data' in out:
        data = {'St' : St, 'Sr' : Sr, 'xyztw_cct' : xyztw_cct, 
                'cct' : cct, 'duv' : duv,
                'xyzti' : xyzti, 'xyztw' : xyztw, 
                'xyzri' : xyzri, 'xyzrw' : xyzrw,
                'Rg' : Rg, 
                'hue_bin_data' : hue_bin_data,
                'cri_type' : cri_type}

    if (out == 'Rg'):
        return Rg
    elif (out == 'Rg,data'):
        return Rg, data
    elif (out == 'data'):
        return data
    elif (out == 'Rg,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr'):
        return Rg,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr
    else:
        return eval(out)
    
#------------------------------------------------------------------------------
def spd_to_cri(St, cri_type = _CRI_TYPE_DEFAULT, out = 'Rf', wl = None, \
               sampleset = None, ref_type = None, cieobs = None, avg = None, \
               scale = None, opt_scale_factor = False, cspace = None, catf = None,\
               cri_specific_pars = None, rg_pars = None, fit_gamut_ellipse = False):
    """
    Calculates the color rendering fidelity index, Rf, of spectral data. 
    
    Args:
        :St: 
            | ndarray with spectral data 
            | (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'Rf' or str, optional
            | Specifies requested output (e.g. 'Rf,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
            | override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
        :rg_pars: 
            | None or dict, optional
            | Dict containing specifying parameters for slicing the gamut 
            | and calculating hue bin specific indices.
            | Dict structure: 
            |     {'nhbins' : None, 'start_hue' : 0, 
            |       'normalize_gamut' : False, 'normalized_chroma_ref': 100.0}
            |    - key: 'nhbins': int, number of hue bins to slice gamut 
            |                 (None use the one specified in :cri_type: dict).
            |    - key: 'start_hue': float (°), hue at which to start slicing
            |    - key: 'normalize_gamut': True or False: 
            |                normalize gamut or not before calculating a gamut 
            |                area index Rg. 
            |    - key: 'normalized_chroma_ref': 100.0 or float, optional
            |                Controls the size (chroma/radius) 
            |                of the normalization circle/gamut.
            |    - key 'use_bin_avg_DEi': True or False
            |               Note that following IES-TM30 DEhj from gamut_slicer()
            |               is obtained by averaging the DEi per hue bin (True),
            |               and NOT by averaging the jabt and jabr per hue bin 
            |               and then calculating the DEhj (False).
        :avg: 
            | None or fcn handle, optional
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.
        :scale:
            | None or dict, optional
            | Specifies scaling of color differences to obtain CRI.
            |     - None use the one specified in :cri_type: dict.
            |     - dict: user specified dict with scaling parameters.
            |         - key: 'fcn': function handle to type of cri scale, 
            |                 e.g. 
            |                 * linear()_scale --> (100 - scale_factor*DEi), 
            |                 * log_scale --> (cfr. Ohno's CQS), 
            |                 * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
            |        - key: 'cfactor': factors used in scaling function, 
            |              If None: 
            |                     Scaling factor value(s) will be optimized to 
            |                     minimize the rms between the Rf's of the 
            |                     requested metric and the target metric specified
            |                     in:
            |
            |                  - key: 'opt_cri_type':  str 
            |                      * str: one of the preset _CRI_DEFAULTS
            |                      * dict: user speciied 
            |                      (dict must contain all keys as normal)
            |                     Note that if key not in :scale: dict, 
            |                     then 'opt_cri_type' is added with default 
            |                     setting = 'ciera'.
            |                 - key: 'opt_spd_set': ndarray with set of light 
            |                     source spds used to optimize cfactor. 
            |                     Note that if key not in :scale: dict, 
            |                     then default = 'F1-F12'.
        :opt_scale_factor: 
            | True or False, optional
            | True: optimize scaling-factor, else do nothing and use value of 
            | scaling-factor in :scale: dict.   
        :fit_gamut_ellipse:
            | fit ellipse to normalized color gamut 
            | (extract from function using out; also stored in hue_bin_data['gamut_ellipse_fit'])
    
    Returns:
        :returns: 
            | float or ndarray with Rf for :out: 'Rf'
            | Other output is also possible by changing the :out: str value.
            | E.g. out == 'Rg,data' would output an ndarray with Rf values 
            | 
            | and a dictionary :data: with keys:
            | 
            | - 'St, Sr'  : ndarray of test SPDs and corresponding ref. illuminants.
            | - 'xyz_cct': xyz of white point calculate with cieobs defined for cct calculations in cri_type['cieobs']
            | - 'cct, duv': CCT and Duv obtained with cieobs in cri_type['cieobs']['cct']
            | - 'xyzti, xyzri': ndarray tristimulus values of test and ref. samples (obtained with with cieobs in cri_type['cieobs']['xyz'])
            | - 'xyztw, xyzrw': ndarray tristimulus values of test and ref. white points (obtained with with cieobs in cri_type['cieobs']['xyz'])
            | - 'DEi, DEa': ndarray with individual sample color differences DEi and average DEa between test and ref.       
            | - 'Rf'  : ndarray with general color fidelity index values
            | - 'Rg'  : ndarray with color gamut area index values
            | - 'Rfi'  : ndarray with specific (sample) color fidelity indices
            | - 'Rfhj' : ndarray with local (hue binned) fidelity indices
            | - 'DEhj' : ndarray with local (hue binned) color differences
            | - 'Rcshj': ndarray with local chroma shifts indices
            | - 'Rhshj': ndarray with local hue shifts indices
            | - 'hue_bin_data': dict with output from _get_hue_bin_data() [see its help for more info]
            | - 'cri_type': same as input (for reference purposes)
            
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `CIE224:2017. CIE 2017 Colour Fidelity Index for accurate scientific use. 
        Vienna, Austria: CIE. (2017).
        <http://www.cie.co.at/index.php?i_ca_id=1027>`_
        
        4. `Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
        CRI2012: A proposal for updating the CIE colour rendering index. 
        Lighting Research and Technology, 45, 689–709. 
        <http://lrt.sagepub.com/content/45/6/689>`_
        
        5. `CIE13.3-1995. Method of Measuring and Specifying 
        Colour Rendering Properties of Light Sources 
        (Vol. CIE13.3-19). Vienna, Austria: CIE. (1995).
        <http://www.cie.co.at/index.php/index.php?i_ca_id=303>`_
                    

    """
    outlist = out.split(',')
    
    #Override input parameters with data specified in cri_type:
    args = copy.deepcopy(locals()) # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)

    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_cri')
    
    # unpack some keys:
    if (opt_scale_factor is None) | (opt_scale_factor is False):
        scale_factor = cri_type['scale']['cfactor']
    scale_fcn = cri_type['scale']['fcn']
    avg = cri_type['avg']  
    
    # Input parsing: optimize scale_factor for input based on F1-F12 (default) if scale_factor is NaN or None:
    scale_factor = optimize_scale_factor(cri_type,opt_scale_factor, scale_fcn, avg)

    if np.isnan(scale_factor).any():
        raise Exception ('Unable to optimize scale_factor.')

    # A. get DEi of for ciera and of requested cri metric for spds in or specified by scale_factor_optimization_spds':
    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    (jabt,jabr,
     xyzti,xyztw,
     xyzri,xyzrw,
     xyztw_cct,
     cct,duv,St,Sr) = spd_to_jab_t_r(St, wl = wl, cri_type = cri_type, 
                                          out = 'jabt,jabr,xyzti,xyztw,xyzri,xyzrw,xyztw_cct,cct,duv,St,Sr') 

    # E. calculate DEi, DEa:
    DEi = ((jabt - jabr)**2).sum(axis = -1)**0.5
    DEa = np2d(avg(DEi,axis=0))
    
    # B. convert DEi to color rendering index:
    Rfi = scale_fcn(DEi, scale_factor)
    Rf = np2d(scale_fcn(DEa, scale_factor))
    
    # C. get binned jabt jabr and DEi:
    if ('Rg' in outlist) | ('Rfhj' in outlist) | ('DEhj' in outlist) | \
       ('Rhshj' in outlist) | ('Rcshj' in outlist) | ('hue_bin_data' in outlist) |\
       ('data' in outlist) | (fit_gamut_ellipse == True):
        
        rg_pars = cri_type['rg_pars'] 
        if 'use_bin_avg_DEi' not in rg_pars: rg_pars['use_bin_avg_DEi'] = True
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue, use_bin_avg_DEi  = [rg_pars[x] for x in sorted(rg_pars.keys())]

        
        # get hue_bin_data:
        hue_bin_data = _get_hue_bin_data(jabt, jabr, 
                                    start_hue = start_hue, nhbins = nhbins,
                                    normalized_chroma_ref = normalized_chroma_ref)
        # Calculate color gamut area index, Rg:
        Rg = _hue_bin_data_to_rg(hue_bin_data, normalize_gamut = normalize_gamut)
        
        # Fit an ellipse to the normalized color gamut:
        if fit_gamut_ellipse:
            gamut_ellipse_fit = _hue_bin_data_to_ellipsefit(hue_bin_data)
            hue_bin_data['gamut_ellipse_fit'] = gamut_ellipse_fit
        else:
            gamut_ellipse_fit = {}
        
    else:
        Rg, hue_bin_data = None, None


    # D. # Calculate local fidelity, chroma shifts and hue shifts:
    if hue_bin_data is not None:
        Rcshj, Rhshj, Rfhj, DEhj = _hue_bin_data_to_rxhj(hue_bin_data, 
                                                         scale_fcn = scale_fcn,
                                                         scale_factor = scale_factor,
                                                         cri_type = cri_type,
                                                         use_bin_avg_DEi = use_bin_avg_DEi)


    if 'data' in out:
        data = {'St' : St, 'Sr' : Sr, 'xyztw_cct' : xyztw_cct,
                'cct' : cct, 'duv' : duv,
                'xyzti' : xyzti, 'xyztw' : xyztw, 
                'xyzri' : xyzri, 'xyzrw' : xyzrw,
                'DEi' : DEi, 'DEa' : DEa,
                'Rf' : Rf, 'Rg' : Rg, 'Rfi' : Rfi,
                'Rcshj' : Rcshj, 'Rhshj' : Rhshj, 'Rfhj' : Rfhj,
                'hue_bin_data' : hue_bin_data,
                'cri_type' : cri_type}

    if (out == 'Rf'):
        return Rf
    elif (out == 'Rf,Rg'):
        return Rf, Rg
    elif (out == 'Rf,data'):
        return Rf, data
    elif (out == 'data'):
        return data
    elif (out == 'Rf,Rg,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,cct,duv,St,Sr'):
        return Rf,Rg,jabt,jabr,xyzti,xyztw,xyzri,xyzrw,cct,duv,St,Sr
    else:
        return eval(out)

# For testing:
# from luxpy import _CIE_F4, _CIE_D65, _IESTM3018

# F4 = cie_interp(_CIE_F4, wl_new=[360,830,1], kind = 'spd')
# D65 = cie_interp(_CIE_D65, wl_new=[360,830,1], kind = 'spd')
# out = spd_to_cri(np.vstack((F4,D65[1:])), out = 'Rf,Rg,data') #, 
#                  # opt_scale_factor = True, 
#                  # scale = {'fcn':log_scale, 
#                  #          'cfactor':[None],
#                  #          'opt_spd_set' : _IESTM3018['S']['data'][:100,:].copy() })
# print(out[:2])