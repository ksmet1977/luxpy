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

Extension module for IES TM30 metric calculation with additional Vector Field support
=====================================================================================

 :spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import copy
from luxpy import _CRI_RFL
from luxpy.utils import np


from luxpy.color.cri.utils.helpers2 import (_get_hue_bin_data, 
                                            _hue_bin_data_to_rxhj, 
                                            _hue_bin_data_to_rfi,
                                            spd_to_cri)
from luxpy.color.cri.utils.init_cri_defaults_database import _CRI_DEFAULTS

from luxpy.color.cri.VFPX.vectorshiftmodel import  _VF_MODEL_TYPE, _VF_PCOLORSHIFT, VF_colorshift_model
from luxpy.color.cri.VFPX.VF_PX_models import plot_VF_PX_models

__all__ = ['spd_to_ies_tm30_metrics']

def spd_to_ies_tm30_metrics(St, cri_type = None, \
                            hbins = 16, start_hue = 0.0,\
                            scalef = 100, \
                            vf_model_type = _VF_MODEL_TYPE, \
                            vf_pcolorshift = _VF_PCOLORSHIFT,\
                            scale_vf_chroma_to_sample_chroma = False):
    """
    Calculates IES TM30 metrics from spectral data.      
      
      Args:
        :St:
            | numpy.ndarray with spectral data 
        :cri_type:
            | None, optional
            | If None: defaults to cri_type = 'iesrf'.
            | Not none values of :hbins:, :start_hue: and :scalef: overwrite 
            | input in cri_type['rg_pars'] 
        :hbins:
            | None or numpy.ndarray with sorted hue bin centers (°), optional
        :start_hue: 
            | None, optional
        :scalef:
            | None, optional
            | Scale factor for reference circle.
        :vf_pcolorshift:
            | _VF_PCOLORSHIFT or user defined dict, optional
            | The polynomial models of degree 5 and 6 can be fully specified or 
            | summarized by the model parameters themselved OR by calculating the
            | dCoverC and dH at resp. 5 and 6 hues. :VF_pcolorshift: specifies 
            | these hues and chroma level.
        :scale_vf_chroma_to_sample_chroma: 
            | False, optional
            | Scale chroma of reference and test vf fields such that average of 
            | binned reference chroma equals that of the binned sample chroma
            | before calculating hue bin metrics.
            
    Returns:
        :data: 
            | Dictionary with color rendering data:
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
            | - 'vf' : dictionary with vector field measures and data.
            |         Keys:
            |           - 'Rt'  : ndarray with general metameric uncertainty index Rt
            |           - 'Rti' : ndarray with specific metameric uncertainty indices Rti
            |           - 'Rfhj' : ndarray with local (hue binned) fidelity indices 
            |                            obtained from VF model predictions at color space
            |                            pixel coordinates
            |           - 'DEhj' : ndarray with local (hue binned) color differences
            |                           (same as above)
            |           - 'Rcshj': ndarray with local chroma shifts indices for vectorfield coordinates
            |                           (same as above)
            |           - 'Rhshj': ndarray with local hue shifts indicesfor vectorfield coordinates
            |                           (same as above)
            |           - 'Rfi': ndarray with sample fidelity indices for vectorfield coordinates
            |                           (same as above)
            |           - 'DEi': ndarray with sample color differences for vectorfield coordinates
            |                           (same as above)
            |           - 'hue_bin_data': dict with output from _get_hue_bin_data() for vectorfield coordinates
            |           - 'dataVF': dictionary with output of cri.VFPX.VF_colorshift_model()
    """
    if cri_type is None:
        cri_type = 'iesrf'

    if isinstance(cri_type,str): # get dict 
        cri_type = copy.deepcopy(_CRI_DEFAULTS[cri_type])
    if hbins is not None:
        cri_type['rg_pars']['nhbins'] = hbins 
    if start_hue is not None:
        cri_type['rg_pars']['start_hue'] = start_hue
    if scalef is not None:
        cri_type['rg_pars']['normalized_chroma_ref'] = scalef
    
    #Calculate color rendering measures for SPDs in St:      
    data,_ = spd_to_cri(St, cri_type = cri_type, out = 'data,hue_bin_data', 
                        fit_gamut_ellipse = True)
    hdata = data['hue_bin_data']
    Rfhj, Rcshj, Rhshj = data['Rfhj'], data['Rcshj'], data['Rhshj']
    cct = data['cct']
    
    #Calculate Metameric uncertainty and base color shifts:
    dataVF = VF_colorshift_model(St, cri_type = cri_type, 
                                 model_type = vf_model_type, 
                                 cspace = cri_type['cspace'], 
                                 sampleset = eval(cri_type['sampleset']), 
                                 pool = False, 
                                 pcolorshift = vf_pcolorshift, 
                                 vfcolor = 0)
    Rf_ = np.array([dataVF[i]['metrics']['Rf'] for i in range(len(dataVF))]).T
    Rt = np.array([dataVF[i]['metrics']['Rt'] for i in range(len(dataVF))]).T
    Rti = np.array([dataVF[i]['metrics']['Rti'] for i in range(len(dataVF))][0])
    _data_vf = {'Rt' : Rt, 'Rti' : Rti, 'Rf_' : Rf_} # add to dict for output

    # Get normalized and sliced hue-bin _hj data for plotting:
    rg_pars = cri_type['rg_pars']
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
    
    # Get chroma of samples:    
    if scale_vf_chroma_to_sample_chroma == True:
        jabt_hj_closed, jabr_hj_closed = hdata['jabt_hj_closed'], hdata['jabr_hj_closed']
        Cr_hj_s = (np.sqrt(jabr_hj_closed[:-1,...,1]**2 + jabr_hj_closed[:-1,...,2]**2)).mean(axis=0) # for rescaling vector field average reference chroma

    #jabtn_hj_closed, jabrn_hj_closed = hdata['jabtn_hj_closed'], hdata['jabrn_hj_closed']
    
    # get vector field data for each source (must be on 2nd dim)
    jabt_vf = np.transpose(np.array([np.hstack((np.ones(dataVF[i]['fielddata']['vectorfield']['axt'].shape),dataVF[i]['fielddata']['vectorfield']['axt'],dataVF[i]['fielddata']['vectorfield']['bxt'])) for i in range(cct.shape[0])]),(1,0,2))
    jabr_vf = np.transpose(np.array([np.hstack((np.ones(dataVF[i]['fielddata']['vectorfield']['axr'].shape),dataVF[i]['fielddata']['vectorfield']['axr'],dataVF[i]['fielddata']['vectorfield']['bxr'])) for i in range(cct.shape[0])]),(1,0,2))
    
    # Get hue bin data for vector field data:
    hue_bin_data_vf = _get_hue_bin_data(jabt_vf, jabr_vf, 
                                        start_hue = start_hue, nhbins = nhbins,
                                        normalized_chroma_ref = normalized_chroma_ref )
    
    # Rescale chroma of vector field such that it is on average equal to that of the binned samples:
    if scale_vf_chroma_to_sample_chroma == True:
        Cr_vf_hj, Cr_vf, Ct_vf = hue_bin_data_vf['Cr_hj'], hue_bin_data_vf['Cr'], hue_bin_data_vf['Ct']
        hr_vf, ht_vf = hue_bin_data_vf['hr'], hue_bin_data_vf['ht']
        fC = np.nanmean(Cr_hj_s)/np.nanmean(Cr_vf_hj)
        jabr_vf[...,1], jabr_vf[...,2] = fC * Cr_vf*np.cos(hr_vf), fC * Cr_vf*np.sin(hr_vf)
        jabt_vf[...,1], jabt_vf[...,2] = fC * Ct_vf*np.cos(ht_vf), fC * Ct_vf*np.sin(ht_vf)
        
        # Get new hue bin data for rescaled vector field data:
        hue_bin_data_vf = _get_hue_bin_data(jabt_vf, jabr_vf, 
                                            start_hue = start_hue, nhbins = nhbins,
                                            normalized_chroma_ref = normalized_chroma_ref )
    
    # Get scale factor and scaling function for Rfx:
    scale_factor = cri_type['scale']['cfactor']
    scale_fcn = cri_type['scale']['fcn']

    # Calculate Local color fidelity, chroma and hue shifts for vector field data:
    (Rcshj_vf, Rhshj_vf,
     Rfhj_vf, DEhj_vf) = _hue_bin_data_to_rxhj(hue_bin_data_vf, 
                                               cri_type = cri_type,
                                               scale_factor = scale_factor,
                                               scale_fcn = scale_fcn) 
                                               
    # Get sample color fidelity for vector field data:
    (Rfi_vf, DEi_vf) = _hue_bin_data_to_rfi(hue_bin_data_vf, 
                                            cri_type = cri_type,
                                            scale_factor = scale_factor,
                                            scale_fcn = scale_fcn)
    # Store in dict:
    _data_vf.update({'Rfi' : Rfi_vf, 'DEi' : DEi_vf,
                     'Rcshj' : Rcshj_vf, 'Rhshj' : Rhshj_vf,
                     'Rfhj' : Rfhj_vf, 'DEhj': DEhj_vf,
                     'dataVF' : dataVF, 'hue_bin_data' : hue_bin_data_vf})
    
    # Add to main dictionary:
    data['vf'] = _data_vf
    return data

if __name__ == '__main__':
    # for testing:
    import luxpy as lx
    F4 = lx.cie_interp(lx._CIE_F4,wl_new=[360,830,1],kind='spd')
    D65 = lx.cie_interp(lx._CIE_D65,wl_new=[360,830,1],kind='spd')
    spds = lx._IESTM3018['S']['data'].copy()
    spds = lx.cie_interp(spds,wl_new = [360,830,1],kind='spd')
    
    spd = np.vstack((F4,D65[1:]))
    d = spd_to_ies_tm30_metrics(spd, cri_type = None, \
                                hbins = 16, start_hue = 0.0,\
                                scalef = 100, \
                                vf_model_type = _VF_MODEL_TYPE, \
                                vf_pcolorshift = _VF_PCOLORSHIFT,\
                                scale_vf_chroma_to_sample_chroma = False)
    