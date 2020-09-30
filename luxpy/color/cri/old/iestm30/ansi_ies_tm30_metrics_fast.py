# -*- coding: utf-8 -*-
"""
Module for fast ANSI/IES-TM30 calculations
==========================================

Created on Mon Sep 28 16:34:14 2020

@author: ksmet1977@gmail.com
"""


import copy
import luxpy as lx
import numpy as np
import matplotlib.pyplot as plt

from luxpy import (spd_to_xyz, xyz_to_cct, getwld, getwlr, _CMF, blackbody, daylightphase, 
                   _CRI_RFL, _CRI_REF_TYPES, _CRI_REF_TYPE,_CIEOBS, xyzbar, cie_interp)
from luxpy.color.cam import (xyz_to_jab_cam02ucs, hue_angle)
from luxpy.color.cri.utils.DE_scalers import log_scale
from luxpy.color.cri.utils.helpers2 import _get_hue_bin_data 
from luxpy import math

# __all__ = ['spd_to_xyz','spd_to_ler','xyz_to_cct'] # direct imports from luxpy
__all__ = ['_cri_ref','_xyz_to_jab_cam02ucs','spd_to_tm30'] # new or redefined


_DL = 1
_WL3 = [360,830,_DL]
_WL = getwlr(_WL3)
_POS_WL560 = np.where(np.abs(_WL - 560.0) == np.min(np.abs(_WL - 560.0)))[0]
_TM30_SAMPLE_SET = _CRI_RFL['ies-tm30-18']['99']['{:1.0f}nm'.format(_DL)]


def _cri_ref_i(cct, wl3 = _WL, ref_type = 'iestm30', mix_range = [4000,5000], 
            cieobs = '1931_2', force_daylight_below4000K = False, n = None,
            daylight_locus = None):
    """
    Calculates a reference illuminant spectrum based on cct 
    for color rendering index calculations.
    """   
    if mix_range is None:
        mix_range =  _CRI_REF_TYPES[ref_type]
    if (cct < mix_range[0]) | (ref_type == 'BB'):
        return blackbody(cct, wl3, n = n)
    elif (cct > mix_range[0]) | (ref_type == 'DL'):
        return daylightphase(cct,wl3,force_daylight_below4000K = force_daylight_below4000K, cieobs = cieobs, daylight_locus = daylight_locus)
    else:
        SrBB = blackbody(cct, wl3, n = n)
        SrDL = daylightphase(cct, wl3, verbosity = None,force_daylight_below4000K = force_daylight_below4000K, cieobs = cieobs, daylight_locus = daylight_locus)
        cmf = _CMF[cieobs]['bar'] if isinstance(cieobs,str) else cieobs 
        wl = SrBB[0]
        ld = getwld(wl)

        SrBB = 100.0*SrBB[1]/np.array(np.sum(SrBB[1]*cmf[2]*ld))
        SrDL = 100.0*SrDL[1]/np.array(np.sum(SrDL[1]*cmf[2]*ld))
        Tb, Te = float(mix_range[0]), float(mix_range[1])
        cBB, cDL = (Te-cct)/(Te-Tb), (cct-Tb)/(Te-Tb)
        if cBB < 0.0:
            cBB = 0.0
        elif cBB > 1:
            cBB = 1.0
        if cDL < 0.0:
            cDL = 0.0
        elif cDL > 1:
            cDL = 1.0

        Sr = SrBB*cBB + SrDL*cDL
        Sr[Sr == float('NaN')] = 0.0
        Sr = np.vstack((wl,(Sr/Sr[_POS_WL560])))
               
        return  Sr
    
def _cri_ref(ccts, wl3 = _WL, ref_type = 'iestm30', mix_range = [4000,5000], 
             cieobs = '1931_2', force_daylight_below4000K = False, n = None,
             daylight_locus = None, wl = [360,830,1]):
    """
    Calculates multiple reference illuminant spectra based on ccts 
    for color rendering index calculations.
    """  
    if mix_range is None:
        mix_range =  _CRI_REF_TYPES[ref_type]
    if isinstance(ccts,float): ccts = [ccts]
    wlr = getwlr(wl3)
    Srs = np.zeros((len(ccts)+1,len(wlr)))
    Srs[0] = wlr
    for i,cct in enumerate(ccts):
        Srs[i+1,:] = _cri_ref_i(cct, wl3 = wl3, ref_type = ref_type, 
                      mix_range = mix_range, cieobs = cieobs, 
                      force_daylight_below4000K = force_daylight_below4000K, n = n,
                      daylight_locus = daylight_locus)[1:]
    
    return Srs  


def _xyz_to_jab_cam02ucs(xyz, xyzw, conditions = None):
    
    #--------------------------------------------
    # Get/ set conditions parameters:
    if conditions is not None:
        surround_parameters =  {'surrounds': ['avg', 'dim', 'dark'], 
                                'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 
                                'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,
                                'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}
        La = conditions['La']
        Yb = conditions['Yb']
        D = conditions['D']
        surround = conditions['surround']
        if isinstance(surround, str):
            surround = surround_parameters[conditions['surround']]
        F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
    else:
        # set defaults:
        La, Yb, D, F, FLL, Nc, c = 100, 20, 1, 1, 1, 1, 0.69
        
    #--------------------------------------------
    # Define sensor space and cat matrices:        
    mhpe = np.array([[0.38971,0.68898,-0.07868],
                     [-0.22981,1.1834,0.04641],
                     [0.0,0.0,1.0]]) # Hunt-Pointer-Estevez sensors (cone fundamentals)
    
    mcat = np.array([[0.7328, 0.4296, -0.1624],
                       [ -0.7036, 1.6975,  0.0061],
                       [ 0.0030, 0.0136,  0.9834]]) # CAT02 sensor space
    
    #--------------------------------------------
    # pre-calculate some matrices:
    invmcat = np.linalg.inv(mcat)
    mhpe_x_invmcat = np.dot(mhpe,invmcat)
    
    #--------------------------------------------
    # calculate condition dependent parameters:
    Yw = xyzw[...,1:2].T
    k = 1.0 / (5.0*La + 1.0)
    FL = 0.2*(k**4.0)*(5.0*La) + 0.1*((1.0 - k**4.0)**2.0)*((5.0*La)**(1.0/3.0)) # luminance adaptation factor
    n = Yb/Yw 
    Nbb = 0.725*(1/n)**0.2   
    Ncb = Nbb
    z = 1.48 + FLL*n**0.5
    
    if D is None:
        D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))
        
    #--------------------------------------------
    # transform from xyz, xyzw to cat sensor space:
    rgb = math.dot23(mcat, xyz.T)
    rgbw = mcat @ xyzw.T
    
    #--------------------------------------------  
    # apply von Kries cat:
    rgbc = ((D*Yw/rgbw)[...,None] + (1 - D))*rgb # factor 100 from ciecam02 is replaced with Yw[i] in ciecam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)
    rgbwc = ((D*Yw/rgbw) + (1 - D))*rgbw # factor 100 from ciecam02 is replaced with Yw[i] in ciecam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)
 
    #--------------------------------------------
    # convert from cat02 sensor space to cone sensors (hpe):
    rgbp = math.dot23(mhpe_x_invmcat,rgbc).T
    rgbwp = (mhpe_x_invmcat @ rgbwc).T

    #--------------------------------------------
    # apply Naka_rushton repsonse compression:
    naka_rushton = lambda x: 400*x**0.42/(x**0.42 + 27.13) + 0.1
    
    rgbpa = naka_rushton(FL*rgbp/100.0)
    p = np.where(rgbp<0)
    rgbpa[p] = 0.1 - (naka_rushton(FL*np.abs(rgbp[p])/100.0) - 0.1)
    
    rgbwpa = naka_rushton(FL*rgbwp/100.0)
    pw = np.where(rgbwp<0)
    rgbwpa[pw] = 0.1 - (naka_rushton(FL*np.abs(rgbwp[pw])/100.0) - 0.1)

    #--------------------------------------------
    # Calculate achromatic signal:
    A  =  (2.0*rgbpa[...,0] + rgbpa[...,1] + (1.0/20.0)*rgbpa[...,2] - 0.305)*Nbb
    Aw =  (2.0*rgbwpa[...,0] + rgbwpa[...,1] + (1.0/20.0)*rgbwpa[...,2] - 0.305)*Nbb
    
    #--------------------------------------------
    # calculate initial opponent channels:
    a = rgbpa[...,0] - 12.0*rgbpa[...,1]/11.0 + rgbpa[...,2]/11.0
    b = (1.0/9.0)*(rgbpa[...,0] + rgbpa[...,1] - 2.0*rgbpa[...,2])
        
    #--------------------------------------------
    # calculate hue h and eccentricity factor, et:
    h = np.arctan2(b,a)
    et = (1.0/4.0)*(np.cos(h + 2.0) + 3.8)
    
    #--------------------------------------------
    # calculate lightness, J:
    J = 100.0* (A / Aw)**(c*z)
    
    #--------------------------------------------
    # calculate chroma, C:
    t = ((50000.0/13.0)*Nc*Ncb*et*((a**2.0 + b**2.0)**0.5)) / (rgbpa[...,0] + rgbpa[...,1] + (21.0/20.0*rgbpa[...,2]))
    C = (t**0.9)*((J/100.0)**0.5) * (1.64 - 0.29**n)**0.73
    
    #--------------------------------------------  
    # Calculate colorfulness, M:
    M = C*FL**0.25
        
    #--------------------------------------------
    # convert to cam02ucs J', aM', bM':
    KL, c1, c2 =  1.0, 0.007, 0.0228
    Jp = (1.0 + 100.0*c1)*J / (1.0 + c1*J)
    Mp = (1.0/c2) * np.log(1.0 + c2*M)
    aMp = Mp * np.cos(h)
    bMp = Mp * np.sin(h)
    
    return np.dstack((Jp,aMp,bMp))

    
def _polyarea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1,axis=0))-np.dot(y,np.roll(x,1,axis=0)))

def _hue_bin_data_to_rg(hue_bin_data):
    jabt_closed = np.vstack((hue_bin_data['jabt_hj'],hue_bin_data['jabt_hj'][:1,...]))
    jabr_closed = np.vstack((hue_bin_data['jabr_hj'],hue_bin_data['jabr_hj'][:1,...]))
    notnan_t = np.logical_not(np.isnan(jabt_closed[...,1])) # avoid NaN's (i.e. empty hue-bins)
    notnan_r = np.logical_not(np.isnan(jabr_closed[...,1]))
    Rg = np.array([[100*_polyarea(jabt_closed[notnan_t[:,i],i,1],jabt_closed[notnan_t[:,i],i,2]) / _polyarea(jabr_closed[notnan_r[:,i],i,1],jabr_closed[notnan_r[:,i],i,2]) for i in range(notnan_r.shape[-1])]])
    return Rg

def _hue_bin_data_to_Rxhj(hue_bin_data, scale_factor):
    
    nhbins = hue_bin_data['nhbins']
    start_hue = hue_bin_data['start_hue']
    
    # A. Local color fidelity, Rfhj:
    #DEhj = ((hue_bin_data['jabt_hj']-hue_bin_data['jabr_hj'])**2).sum(axis=-1)**0.5
    DEhj = hue_bin_data['DE_hj'] #TM30 specifies average of DEi per hue bin, not DE of average jabt, jabr
    Rfhj = log_scale(DEhj, scale_factor = scale_factor)
    
    # B.Local chroma shift and hue shift, [Rcshi, Rhshi]:
    # B.1 relative paths:
    dab = (hue_bin_data['jabt_hj']- hue_bin_data['jabr_hj'])[...,1:]/(hue_bin_data['Cr_hj'][...,None])

    # B.2 Reference unit circle:
    hbincenters = np.arange(start_hue + np.pi/nhbins, 2*np.pi, 2*np.pi/nhbins)[...,None]
    arc = np.cos(hbincenters)
    brc = np.sin(hbincenters)

    # B.3 calculate local chroma shift, Rcshi:
    Rcshi = dab[...,0] * arc + dab[...,1] * brc
    
    # B.4 calculate local hue shift, Rcshi:
    Rhshi = dab[...,1] * arc - dab[...,0] * brc
    
    return Rcshi, Rhshi, Rfhj, DEhj 

def _hue_bin_data_to_ellipsefit(hue_bin_data):
    
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
 

def spd_to_tm30(St):
    
    # calculate CIE 1931 2° white point xyz:
    xyzw_cct, _ = spd_to_xyz(St, cieobs = '1931_2', relative = True, out = 2)
    
    # calculate cct, duv:
    cct, duv = xyz_to_cct(xyzw_cct, cieobs = '1931_2', out = 'cct,duv')
    
    # calculate ref illuminant:
    Sr = _cri_ref(cct, mix_range = [4000, 5000], cieobs = '1931_2', wl3 = St[0])

    # calculate CIE 1964 10° sample and white point xyz under test and ref. illuminants:
    xyz, xyzw = spd_to_xyz(np.vstack((St,Sr[1:])), cieobs = '1964_10', 
                           rfl = _TM30_SAMPLE_SET, relative = True, out = 2)
    N = St.shape[0]-1
    
    xyzt, xyzr =  xyz[:,:N,:], xyz[:,N:,:]
    xyzwt, xyzwr =  xyzw[:N,:], xyzw[N:,:]

    
    # calculate CAM02-UCS coordinates 
    # (standard conditions = {'La':100.0,'Yb':20.0,'surround':'avg','D':1.0):
    jabt = _xyz_to_jab_cam02ucs(xyzt, xyzw = xyzwt)
    jabr = _xyz_to_jab_cam02ucs(xyzr, xyzw = xyzwr)
   
    
    # calculate DEi, Rfi:
    DEi = (((jabt-jabr)**2).sum(axis=-1,keepdims=True)**0.5)[...,0]
    Rfi = log_scale(DEi, scale_factor = [6.73])
    
    # calculate Rf
    DEa = DEi.mean(axis = 0,keepdims = True)
    Rf = log_scale(DEa, scale_factor = [6.73])
        
    # calculate hue-bin data:
    hue_bin_data = _get_hue_bin_data(jabt, jabr, start_hue = 0, nhbins = 16)       

    # calculate Rg:
    Rg = _hue_bin_data_to_rg(hue_bin_data)                 
        
    # calculate local color fidelity values, Rfhj,
    # local hue shift, Rhshj and local chroma shifts, Rcshj:
    Rcshj, Rhshj, Rfhj, DEhj = _hue_bin_data_to_Rxhj(hue_bin_data, 
                                                    scale_factor = [6.73])
    
    # Fit ellipse to gamut shape of samples under test source:
    gamut_ellipse_fit = _hue_bin_data_to_ellipsefit(hue_bin_data)
    hue_bin_data['gamut_ellipse_fit'] = gamut_ellipse_fit
    
    # return output dict:
    return {'St' : St, 'Sr' : Sr, 
            'xyzw_cct' : xyzw_cct, 'xyzwt' : xyzwt, 'xyzwr' : xyzwr,
            'xyzt' : xyzt, 'xyzr' : xyzr, 
            'cct': cct.T, 'duv': duv.T,
            'jabt' : jabt, 'jabr' : jabr, 
            'DEi' : DEi, 'DEa' : DEa, 'Rfi' : Rfi, 'Rf' : Rf,
            'hue_bin_data' : hue_bin_data, 'Rg' : Rg,
            'DEhj' : DEhj, 'Rfhj' : Rfhj,
            'Rcshj': Rcshj,'Rhshj':Rhshj,
            'hue_bin_data' : hue_bin_data}


if __name__ == '__main__':
    #------------------------------------------------------------------------------
    # For comparison
    _CRI_TYPE_TM30 = copy.deepcopy(lx.cri._CRI_DEFAULTS['iesrf-tm30-18'])
    _CRI_TYPE_TM30['sampleset'] = "_CRI_RFL['ies-tm30-18']['99']['1nm']"
    def _spd_to_tm30(spd, cieobs = '1931_2', mixer_type = '3mixer'):
        # Call function that calculates ref.illuminant and jabt & jabr only once to obtain Rf & Rg:
    
        data = lx.cri._tm30_process_spd(spd, cri_type = _CRI_TYPE_TM30) # use 1nm samples to avoid interpolation
        Rf, Rg = data['Rf'], data['Rg']
        
        thetad = data['hue_bin_data']['gamut_ellipse_fit']['thetad']
        ecc = data['hue_bin_data']['gamut_ellipse_fit']['a/b']
        
        xyzw = lx.spd_to_xyz(spd, cieobs = cieobs, relative = False) # set K = 1 to avoid overflow when _FLOAT_TYPE = np.float16
        data['xyzw'] = xyzw
        return data
    
    spds = lx._IESTM3018['S']['data'].copy()
    spds = lx.cie_interp(spds,wl_new = _WL,kind='spd')
    spds = spds[:202,:]
    data = spd_to_tm30(spds[[0,104],:])
    # data = spd_to_tm30(lx._CIE_F4)
    jabr = data['hue_bin_data']['jabrn_hj_closed']
    jabt = data['hue_bin_data']['jabtn_hj_closed']
    plt.plot(jabt[...,1],jabt[...,2],'b+-')
    plt.plot(jabr[...,1],jabr[...,2],'rx-')
    
        



