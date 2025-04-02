# IES TM30 test

#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')


#-----other imports-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# --- xyz_to_cct_ohno2014 --------------------------------------------------
from luxpy import _BB
from luxpy import getwlr, _RFL, spd_to_xyz, xyz_to_Yuv60, _S012_DAYLIGHTPHASE, cat, math, cri
from luxpy.utils import getdata

pi = np.pi#3.141592653589#np.pi

# -- Helper functions ------------------------------------------------------

round = lambda x,n: lx.math.round(x,n=(n,'sigfig')) # round to n significant figures

def polyarea(x,y):
    return 0.5 * np.abs(np.sum(x * np.roll(y, shift=1, axis=1) - y * np.roll(x, shift=1, axis=1), axis=1))


# --- illuminants ---------------------------------------------------------
def _blackbody(cct, wl, n = 1, relative = True, c2 = None, add_wl = True):
    _EPS = 1e-16
    if c2 is None: c2 = _BB['c2']
    wl = getwlr(wl)
    if relative:
      Sb = ((wl*1.0e-9)**(-5))*(np.exp(c2*((n*wl*1.0e-9*(cct[...,None]+_EPS))**(-1.0)))-1.0)**(-1.0)
    else:
       Sb = (1/pi)*_BB['c1']*((wl*1.0e-9)**(-5))*(n**(-2.0))*(np.exp(c2*((n*wl*1.0e-9*(cct[...,None]+_EPS))**(-1.0)))-1.0)**(-1.0)
    
    if add_wl: 
        Sb = np.vstack((wl,Sb))
    return Sb 

def _daylightlocus(ccts):
    xD = (-4.6070e9)/ccts**3 + (2.9678e6)/ccts**2 + (0.09911e3)/ccts + 0.244063
    cnd = ccts > 7000
    xD[cnd] = (-2.0064e9)/ccts[cnd]**3 + (1.9018e6)/ccts[cnd]**2 + (0.24748e3)/ccts[cnd] + 0.23704
    yD = -3.000*xD**2 + 2.870*xD - 0.275
    return xD,yD


def _daylightphase(ccts, wl, add_wl = True, round_Mi_to_cie_recommended = True):
    
    xD,yD = _daylightlocus(ccts)
    
    denom = (0.0241 + 0.2562*xD - 0.7341*yD)
    M1 = (-1.3515 - 1.7703*xD +  5.9114*yD) / denom
    M2 = (0.0300 - 31.4424*xD + 30.0717*yD) / denom
    
    if round_Mi_to_cie_recommended: 
        M1, M2 = np.round(M1,3), np.round(M2,3) # round M1, M2 to 3 decimals (CIE recommendation):

    wl = getwlr(wl)
    S012 = np.array([np.interp(wl,_S012_DAYLIGHTPHASE[0], _S012_DAYLIGHTPHASE[i+1]) for i in range(3)])

    Sd = S012[0:1] + M1[:,None] * S012[1:2] + M2[:,None] * S012[2:3]

    if add_wl: 
        Sd = np.vstack((wl,Sd)) 

    return Sd
                      


def cri_ref(ccts, wl, cieobs = "1931_2", n = 1, c2 = None, round_daylightphase_Mi_to_cie_recommended = True, add_wl = True):
    
    # Calculate blackbody radiator and daylightphase for input ccts:
    if ccts.ndim == 2: ccts = ccts[:,0]
    Sb = _blackbody(ccts, wl = wl, n = n, c2 = c2, add_wl = True)
    Sd = _daylightphase(ccts, wl, add_wl = True, round_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended)
    
    # Normalize to equal Yw (Phi):
    #cieobs = "1964_10" 
    xyzb = spd_to_xyz(Sb, cieobs = cieobs, relative = False)
    xyzd = spd_to_xyz(Sd, cieobs = cieobs, relative = False)
    
    Phib, Phid = xyzb[...,1], xyzd[...,1]
    Sb = 100*Sb[1:]/Phib[:,None]
    Sd = 100*Sd[1:]/Phid[:,None]

    # CCT < 4000 K:
    Sr = Sb.copy()

    # CCT >= 5000 K:
    cnd = ccts>=5000
    if cnd.any():
        Sr[cnd] = Sd[cnd]

    # CCT >= 4000 K and CCT < 5000 K: 
    cnd = (ccts>=4000) & (ccts<5000)
    if cnd.any():
        wb = ((5000 - ccts[cnd])/1000)[:,None]
        Sr[cnd] = wb * Sb[cnd] + (1 - wb) * Sd[cnd]

    if add_wl: 
        Sr = np.vstack((wl,Sr)) 

    return Sr


# --- CCT & Duv ---------------------------------------------------------
def _get_cct_lut(wl = [360,830,1], read = False, cieobs = "1931_2", n = 1, c2 = None, 
                spacing = 0.25, T0 = 1000, Tn = 41000, unit = '%'):
   if read == False:
        if unit == '%': 
            N = np.floor(np.log(Tn/T0)/np.log(1+spacing/100))+2
            ccts = T0*(1 + spacing/100)**np.arange(-1,N)
        elif unit == 'K':
            ccts = np.arange(T0-spacing,Tn+2*spacing,spacing)
        Ss = _blackbody(ccts, wl = wl, n = n, c2 = c2, add_wl = True)
        xyzs = spd_to_xyz(Ss, cieobs = cieobs, relative = True)
        uvs = xyz_to_Yuv60(xyzs)[...,1:]
        lut = np.vstack((ccts, uvs.T)).T
   else:
      pass
   return lut

def _xyz_to_cct123(xyzw, lut = None, wl = [360,830,1], cieobs = "1931_2",
                             T0 = 1000, Tn = 41000, spacing = 0.25, unit = '%'):
    if lut is None:
        lut = _get_cct_lut(wl = wl, cieobs = cieobs, T0 = T0, Tn = Tn, spacing = spacing, unit = unit)

    uv = xyz_to_Yuv60(xyzw)[...,1:][:,None]
    d = ((uv - lut[...,1:])**2).sum(axis=-1)**0.5
    d = np.atleast_2d(d)
    i = d.argmin(axis = -1)
    im1 = i.copy() - 1
    ip1 = i.copy() + 1
    im1[im1<0] = 0
    ip1[ip1>d.shape[1]-1] = d.shape[1]-1
    d = d.T

    cct1, u1, v1, d1 = lut[im1,0], lut[im1,1], lut[im1,2], np.diag(d[im1])
    cct2, u2, v2, d2 = lut[i,0], lut[i,1], lut[i,2], np.diag(d[i])
    cct3, u3, v3, d3 = lut[ip1,0], lut[ip1,1], lut[ip1,2], np.diag(d[ip1])

    return uv, (cct1, u1, v1, d1), (cct2, u2, v2, d2), (cct3, u3, v3, d3)

def _sign_of_Duv(uv, u1,v1,d1,u3,v3,d3):
    # part of triangular solution (Ohno2014):
    d_tri = ((u3 - u1)**2 + (v3 - v1)**2)**0.5
    x = (d1**2 - d3**2 + d_tri**2) / (2 * d_tri)
    
    # Duv Sign:
    v_Planck = v1 + (v3 - v1) * x / d_tri
    sign = 1 if (uv[:,0,1] > v_Planck) else -1
    return sign

def _get_abs_duv(uv, cct, wl = [360,830,1], cieobs = "1931_2"):
    Ss = _blackbody(cct, wl = wl, add_wl = True)
    xyzs = spd_to_xyz(Ss, cieobs = cieobs, relative = True)
    uvs = xyz_to_Yuv60(xyzs)[...,1:]
    return ((uv - uvs)**2).sum(axis=-1)**0.5


def xyz_to_cctduv_bruteforce(xyzw, lut = None, wl = [360,830,1], cieobs = "1931_2", 
                             atol = 1e-15, Nmax = 1e4, down_sampling_factor = 10, use_fast_duv = True,
                             T0 = 1000, Tn = 41000, spacing = 0.25, unit = '%'): 
    
    if wl is not None: wl = getwlr(wl)

    (uv, 
    (cct1, u1, v1, d1),
    (cct2, u2, v2, d2),
    (cct3, u3, v3, d3)) = _xyz_to_cct123(xyzw, lut = lut, wl = wl, cieobs = cieobs,
                            T0 = T0, Tn = Tn, spacing = spacing, unit = unit)
    
    ccts = np.zeros(xyzw.shape[0])
    duvs = np.zeros(xyzw.shape[0])
    for i in range(xyzw.shape[0]):

        xyzwi = xyzw[i:i+1]
        uvi = uv[i:i+1]
        cct1i, cct2i, cct3i = cct1[i], cct2[i], cct3[i]
        u1i, u2i, u3i = u1[i], u2[i], u3[i]
        v1i, v2i, v3i = v1[i], v2[i], v3[i]
        d1i, d2i, d3i = d1[i], d2[i], d3[i]
        spacingi = spacing

        j = 0
        spacingi_is_ok = (np.log(1+spacingi/(down_sampling_factor*100))>0) if (unit == '%') else True
        while True & (j<=Nmax) & spacingi_is_ok:
            j+=1
            spacingi = spacingi/10
            #print(j, (cct3i - cct1i), spacingi, atol, cct1i, cct2i, cct3i)
            if (np.abs(cct3i - cct1i) < atol).any():
                ccti, ui, vi, di = cct2i, u2i, v2i, d2i
                break   
            else:
                (uvi,
                (cct1i, u1i, v1i, d1i),
                (cct2i, u2i, v2i, d2i),
                (cct3i, u3i, v3i, d3i)) = _xyz_to_cct123(xyzwi, lut = None, wl = wl, cieobs = cieobs,
                                        T0 = cct1i, Tn = cct3i, spacing = spacingi, unit = unit)
                cct1i, cct2i, cct3i = cct1i[0], cct2i[0], cct3i[0]
                u1i, u2i, u3i = u1i[0], u2i[0], u3i[0]
                v1i, v2i, v3i = v1i[0], v2i[0], v3i[0]
                d1i, d2i, d3i = d1i[0], d2i[0], d3i[0]
                ccti, ui, vi, di = cct2i, u2i, v2i, d2i
            spacingi_is_ok = (np.log(1+spacingi/(down_sampling_factor*100))>0) if (unit == '%') else (spacingi/10>0)
 
        sign = _sign_of_Duv(uvi, u1i,v1i,d1i,u3i,v3i,d3i)
        if use_fast_duv == False: di = _get_abs_duv(uvi, ccti, wl = wl, cieobs = cieobs)
        ccts[i] = ccti
        duvs[i] = di*sign
    return ccts[:,None], duvs[:,None]


def xyz_to_cctduv_ohno2014(xyzw, lut = None, wl = [360,830,1], cieobs = "1931_2"):
    #xyzw = np.vstack((xyzw,xyzw,xyzw)) # for testing multiple xyzw's

    if lut is None:
        lut = _get_cct_lut(wl = wl, cieobs = cieobs)

    uv = xyz_to_Yuv60(xyzw)[...,1:][:,None]
    d = ((uv - lut[...,1:])**2).sum(axis=-1)**0.5
    d = np.atleast_2d(d)
    i = d.argmin(axis = -1)
    im1 = i.copy() - 1
    ip1 = i.copy() + 1
    im1[im1<0] = 0
    ip1[ip1>d.shape[1]-1] = d.shape[1]-1
    d = d.T

    cct1, u1, v1, d1 = lut[im1,0], lut[im1,1], lut[im1,2], np.diag(d[im1])
    cct2, u2, v2, d2 = lut[i,0], lut[i,1], lut[i,2], np.diag(d[i])
    cct3, u3, v3, d3 = lut[ip1,0], lut[ip1,1], lut[ip1,2], np.diag(d[ip1])

    # triangular solution:
    d_tri = ((u3 - u1)**2 + (v3 - v1)**2)**0.5
    x = (d1**2 - d3**2 + d_tri**2) / (2 * d_tri)
    CCT_triangular = cct1 + (cct3 - cct1) * x / d_tri

    # Duv Sign:
    v_Planck = v1 + (v3 - v1) * x / d_tri
    sign = np.ones_like(v_Planck)
    c = uv[:,0,1] > v_Planck
    sign[~c] = -1

    # Triangular Solution Duv
    Duv_triangular_dist = ((d1**2 - x**2)**0.5)
    Duv_triangular = Duv_triangular_dist * sign

    # Parabolic Solution:
    a_ = d1 / (cct1 - cct2) / (cct1 - cct3)
    b_ = d2 / (cct2 - cct1) / (cct2 - cct3)
    c_ = d3 / (cct3 - cct1) / (cct3 - cct2)
    A = a_ + b_ + c_
    B = -1 * (a_ * (cct3 + cct2) + b_ * (cct1 + cct3) + c_ * (cct2 + cct1))
    C = a_ * cct2 * cct3 + b_ * cct1 * cct3 + c_ * cct1 * cct2
    CCT_parabolic = -B / (2 * A)
    Duv_parabolic = (A * CCT_parabolic** 2 + B * CCT_parabolic + C) * sign

    # Shifted Triangular Solution:
    CCT_triangular_shift = CCT_triangular + (CCT_parabolic - CCT_triangular) * Duv_triangular_dist * (1 / 0.002)

    # Set final CCT:
    Calc_CCT = CCT_triangular_shift.copy()
    Calc_CCT[Duv_triangular_dist >= 0.002] = CCT_parabolic[Duv_triangular_dist >= 0.002] # Tolerance of 0.002 is somewhat arbitrary. 0.002 suggested by YO

    #Set final Duv
    Calc_Duv= Duv_triangular.copy()
    Calc_Duv[Duv_triangular >= 0.002] = Duv_parabolic[Duv_triangular >= 0.002] # Tolerance of 0.002 is somewhat arbitrary. 0.002 suggested by YO   

    return Calc_CCT[:,None], Calc_Duv[:,None]

# --- CAM02UCS (fixed conditions of IES TM30) -----------------------------
def xyz_to_jabp(xyz, xyzw, use_fixed_k_FL_n_Nbb_Ncb_z = False):
    
    Yw = xyzw[...,1]
    
    # Initialize parameters:
    Yb = 20 
    Nc = 1 
    c = 0.69
    La = 100
    D = 1
    if use_fixed_k_FL_n_Nbb_Ncb_z:
        k,FL,n,Nbb,Ncb,z = 0.0020, 0.7937, 0.2, 1.0003, 1.0003, 1.9272 # from TM30-24 section 2.7
    else:
        k = 1/(5*La + 1) # 0.0020
        FL = (1/5)*k**4*(5*La) + (1/10)*(1-k**4)**2*(5*La)**(1/3) # 0.7937
        n = Yb/Yw # 0.2
        Nbb = 0.725*n**(-0.2) # 1.0003
        Ncb = Nbb
        z = 1.48 + n**0.5 # 1.9272
    
    # Get chromatic adaptation matrix:
    mcat = cat._MCATS['cat02']

    # Convert xyz to cat02 sensors:
    RGBw = (mcat @ xyzw.T).T
    RGB  = math.dot23(mcat,xyz.T).T

    # Apply von Kries CAT:
    RGBwc = (D*(100/RGBw) + (1-D))*RGBw
    RGBc  = (D*(100/RGBw) + (1-D))*RGB

    # get Hunt-Pointer-Estevez matrix fro conversion of xyz to cone sensors:
    mhpe = cat._MCATS['hpe']

    # Convert adapted cat02 sensors to cone sensors:
    RGBwp = (mhpe @ (np.linalg.inv(mcat) @ RGBwc.T)).T
    mcat_inv = np.linalg.inv(mcat)
    #mcat_inv = np.round(mcat_inv, 6) # as in TM30 excel calculator
    RGBp  = math.dot23(mhpe, (math.dot23(mcat_inv, RGBc.T))).T # note that rounding is important to get same results as excel calculator!

    # Apply Naka-Rushton cone compression and FL luminance adaptation:
    RGBwpa = (400*(FL*RGBwp/100)**0.42) / (27.13 + (FL*RGBwp/100)**0.42) + 0.1
    RGBpa  = (400*(FL* RGBp/100)**0.42) / (27.13 + (FL* RGBp/100)**0.42) + 0.1

    # Convert compressed and adapted cone signals to an achromatic signals and 2 opponent signals:
    Mop = np.array([[2, 1, 1/20], # -0.305 and subsequent multiplication still missing
                    [1, -12/11, 1/11],
                    [1/9, 1/9, -2/9]])
    Aabw = (Mop @ RGBwpa.T).T
    Aab  = math.dot23(Mop, RGBpa.T).T

    Aabw[...,0] = (Aabw[...,0] - 0.305)*Nbb 
    Aab[...,0]  = (Aab[...,0] - 0.305)*Nbb

    # Pre-calculate some variables:
    h = math.positive_arctan(Aab[...,1],Aab[...,2])
    #h = np.arctan2(Aab[...,2],Aab[...,1])*180/pi
    et = (1/4)*(np.cos(pi/180*h + 2) + 3.8)
    t = ((50000/13)*Ncb*Nc*et*(Aab[...,1]**2 + Aab[...,2]**2)**0.5) / (RGBpa[...,0] + RGBpa[...,1] + (21/20)*RGBpa[...,2])

    # Calculate ciecam02 lightness J, Chroma C and colourfulness M: 
    J = 100*(Aab[...,0]/Aabw[...,0])**(c*z)
    C = t**0.9 * (J/100)**0.5 * (1.64 - 0.29**n)**0.73
    M = C * FL**0.25

    # Convert ciecam02 J,a,b to cam02ucs J', a', b' coordinates:
    Mp = (1/0.0228)*np.log(1 + 0.0228*M)
    Jp = 1.7*J / (1 + 0.007*J)
    ap = Mp * np.cos(pi/180*h)
    bp = Mp * np.sin(pi/180*h)

    # group everything in a single value:
    jabp = np.array((Jp,ap,bp))

    return np.transpose(jabp,(1,2,0)) # ensure array has order of sample, light source, colorimetric coordinates

# --- IES TM30-18/24 ---------------------------------------------------------
def _spd_to_jabt_jabr(St, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):


    # select 99 CES:
    _str = '1nm' if use_1nm_rfls else '5nm'
    sampleset = _RFL['cri']['ies-tm30']['99'][_str]

    # CCT calculation:
    # get LUT:
    lut = _get_cct_lut(lx.getwlr([360,830,1])) # [360,830,1] range to match closest to TM30-24 excel calculator (even though xyzw is based on spectra with [380,780,1] range!)
   
    xyzwt2 = spd_to_xyz(St, cieobs = '1931_2')
    cct_t, duv_t = xyz_to_cctduv_ohno2014(xyzwt2, lut)
    if verbosity>0: print('CCT_t, Duv_t:', cct_t[0,0], duv_t[0,0]) # 2939.60116213116, -0.0007401913854794
    
    # Calculate reference illuminant using 1931 2° CMFs:
    Sr = cri_ref(cct_t, wl = St[0], cieobs = "1931_2", round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended) # round_daylightphase_Mi_to_cie_recommended = False to match IES TM30 calculators!
    xyzwr2 = spd_to_xyz(Sr, cieobs = "1931_2")
    #xyzwr2 = lx.Yxy_to_xyz([[100,0.312789087317269,0.32910230622236]])
    cct_r, duv_r = xyz_to_cctduv_ohno2014(xyzwr2, lut)
    if verbosity>0: print('CCT_r, Duv_r:', cct_r[0,0], duv_r[0,0]) # 2939.63568315033, 0.0000034090132648035   
    
    # 1964 10° XYZt,w calculation:
    xyzt10, xyzwt10 = spd_to_xyz(St, cieobs = '1964_10', rfl = sampleset, relative = True, out = 2)
    if verbosity>0: print('xyzwt10:',xyzwt10[0,0],xyzwt10[0,1],xyzwt10[0,2]) # rounding ok up to second to last digit Z10: 40.9652880852489 in excel 

    xyzr10, xyzwr10 = spd_to_xyz(Sr, cieobs = '1964_10', rfl = sampleset, relative = True, out = 2)
    if verbosity>0: print('xyzwr10:',xyzwr10[0,0],xyzwr10[0,1],xyzwr10[0,2]) # rounding ok up to second to last digit Z10: 40.9652880852489 in excel 

    # Calculate jab for 99 CES under test and ref ill.:
    jabt = xyz_to_jabp(xyzt10, xyzwt10)
    jabr = xyz_to_jabp(xyzr10, xyzwr10)

    return {'t' : { 'jabp'   : jabt,  
                    '1931_2' : {'xyzw' : xyzwt2, 'cct'  : cct_t, 'duv'  : duv_t},
                    '1964_10': {'xyz' : xyzt10, 'xyzw' : xyzwt10},
                    'S' : St},
            'r' : { 'jabp'   : jabr,  
                    '1931_2' : {'xyzw' : xyzwr2, 'cct'  : cct_r, 'duv'  : duv_r},
                    '1964_10': {'xyz' : xyzr10, 'xyzw' : xyzwr10},
                    'S' : Sr
                    }
            } 

def _jabt_jabr_to_DEi(jabt, jabr):

    # Calculate DEi:
    fDEi = lambda x1, x2: ((x1-x2)**2).sum(axis=-1)**0.5
    DEi = fDEi(jabt, jabr)
 
    return DEi

def _DEi_to_Rf_Rfi(DEi, verbosity = 0):

    # Calculate Rf and Rfi from DEa and DEi:
    scale_factor = [6.73]
    fscale = lambda x: cri.log_scale(x, scale_factor = scale_factor)
    if verbosity>0: print('Scale_factor:', scale_factor[0])
    Rfi = fscale(DEi)
    Rf = fscale(np.mean(DEi, axis = 0, keepdims = True))
    if verbosity>0:print("Rf (step-by-step:", Rf)

    return Rf, Rfi

def spd_to_iesrf(St, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):

    # Calculate xyzw2, xyzw10, cct2, duv2 for white and xyz10 for samples for both test and reference illuminant (generated from cct of St):
    coldata = _spd_to_jabt_jabr(St, use_1nm_rfls = use_1nm_rfls, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = verbosity)
    jabt, jabr = coldata['t']['jabp'], coldata['r']['jabp']

    # Calculate jab color differences:
    coldata['DEi'] = _jabt_jabr_to_DEi(jabt, jabr)
    
    # Calculate Rf and Rfi from DEa and DEi:
    Rf, Rfi = _DEi_to_Rf_Rfi(coldata['DEi'], verbosity = verbosity)

    return Rf, Rfi, coldata


def _group_per_bin(x, i, binnrs):
    return [[x[i[:,k] == j,k,:] for k in range(x.shape[1])] for j in binnrs]


def _get_hue_bin_indices_and_group(jab):
    
    # setup hue bins:
    nbins = 16
    t0 = 0 # start angle of 1 bin
    dh = 360/nbins
    hbins = np.arange(t0,360 + dh, dh)
    binnrs = np.asarray(np.arange(0,16)+1,dtype=int)
 
    # calculate hues of samples:
    h = math.positive_arctan(jab[...,1], jab[...,2])
    i = np.searchsorted(hbins, h)

    jab_grouped_per_bin = _group_per_bin(jab, i, binnrs)

    return jab_grouped_per_bin, i, {'nbins' : nbins, 't0' : t0, 'dh' : dh, 'binnrs' : binnrs, 'hbins' : hbins, 'h' : h}

    
def _coldata_to_rg_rhj(coldata):
    jabt, jabr, DEi = coldata['t']['jabp'], coldata['r']['jabp'], coldata['DEi']
    jabr_grouped_per_bin, ir, bininfo_r = _get_hue_bin_indices_and_group(jabr)
    jabt_grouped_per_bin = _group_per_bin(jabt, ir, bininfo_r['binnrs'])
    DEi_grouped_per_bin = _group_per_bin(DEi[...,None], ir, bininfo_r['binnrs'])
    coldata['bininfo_r'] = bininfo_r

    jabr_avg_per_bin = np.array([[np.mean(jab[i], axis = 0) for i in range(len(jab))] for jab in jabr_grouped_per_bin])
    jabt_avg_per_bin = np.array([[np.mean(jab[i], axis = 0) for i in range(len(jab))] for jab in jabt_grouped_per_bin])
    coldata['t']['jabp_j'], coldata['r']['jabp_j'] = jabt_avg_per_bin, jabr_avg_per_bin
    coldata['DEj'] = np.array([[np.mean(DEi[i], axis = 0) for i in range(len(DEi))] for DEi in DEi_grouped_per_bin])[...,0]
    _, Rfj = _DEi_to_Rf_Rfi(coldata['DEj'], verbosity = 0)
    
    xt, yt = np.vstack((jabt_avg_per_bin[...,1],jabt_avg_per_bin[0,:,1])),np.vstack((jabt_avg_per_bin[...,2],jabt_avg_per_bin[0,:,2]))  
    xr, yr = np.vstack((jabr_avg_per_bin[...,1],jabr_avg_per_bin[0,:,1])),np.vstack((jabr_avg_per_bin[...,2],jabr_avg_per_bin[0,:,2]))  

    At = polyarea(xt.T, yt.T)
    Ar = polyarea(xr.T, yr.T)
    Rg = 100 * At/Ar

    return Rg[None], Rfj, coldata

def _coldata_to_local_hue_and_chroma_shifts(coldata):
    jabt_avg_per_bin, jabr_avg_per_bin = coldata['t']['jabp_j'], coldata['r']['jabp_j']

    thetaj = np.arange(coldata['bininfo_r']['dh']/2, coldata['bininfo_r']['dh']/2 + coldata['bininfo_r']['nbins']*coldata['bininfo_r']['dh'], coldata['bininfo_r']['dh'])[:,None]*np.pi/180
    chromarj = (jabr_avg_per_bin[...,1:]**2).sum(axis = -1)**0.5
    Rhsj = (-(jabt_avg_per_bin[...,1] - jabr_avg_per_bin[...,1])/chromarj)*np.sin(thetaj) + ((jabt_avg_per_bin[...,2] - jabr_avg_per_bin[...,2])/chromarj)*np.cos(thetaj) 
    Rcsj = 100*(((jabt_avg_per_bin[...,1] - jabr_avg_per_bin[...,1])/chromarj)*np.cos(thetaj) + ((jabt_avg_per_bin[...,2] - jabr_avg_per_bin[...,2])/chromarj)*np.sin(thetaj)) 
   
    return Rhsj, Rcsj

def spd_to_iestm30(spd, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):

    Rf, Rfi, coldata = spd_to_iesrf(spd, use_1nm_rfls = use_1nm_rfls, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = verbosity)

    Rg, Rfj, coldata = _coldata_to_rg_rhj(coldata)

    Rhsj, Rcsj = _coldata_to_local_hue_and_chroma_shifts(coldata)

    return {'Rf' : Rf, 'Rg' : Rg, 'Rfi' : Rfi, 'Rfhj' : Rfj, 'Rhshj' : Rhsj, 'Rcshj' : Rcsj, 'coldata' : coldata}


#---------------------------------------------------------------------------
# Calculate TM30-18/24 using main functions from luxpy:
import luxpy as lx
def _spd_to_jabt_jabr_lx(St, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):

    # select 99 CES:
    _str = '1nm' if use_1nm_rfls else '5nm'
    sampleset = _RFL['cri']['ies-tm30']['99'][_str]

    # CCT calculation:
    xyzwt2 = lx.spd_to_xyz(St, cieobs = '1931_2')
    cct_t, duv_t = lx.xyz_to_cct(xyzwt2, mode = 'ohno2014', force_tolerance = False, out = 'cct,duv')
    if verbosity>0: print('CCT_t, Duv_t:', cct_t[0,0], duv_t[0,0]) # 2939.60116213116, -0.0007401913854794
    
    # Calculate reference illuminant using 1931 2° CMFs:
    Sr = lx.cri_ref(cct_t, ref_type = 'ies-tm30', wl3 = St[0], cieobs = "1931_2", round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended)
    xyzwr2 = lx.spd_to_xyz(Sr, cieobs = "1931_2")
    cct_r, duv_r = lx.xyz_to_cct(xyzwr2, mode = 'ohno2014', force_tolerance = False, out = 'cct,duv')

    if verbosity>0:print('CCT_r, Duv_r:', cct_r[0,0], duv_r[0,0]) # 2939.63568315033, 0.0000034090132648035   
    
    # 1964 10° XYZt,w calculation:
    xyzt10, xyzwt10 = lx.spd_to_xyz(St, cieobs = '1964_10', rfl = sampleset, relative = True, out = 2)
    if verbosity>0: print('xyzwt10:',xyzwt10[0,0],xyzwt10[0,1],xyzwt10[0,2]) # rounding ok up to second to last digit Z10: 40.9652880852489 in excel 

    xyzr10, xyzwr10 = lx.spd_to_xyz(Sr, cieobs = '1964_10', rfl = sampleset, relative = True, out = 2)
    if verbosity>0: print('xyzwr10:',xyzwr10[0,0],xyzwr10[0,1],xyzwr10[0,2]) # rounding ok up to second to last digit Z10: 40.9652880852489 in excel 

    # Calculate jab for 99 CES under test and ref ill.:
    conditions = {'La': 100.0, 'surround': 'avg', 'D': 1.0, 'Yb': 20.0, 'Dtype': None}
    jabt = lx.cam.xyz_to_jab_cam02ucs(xyzt10, xyzw = xyzwt10, **conditions)
    jabr = lx.cam.xyz_to_jab_cam02ucs(xyzr10, xyzw = xyzwr10, **conditions)

    return {'t' : { 'jabp'   : jabt,  
                    '1931_2' : {'xyzw' : xyzwt2, 'cct'  : cct_t, 'duv'  : duv_t},
                    '1964_10': {'xyz' : xyzt10, 'xyzw' : xyzwt10},
                    'S' : St},
            'r' : { 'jabp'   : jabr,  
                    '1931_2' : {'xyzw' : xyzwr2, 'cct'  : cct_r, 'duv'  : duv_r},
                    '1964_10': {'xyz' : xyzr10, 'xyzw' : xyzwr10},
                    'S' : Sr
                    }
            } 


def spd_to_iesrf_lx(St, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):

    # Calculate xyzw2, xyzw10, cct2, duv2 for white and xyz10 for samples for both test and reference illuminant (generated from cct of St):
    coldata = _spd_to_jabt_jabr_lx(St, use_1nm_rfls = use_1nm_rfls, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = verbosity)
    jabt, jabr = coldata['t']['jabp'], coldata['r']['jabp']

    # Calculate jab color differences:
    coldata['DEi'] = _jabt_jabr_to_DEi(jabt, jabr)
    
    # Calculate Rf and Rfi from DEa and DEi:
    Rf, Rfi = _DEi_to_Rf_Rfi(coldata['DEi'], verbosity = verbosity)

    return Rf, Rfi, coldata


def spd_to_iestm30_lx(spd, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = True, verbosity = 0):

    Rf, Rfi, coldata = spd_to_iesrf_lx(spd, use_1nm_rfls = use_1nm_rfls, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = verbosity)

    Rg, Rfj, coldata = _coldata_to_rg_rhj(coldata)

    Rhsj, Rcsj = _coldata_to_local_hue_and_chroma_shifts(coldata)

    return {'Rf' : Rf, 'Rg' : Rg, 'Rfi' : Rfi, 'Rfhj' : Rfj, 'Rhshj' : Rhsj, 'Rcshj' : Rcsj, 'coldata' : coldata}




   
#===========================================================================
# --- TESTING -------------------------------------------------------------

# Test cct and duv calculations:
if __name__ == '__main__x':
    print('\nCCT and Duv calculations:')
    
    import luxpy as lx

    np.printoptions(precision=6)

    # Get lamp data (from TM30-24 calculator):
    F4_1 = pd.read_csv('./tmp_data/F4_1nm.csv', header = None).values.T
    F5_1 = lx._CIE_ILLUMINANTS['F6']
    spd = np.vstack((F4_1,F5_1[1]))

    xyzw = spd_to_xyz(spd, cieobs = '1931_2')

    # test bruteforce, ohno2014 and robertson2023 cct and duv calculations:
    #lut = _get_cct_lut(cieobs = '1931_2', wl = spd[0]) # to 3x speed up bruteforce calculation
    cct_bf, duv_bf = xyz_to_cctduv_bruteforce(xyzw, wl = spd[0], cieobs = "1931_2")
    print('\nResults:\n Bruteforce: CCT, Duv:', cct_bf.T, duv_bf.T) # 2939.60116213116

    cct_bf_lx, duv_bf_lx = lx.xyz_to_cct_bruteforce(xyzw, wl = spd[0], cmfs = "1931_2", out = 'cct,duv')
    #cct_bf_lx, duv_bf_lx = cctduv_bf_lx[:,0].T, cctduv_bf_lx[:,1]
    print(' Bruteforce luxpy: CCT, Duv:', cct_bf_lx.T, duv_bf_lx.T) # 2939.60116213116

    cct_bf_lx_nr, duv_bf_lx_nr = lx.xyz_to_cct_bruteforce(xyzw, wl = spd[0], cmfs = "1931_2", out = 'cct,duv', use_newton_raphson = True)
    #cct_bf_lx, duv_bf_lx = cctduv_bf_lx[:,0].T, cctduv_bf_lx[:,1]
    print(' Bruteforce with nr luxpy: CCT, Duv:', cct_bf_lx_nr.T, duv_bf_lx_nr.T) # 2939.60116213116


    cct_o, duv_o = xyz_to_cctduv_ohno2014(xyzw, wl = spd[0])
    print(' Ohno2014: CCT, Duv:', cct_o.T, duv_o.T) # 2939.60116213116

    cct_ro23, duv_ro23 = lx.xyz_to_cct(xyzw, wl = spd[0], mode = "robertson2023", force_tolerance = True, out = 'cct,duv')
    print(' Robertson 2023 luxpy: CCT, Duv:', cct_ro23.T, duv_ro23.T) # 2939.60116213116

    print('\n- Brute force and Roberston 2023 (with Newton-Raphson turned on \n  give same results (timings are comparable when LUT is pre-calculated\n  for the bruteforce calculation! Ohno2014 is faster and gives same results\n  as bruteforce and Robertson 2023),\n- Ohno2014 gave slightly difference results.')

# Test IES TM30-18/24 calculations:
if __name__ == '__main__x':
    print('\n\nIES TM30-18/24 calculations:')
    
    # Calculate Rf, Rfi, Rfj, Rg:
    iestm30_lx_ = lx.cri.spd_to_ies_tm30_metrics(spd)
    Rf_lx_ = lx.cri.spd_to_iesrf(spd)
    Rf_excel = 56.8377108949956 # from 1 nm TM30-24 excel calculator
    Rg_lx_ = lx.cri.spd_to_iesrg(spd)
    Rg_excel = 83.5353059489586
    
    iestm30 = spd_to_iestm30(spd, use_1nm_rfls = True, verbosity = 0)
    Rf, Rg, Rfi, Rfj, Rhsj, Rcsj, coldata = [iestm30[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj', 'coldata']]

    iestm30_ = spd_to_iestm30(spd, use_1nm_rfls = False, verbosity = 0)
    Rf_, Rg_, Rfi_, Rfj_, Rhsj_, Rcsj_, coldata_ = [iestm30_[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj', 'coldata']]


    iestm30_lx = spd_to_iestm30_lx(spd, use_1nm_rfls = True, verbosity = 0)
    Rf_lx, Rg_lx, Rfi_lx, Rfj_lx, Rhsj_lx, Rcsj_lx, coldata_lx = [iestm30_lx[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj',  'coldata']]


    print('\nResults:\n Rf:',  Rf_excel, Rf_lx_[0,0], Rf_lx[0,0], Rf[0,0]) # 56.8377108949956
    print(' Diff. Rf:',  Rf_lx_[0,0] - Rf_excel, Rf_lx[0,0] - Rf_excel, Rf[0,0] - Rf_excel) 

    print('\n Rg:', Rg_excel, Rg_lx_[0,0], Rg_lx[0,0], Rg[0,0])# 83.5353059489586
    print(' Diff. Rg:', Rg_lx_[0,0] - Rg_excel, Rg_lx[0,0] - Rg_excel, Rg[0,0] - Rg_excel) # 83.5353059489586

    print('\n- Jump from ~1e-9 to ~1e-6 in error when using luxpy\n  due to not rounding the inv(cat02) matrix  to 6 decimals!')
    print('- Rounding xyzw10 to 15 decimals (as in excel calculator) or not gave Rf, Rg differences of ~1e-13!')
    print('- Using the 5 nm instead of 1 nm rfls gave Rf, Rg differences of up to ~1e-4!')

        # Local Jabt, Jabr from M30-24 excel (F4):
        # 17.57	    2.73		24.04	5.11
        # 12.76	    16.21		20.58	15.10
        # 5.65	    26.19		14.84	22.89
        # -2.71	    24.46		2.72	20.40
        # -5.57	    19.67		-2.93	16.78
        # -11.79	20.18		-12.75	16.87
        # -14.91	18.70		-18.74	13.09
        # -16.28	7.52		-21.45	5.64
        # -17.14	-2.01		-22.58	-3.88
        # -11.41	-12.32		-18.08	-11.97
        # -4.25	    -16.64		-10.88	-15.82
        # -2.29	    -23.00		-6.60	-21.17
        # 3.75	    -20.37		3.27	-18.03
        # 7.93	    -17.21		9.38	-13.98
        # 15.57	    -18.60		18.91	-13.44
        # 12.39	    -6.88		16.89	-4.18

# Test IES TM30-18/24 calculations for all 318 sources in TM30 excel calculator:
if __name__ == '__main__':

    import pandas as pd
    import numpy as np


    print('\n\nIES TM30-18/24 calculations for all 318 sources in TM30 excel calculator::')

    # Get TM30-24 spds and calculation results of excel calculator:
    file = './tmp_data/IES-TM30-24-318_sources.xlsx'
    tmp = pd.read_excel(file, sheet_name = 'Example SPDs')
    spd_names = tmp.values[0,1:]
    spd = np.vstack((lx.getwlr([380,780,1]),np.asarray(tmp.values[19:,1:].T,float)))
    tmp2 = pd.read_excel(file, sheet_name = 'Multiple Calculator', skiprows = 19)
    result_names = tmp2.values[:,4]
    result_excel = np.asarray(tmp2.values[:,5:], float) 
    result_excel[4:4+16,:]*=100 # Rcshj in procent

    # generates extreme difference for ..._lx
    spd, result_excel = spd[[0,300+1]], result_excel[:,300:301]

    # Calculate Rf, Rfi, Rfj, Rg:
    iestm30_lx_ = lx.cri.spd_to_ies_tm30_metrics(spd)
    cct_lx_, duv_lx_, Rf_lx_, Rg_lx_, Rfi_lx_, Rfj_lx_, Rhsj_lx_, Rcsj_lx_  = [iestm30_lx_[x] for x in ['cct', 'duv', 'Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj']]
    Rcsj_lx_ *= 100 #in procent
    result_lx_ = np.vstack((cct_lx_.T,duv_lx_.T, Rf_lx_, Rg_lx_, Rcsj_lx_, Rhsj_lx_, Rfj_lx_))

    # Calculate Rf, Rfi, Rfj, Rg:
    iestm30_lx_f = lx.cri.spd_to_tm30_fast(spd)
    cct_lx_f, duv_lx_f, Rf_lx_f, Rg_lx_f, Rfi_lx_f, Rfj_lx_f, Rhsj_lx_f, Rcsj_lx_f  = [iestm30_lx_f[x] for x in ['cct', 'duv', 'Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj']]
    Rcsj_lx_f *= 100 #in procent
    result_lx_f = np.vstack((cct_lx_f.T,duv_lx_f.T, Rf_lx_f, Rg_lx_f, Rcsj_lx_f, Rhsj_lx_f, Rfj_lx_f))


    round_daylightphase_Mi_to_cie_recommended = False 

    iestm30 = spd_to_iestm30(spd, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = 0)
    Rf, Rg, Rfi, Rfj, Rhsj, Rcsj, coldata = [iestm30[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj', 'coldata']]
    cct, duv = coldata['t']['1931_2']['cct'],  coldata['t']['1931_2']['duv']
    result = np.vstack((cct.T,duv.T, Rf, Rg, Rcsj, Rhsj, Rfj))

    iestm30_ = spd_to_iestm30(spd, use_1nm_rfls = False, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = 0)
    Rf_, Rg_, Rfi_, Rfj_, Rhsj_, Rcsj_, coldata_ = [iestm30_[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj', 'coldata']]
    cct_, duv_ = coldata_['t']['1931_2']['cct'],  coldata_['t']['1931_2']['duv']
    result_ = np.vstack((cct_.T,duv_.T, Rf_, Rg_, Rcsj_, Rhsj_, Rfj_))

    iestm30_lx = spd_to_iestm30_lx(spd, use_1nm_rfls = True, round_daylightphase_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended, verbosity = 0)
    Rf_lx, Rg_lx, Rfi_lx, Rfj_lx, Rhsj_lx, Rcsj_lx, coldata_lx = [iestm30_lx[x] for x in ['Rf', 'Rg', 'Rfi', 'Rfhj', 'Rhshj', 'Rcshj',  'coldata']]
    cct_lx, duv_lx = coldata_lx['t']['1931_2']['cct'],  coldata_lx['t']['1931_2']['duv']
    result_lx = np.vstack((cct_lx.T,duv_lx.T, Rf_lx, Rg_lx, Rcsj_lx, Rhsj_lx, Rfj_lx))


    # print max difference for each quantity between various implementations and the excel calculator:
    print("\nMax difference for each quantity between various implementations and the excel calculator:")
    for i in range(result.shape[0]):
        print('Max. diff:',result_names[i],np.abs(result_excel - result_lx_)[i,:].max(),np.abs(result_excel - result_lx_f)[i,:].max(),np.abs(result_excel - result)[i,:].max(),np.abs(result_excel - result_)[i,:].max(),np.abs(result_excel - result_lx)[i,:].max())
    print('Max diff.: all:', np.abs(result_excel - result_lx_).max(),np.abs(result_excel - result_lx_f).max(),np.abs(result_excel - result).max(),np.abs(result_excel - result_).max(),np.abs(result_excel - result_lx).max())


    # # print procent difference between 1nm and 5 nm sampleset implementations:
    # print("\nMax. procent difference between 1nm and 5 nm sampleset implementations:")
    # for i in range(result.shape[0]):
    #     print('Max. % diff:',result_names[i],100*(np.abs(result - result_)/result)[i,:].max())
    # print('Max % diff.: all:', (100*np.abs(result - result_)/result).max())
    # print("! Using a 5 nm instead of a 1 nm sampleset can cause large differences (e.g. up to 15)\nfor the hue-shift in each bin !! (The others quantities are much more similar 0.015 % or better)")

    # print procent difference between 1nm and 5 nm sampleset implementations:
    print("\nMax. procent difference between 1nm and 5 nm sampleset implementations:")
    for i in range(result.shape[0]):
        print('Max. % diff:',result_names[i],100*(np.abs(result_excel - result)/result)[i,:].max())
    print('Max % diff.: all:', (100*np.abs(result_excel - result)/result).max())
    #print(iestm30['coldata']['r']['1931_2']['cct'][0,0])

    # D65 = lx._CIE_D65
    # xyzd65 = spd_to_xyz(D65) 
    # wl = [360,830,1]
    # cctd65,duvd65 = xyz_to_cctduv_bruteforce(xyzd65, wl = wl)
    # Sd = _daylightphase(cctd65[0], wl, round_Mi_to_cie_recommended=False)
    # Sd_round = _daylightphase(cctd65[0], wl, round_Mi_to_cie_recommended=False)
    
    # xyzd65_ = spd_to_xyz(Sd)
    # xyzd65_round = spd_to_xyz(Sd_round)
    # cctd65_,duvd65_ = xyz_to_cctduv_bruteforce(xyzd65_, wl = wl)
    # print(cctd65,cctd65_,cctd65-cctd65_)
    # cctd65_round,duvd65_round = xyz_to_cctduv_bruteforce(xyzd65_round, wl = wl)
    # print(cctd65,cctd65_,cctd65-cctd65_round)
    # print('\nRounding of M1, M2 coefficients in daylight phase calculations \nto 3 decimals is required to get a more accurate D65 calculation!')

    for i in range(result.shape[0]):
        print(result_names[i],np.abs(result_lx_f - result_lx_)[i,:].max())