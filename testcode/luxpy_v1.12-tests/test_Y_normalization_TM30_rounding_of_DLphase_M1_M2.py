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

from ies_tm30_test import xyz_to_cctduv_bruteforce, _blackbody, _daylightphase, spd_to_xyz
from luxpy import getwlr, xyz_to_cct 

if __name__ == '__main__':

    ccts = np.arange(4000,5000+10,10)#np.array([4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000])
    cieobss = ["1931_2","1964_10"]
    colors = ["b","r"]
    wl = getwlr([380,780,1])
    add_wl = True
    n = 1
    c2 = None
    round_daylightphase_Mi_to_cie_recommended = False
    
    cctr = []
    for i,cieobs in enumerate(cieobss):
        # Calculate blackbody radiator and daylightphase for input ccts:
        if ccts.ndim == 2: ccts = ccts[:,0]
        Sb = _blackbody(ccts, wl = wl, n = n, c2 = c2, add_wl = True)
        Sd = _daylightphase(ccts, wl, add_wl = True, round_Mi_to_cie_recommended = round_daylightphase_Mi_to_cie_recommended)
        
        # Normalize to equal Yw (Phi):
        xyzb = spd_to_xyz(Sb, cieobs = cieobs, relative = False)
        xyzd = spd_to_xyz(Sd, cieobs = cieobs, relative = False)
        Phib, Phid = xyzb[...,1], xyzd[...,1]

        # if cieobs == "1964_10": 
        #     Phib = Sb[1:].sum(-1)
        #     Phid = Sd[1:].sum(-1)

        Sb = 100*Sb[1:]/Phib[:,None]
        Sd = 100*Sd[1:]/Phid[:,None]

        # CCT < 4000 K:
        Sr = Sb.copy()

        # CCT >= 5000 K:
        cnd = ccts>=5000
        if cnd.any():
            Sr[cnd] = Sd[cnd]

        # CCT >= 4000 K and CCT < 5000 K: 
        cnd = (ccts>=4000) & (ccts<=5000)
        if cnd.any():
            wb = ((5000 - ccts[cnd])/1000)[:,None]
            Sr[cnd] = wb * Sb[cnd] + (1 - wb) * Sd[cnd]

        if add_wl: 
            Sr = np.vstack((wl,Sr)) 
            Sb = np.vstack((wl,Sb))
            Sd = np.vstack((wl,Sd))  

        cieobs_ = "1931_2"
        #cieobs_ = cieobs
        xyzb = spd_to_xyz(Sb, cieobs = cieobs_, relative = False)
        xyzd = spd_to_xyz(Sd, cieobs = cieobs_, relative = False)
        xyzr = spd_to_xyz(Sr, cieobs = cieobs_, relative = False)
        Phib, Phid, Phir = xyzb[...,1], xyzd[...,1], xyzr[...,1]

        cctr_,duvr = xyz_to_cct(xyzr, wl = wl, out = 'cct,duv', force_tolerance=True)
        for j in range(len(ccts)):
            print(cieobs,ccts[j]-cctr_[j,0],Phir[j])
        plt.plot(ccts,ccts-cctr_[:,0],colors[i]+'.')
        cctr.append(cctr_)

    plt.figure()
    plt.plot(ccts, cctr[0]-cctr[1], 'b.')
    print('Max CCT error by mixing in 1931_2 vs 1964_10:', (cctr[0]-cctr[1]).max())

    # impact of rounding of M1 and M2 in daylight phase calculations
    print("\nImpact of rounding of M1 and M2 in daylight phase calculations")
    ccts = np.arange(4000,7000+10,10)
    wl = getwlr([380,780,1])
    #wl = getwlr([360,830,5])
    Sd_ = _daylightphase(ccts, wl, round_Mi_to_cie_recommended=False)
    Sd_round = _daylightphase(ccts, wl, round_Mi_to_cie_recommended=True)
    xyz_ = spd_to_xyz(Sd_)
    xyz_round = spd_to_xyz(Sd_round)
    wl_ = wl#None

    #cct_,duv_ = xyz_to_cct(xyz_, wl = wl_, out = 'cct,duv', force_tolerance = True, atol = 1e-15, rtol = 1e-20)
    #cct_round,duv_round = xyz_to_cct(xyz_round, wl = wl_, out = 'cct,duv', force_tolerance = True, atol = 1e-15, rtol = 1e-20)
    
    cct_,duv_ = xyz_to_cctduv_bruteforce(xyz_, wl = wl_, atol = 1e-15)
    cct_round,duv_round = xyz_to_cctduv_bruteforce(xyz_round, wl = wl_, atol = 1e-15)
    
    plt.figure()
    plt.plot(ccts,np.abs(ccts - cct_round[:,0]),'b.', label = 'Rounded M1 M2 (3 decimals)')
    plt.plot(ccts,np.abs(ccts - cct_[:,0]),'r.', label = 'Unrounded M1 M2')
    if wl_ is None: 
        plt.title(f'wl = [{wl[0]:1.0f},{wl[-1]:1.0f},{wl[1]-wl[0]:1.0f}] (CCT calculation->[360,830,1])')
    else:
        plt.title(f'wl = [{wl[0]:1.0f},{wl[-1]:1.0f},{wl[1]-wl[0]:1.0f}] ')  
    plt.legend()
    
