
# -*- coding: utf-8 -*-
#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

#------other imports------------------------------------------------------
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

# test difference between vlbar and ybar from CMF:
spds = lx._IESTM3018['S']['data'].copy()
spd = spds[[0,1]]

cmf2,K2 = lx._CMF["1931_2"]["bar"].copy(),lx._CMF["1931_2"]["K"]
cmf2_ = lx.xyzbar(cieobs="1931_2")
print('cmf2==cmf2_ ?:', (((cmf2-cmf2_)==0).all()))

vl = lx.utils.getdata(r'./tmp_data/vl1924.csv')
vl1924 = np.array([[int(vl[i][0]),float(vl[i][1])] for i in range(len(vl)) if (i%2==0)]).T
print('cmf2_ybar==vl1924 ?:', (((cmf2[2]-vl1924[1])==0).all()))

xyz1 = lx.spd_to_xyz(spd, cieobs = "1931_2", relative = False)
xyz2 = lx.spd_to_xyz(spd, cieobs = cmf2, K=K2, relative = False)
print("xyz1==xyz2 ?:", ((xyz1-xyz2)==0).all())

E1 = lx.spd_to_power(spd, ptype = 'pu', cieobs = "1931_2")
E2 = lx.spd_to_power(spd, ptype = 'pu', cieobs = cmf2, K=K2)
print("E1==E2 ?:", ((E1-E2)==0).all())

Vl1, Km1 = lx.vlbar(cieobs = "1931_2", K = None, src = 'dict', wl_new = spd[0], out = 2, 
                       interp_settings = None, kind = None, extrap_kind = None, extrap_values = None)
Vl2, Km2 = lx.vlbar(cieobs = cmf2, K = K2, src = 'vltype', wl_new = spd[0], out = 2, 
                       interp_settings = None, kind = None, extrap_kind = None, extrap_values = None)
# plt.plot(vl1924[0],vl1924[1],'r-')
# plt.plot(Vl1[0],Vl1[1],'b--')
# plt.plot(Vl2[0],Vl2[1],'g:')
print("Vl1 == Vl2 ?:", ((Vl1[1]-Vl2[1])==0).all())

from luxpy.toolboxes.photbiochem import _ACTIONSPECTRA, spd_to_aopicE, spd_to_aopicEDI, spd_to_aopicELR
Eea = spd_to_aopicE(spd, out = 'Eeas')*1000
print("Eea:",Eea)
EDI = spd_to_aopicEDI(spd)
print("EDI:",EDI)
ELR = spd_to_aopicELR(spd)
print("ELR:",ELR)

as26 = _ACTIONSPECTRA[:,20:-50] # sample to 380-780 nm range
as26i = lx.cie_interp(_ACTIONSPECTRA,spd[0], kind = 'cmf')
print("as26 == as26i: ?", ((as26 - as26i)==0).all())

as26n = lx.spd(as26, wl = spd[0], datatype = 'cmf', norm_type = None, interp_settings = None)
print("as26 == as26n: ?", ((as26 - as26n)==0).all())
print('Norm to max causes issues, but original data is not max at 1!-> delete max-normalisation in luxpy Dec 18, 2024')
# for i in range(5):
#     plt.plot(as26[0],as26[i+1],'r-')
#     plt.plot(as26n[0],as26n[i+1],'g:')
#     print(f'{i}->as26-as26n:', as26[i+1]-as26n[i+1])

dl = lx.getwld(spd[0])
Eea2 = 1000*(dl*spd[1:]) @ as26[1:].T
for i in range(5):
    print(Eea2[0,i])
print(Eea2-Eea)

ELR_1 = Eea / E1
print('ELR_1 (mW/lm): ', ELR_1)

ELR_2 = spd_to_aopicELR(spd, cieobs = '1931_2')*1000
print('ELR_2 (mW/lm): ', ELR_2)
print('Delta(ELR_1,ELR_2):',ELR_1-ELR_2)

ELR_luox_excel = np.array([0.7792, 1.4294, 1.6102, 1.2962, 1.1533])[[2,1,0,3,4]]
print(ELR_luox_excel-ELR_1)
print(ELR_luox_excel-ELR_2)
print('delta(ELR_1,ELR_2):',ELR_1-ELR_2)


D65 = lx.cie_interp(lx._CIE_D65, spd[0], datatype = 'spd') # make D65 same wavelength range as spd
ELR_D65_1 = spd_to_aopicELR(D65)*1000
print('ELR of D65 (mW/lm):', ELR_D65_1)

ELR_D65_2,E_D65_2 = spd_to_aopicE(D65, out = 'Eeas,E')
ELR_D65_2 *= (1000 / E_D65_2)
print('ELR of D65 (mW/lm):', ELR_D65_2)
print('Delta(ELR_D65_1, ELR_D65_2):', ELR_D65_1 - ELR_D65_2)

ELR_D65_toolbox = np.array([1.62890776589039, 1.45582633881653, 0.817289644883213, 1.4497035760559, 1.32621318911359])
print('ELR of D65 (mW/lm) from Toolbox:', ELR_D65_toolbox)
print('Delta(ELR_D65_2, ELR_D65 Toolbox):', ELR_D65_2 - ELR_D65_toolbox)

EDI_1 = spd_to_aopicEDI(spd, cieobs='1931_2')
print('EDI_1: ', EDI_1)

EDI_2 = Eea / ELR_D65_1 
print('EDI_2: ', EDI_2)
print('Delta(EDI_1, EDI_2):', EDI_1 - EDI_2)

EDI_luox_excel=np.array([21827.12,	22477.68,	22630.31,	20469.64,	19908.66])[[2,1,0,3,4]]
print(EDI_luox_excel-EDI_1)
print(EDI_luox_excel-EDI_2)
print('delta(EDI_1,EDI_2):',EDI_1-EDI_2)



