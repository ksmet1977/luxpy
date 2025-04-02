#testCIEDE2000.py


#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

from luxpy import cam 
from luxpy.color.deltaE.colordifferences import _process_DEi, deltaH, DE2000

#-----other imports-------------------------------------------------------
import numpy as np
import pandas as pd

# def DE2000(xyzt, xyzr, dtype = 'xyz', DEtype = 'jab', avg = None, avg_axis = 0, out = 'DEi',
#               xyzwt = None, xyzwr = None, KLCH = None):
    
#     """
#     Calculate DE2000 color difference.
    
#     Args:
#         :xyzt: 
#             | ndarray with tristimulus values of test data.
#         :xyzr:
#             | ndarray with tristimulus values of reference data.
#         :dtype:
#             | 'xyz' or 'lab', optional
#             | Specifies data type in :xyzt: and :xyzr:.
#         :xyzwt:
#             | None or ndarray, optional
#             |   White point tristimulus values of test data
#             |   None defaults to the one set in lx.xyz_to_lab()
#         :xyzwr:
#             | None or ndarray, optional
#             |    Whitepoint tristimulus values of reference data
#             |    None defaults to the one set in lx.xyz_to_lab()
#         :DEtype:
#             | 'jab' or str, optional
#             | Options: 
#             |    - 'jab' : calculates full color difference over all 3 dimensions.
#             |    - 'ab'  : calculates chromaticity difference.
#             |    - 'j'   : calculates lightness or brightness difference 
#             |             (depending on :outin:).
#             |    - 'j,ab': calculates both 'j' and 'ab' options 
#             |              and returns them as a tuple.
#         :KLCH: 
#             | None, optional
#             | Weigths for L, C, H 
#             | None: default to [1,1,1] 
#         :avg:
#             | None, optional
#             | None: don't calculate average DE, 
#             |       otherwise use function handle in :avg:.
#         :avg_axis:
#             | axis to calculate average over, optional
#         :out: 
#             | 'DEi' or str, optional
#             | Requested output.
        
#     Note:
#         For the other input arguments, see specific color space used.
        
#     Returns:
#         :returns: 
#             | ndarray with DEi [, DEa] or other as specified by :out:
            
#     References:
#         1. `Sharma, G., Wu, W., & Dalal, E. N. (2005). 
#         The CIEDE2000 color‐difference formula: Implementation notes, 
#         supplementary test data, and mathematical observations. 
#         Color Research & Application, 30(1), 21–30. 
#         <https://doi.org/10.1002/col.20070>`_
#     """
    
#     if KLCH is None:
#         KLCH = [1,1,1]
    
#     if dtype == 'xyz':
#         labt = xyz_to_lab(xyzt, xyzw = xyzwt)
#         labr = xyz_to_lab(xyzr, xyzw = xyzwr)
#     else:
#         labt = xyzt
#         labr = xyzr
 
#     Lt = labt[...,0:1]
#     at = labt[...,1:2]
#     bt = labt[...,2:3]
#     Ct = np.sqrt(at**2 + bt**2)
#     #ht = cam.hue_angle(at,bt,htype = 'rad')
    
#     Lr = labr[...,0:1]
#     ar = labr[...,1:2]
#     br = labr[...,2:3]
#     Cr = np.sqrt(ar**2 + br**2)
#     #hr = cam.hue_angle(at,bt,htype = 'rad')
    
#     # Step 1:
#     Cavg = (Ct + Cr)/2
#     G = 0.5*(1 - np.sqrt((Cavg**7.0)/((Cavg**7.0) + (25.0**7))))
#     apt = (1 + G)*at
#     apr = (1 + G)*ar
    
#     Cpt = np.sqrt(apt**2 + bt**2)
#     Cpr = np.sqrt(apr**2 + br**2)
#     Cpprod = Cpt*Cpr


#     hpt = cam.hue_angle(apt,bt, htype = 'deg')
#     hpr = cam.hue_angle(apr,br, htype = 'deg')
#     hpt[(apt==0)*(bt==0)] = 0
#     hpr[(apr==0)*(br==0)] = 0
    
#     # Step 2:
#     dL = (Lr - Lt)
#     dCp = (Cpr - Cpt)
#     dhp_ = hpr - hpt  


#     dhp = dhp_.copy()
#     dhp[np.where((dhp_) > 180)] = dhp[np.where((dhp_) > 180)] - 360
#     dhp[np.where((dhp_) < -180)] = dhp[np.where((dhp_) < -180)] + 360
#     dhp[np.where(Cpprod == 0)] = 0

#     #dH = 2*np.sqrt(Cpprod)*np.sin(dhp/2*np.pi/180)
#     dH = deltaH(dhp, Cpprod, htype = 'deg')

#     # Step 3:
#     Lp = (Lr + Lt)/2
#     Cp = (Cpr + Cpt)/2
    
#     hps = hpt + hpr
#     hp = (hpt + hpr)/2
#     hp[np.where((np.abs(dhp_) > 180) & (hps < 360))] = hp[np.where((np.abs(dhp_) > 180) & (hps < 360))] + 180
#     hp[np.where((np.abs(dhp_) > 180) & (hps >= 360))] = hp[np.where((np.abs(dhp_) > 180) & (hps >= 360))] - 180
#     hp[np.where(Cpprod == 0)] = 0

#     T = 1 - 0.17*np.cos((hp - 30)*np.pi/180) + 0.24*np.cos(2*hp*np.pi/180) +\
#         0.32*np.cos((3*hp + 6)*np.pi/180) - 0.20*np.cos((4*hp - 63)*np.pi/180)
#     dtheta = 30*np.exp(-((hp-275)/25)**2)
#     RC = 2*np.sqrt((Cp**7)/((Cp**7) + (25**7)))
#     SL = 1 + ((0.015*(Lp-50)**2)/np.sqrt(20 + (Lp - 50)**2))
#     SC = 1 + 0.045*Cp
#     SH = 1 + 0.015*Cp*T
#     RT = -np.sin(2*dtheta*np.pi/180)*RC
   
#     kL, kC, kH = KLCH
    
#     DEi = ((dL/(kL*SL))**2 , (dCp/(kC*SC))**2 + (dH/(kH*SH))**2 + RT*(dCp/(kC*SC))*(dH/(kH*SH)))

#     return _process_DEi(DEi, DEtype = DEtype, avg = avg, avg_axis = avg_axis, out = out)


if __name__ == '__main__x':

    ref = np.array([
                    [54,	-37,	-50],
                    [47,	75,	-6]
                    ])

    sam = np.array([
                    [54,	-34,	-52],
                    [47,	72,	-2]
                    ])

    DErs = DE2000(ref, sam, dtype = 'lab')
    DEsr = DE2000(sam, ref, dtype = 'lab')

    print('diff DE2000 12<->21:',(DErs - DEsr).T)
    print('DE2000:',DEsr[0,0],DEsr[1,0])

if __name__ == '__main__':

    data = pd.read_csv('./tmp_data/testCIEDE2000.csv', header = 'infer')

    ref = np.asarray(data.values[:,:3],dtype = float)
    sam = np.asarray(data.values[:,3:6], dtype = float)
    DE00 = data['dE2000'].values

    DErs = DE2000(ref, sam, dtype = 'lab')
    DEsr = DE2000(sam, ref, dtype = 'lab')

    print('diff DE2000 12<->21:',(DErs - DEsr).T.max())
    print('diff DE2000 vs DE00:',(DEsr[:,0]-DE00).T.max())

