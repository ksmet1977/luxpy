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
Module implementing Ohno (2011) CCT&Duv calculation
===================================================

 :xyz_to_cct_ohno2011(): Calculate cct and Duv from CIE 1931 2° xyz following Ohno (CORM 2011).
 
References:
    1. Ohno, Y. (2011). Calculation of CCT and Duv and Practical Conversion Formulae. 
    CORM 2011 Conference, Gaithersburg, MD, May 3-5, 2011

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import xyz_to_Yuv
from luxpy.utils import np

__all__ =['xyz_to_cct_ohno2011']

_KIJ = np.array([[-3.7146000E-03, 5.6061400E-02, -3.307009E-01, 9.750013E-01, -1.5008606E+00, 1.115559E+00, -1.77348E-01],
                 [-3.2325500E-05, 3.5700160E-04, -1.589747E-03, 3.6196568E-03, -4.3534788E-03, 2.1595434E-03, 5.308409E-04],
                 [-2.6653835E-03, 4.17781315E-02, -2.73172022E-01, 9.53570888E-01, -1.873907584E+00, 1.964980251E+00, -8.58308927E-01],
                 [-2.3524950E+01, 2.7183365E+02, -1.1785121E+03, 2.51170136E+03, -2.7966888E+03, 1.49284136E+03, -2.3275027E+02],
                 [-1.731364909E+06, 2.7482732935E+07, -1.81749963507E+08, 6.40976356945E+08, -1.27141290956E+09, 1.34488160614E+09, -5.926850606E+08],
                 [-9.4353083E+02, 2.10468274E+04, -1.9500061E+05, 9.60532935E+05, -2.65299138E+06, 3.89561742E+06, -2.3758158E+06],
                 [5.0857956E+02, -1.321007E+04, 1.4101538E+05, -7.93406005E+05, 2.48526954E+06, -4.11436958E+06, 2.8151771E+06]])

def xyz_to_cct_ohno2011(xyz):
    """
    Calculate cct and Duv from CIE 1931 2° xyz following Ohno (2011).
    
    Args:
        :xyz:
            | ndarray with CIE 1931 2° X,Y,Z tristimulus values
            
    Returns:
        :cct, duv:
            | ndarrays with correlated color temperatures and distance to blackbody locus in CIE 1960 uv
            
    References:
        1. Ohno, Y. (2011). Calculation of CCT and Duv and Practical Conversion Formulae. 
        CORM 2011 Conference, Gaithersburg, MD, May 3-5, 2011
    """
    uvp = xyz_to_Yuv(xyz)[...,1:]
    uv = uvp*np.array([[1,2/3]])
    Lfp = ((uv[...,0] - 0.292)**2 + (uv[...,1] - 0.24)**2)**0.5
    a = np.arctan((uv[...,1] - 0.24)/(uv[...,0] - 0.292))
    a[a<0] = a[a<0] + np.pi
    Lbb = np.polyval(_KIJ[0,:],a)
    Duv = Lfp - Lbb

    T1 = 1/np.polyval(_KIJ[1,:],a)
    T1[a>=2.54] = 1/np.polyval(_KIJ[2,:],a[a>=2.54])
    dTc1 = np.polyval(_KIJ[3,:],a)*(Lbb + 0.01)/Lfp*Duv/0.01
    dTc1[a>=2.54] = 1/np.polyval(_KIJ[4,:],a[a>=2.54])*(Lbb[a>=2.54] + 0.01)/Lfp[a>=2.54]*Duv[a>=2.54]/0.01
    T2 = T1 - dTc1
    c = np.log10(T2)
    dTc2 = np.polyval(_KIJ[5,:],c)
    dTc2[Duv<0] = np.polyval(_KIJ[6,:],c[Duv<0])*np.abs(Duv[Duv<0]/0.03)**2
    Tfinal = T2 - dTc2
    return Tfinal, Duv

if __name__ == '__main__':
    import luxpy as lx
    xyz = lx.spd_to_xyz(np.vstack((lx._CIE_D65,lx._CIE_A[1:,:])))
    cct,duv = xyz_to_cct_ohno2011(xyz)
    print('cct: ', cct)
    print('Duv: ', duv)