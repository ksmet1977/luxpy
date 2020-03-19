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
Module for Modified Friele discrimination ellipses
==================================================
 :get_gij_fmc(): Get gij matrices describing the discrimination ellipses for Yxy using FMC-1 or FMC-2.

 :get_fmc_discrimination_ellipse(): Get n-step discrimination ellipse(s) in v-format (R,r, xc, yc, theta) for Yxy using FMC-1 or FMC-2.


References:
    1. Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4), p.537-541
    2. Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1), p.118-122
 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import (np, plt, math, Yxy_to_xyz, plotSL, plot_chromaticity_diagram_colors, plotellipse)

_M_XYZ_TO_PQS = np.array([[0.724, 0.382, -0.098],[-0.48,1.37,0.1276],[0,0,0.686]])

__all__ = ['get_gij_fmc','get_fmc_discrimination_ellipse']

def _dot_M_xyz(M,xyz):
    """
    Perform matrix multiplication between M and xyz (M*xyz) using einsum.
    
    Args:
        :xyz:
            | 2D or 3D ndarray
        :M:
            | 2D or 3D ndarray
            | if 2D: use same matrix M for each xyz, 
            | if 3D: use M[i] for xyz[i,:] (2D xyz) or xyz[:,i,:] (3D xyz)
            
    Returns:
        :M*xyz:
            | ndarray with same shape as xyz containing dot product of M and xyz.
    """
    # convert xyz to ...:
    if np.ndim(M)==2:
        if len(xyz.shape) == 3:
            return np.einsum('ij,klj->kli', M, xyz)
        else:
            return np.einsum('ij,lj->li', M, xyz)
    else:
        if len(xyz.shape) == 3: # second dim of x must match dim of 1st of M and 1st dim of xyzw
            return np.concatenate([np.einsum('ij,klj->kli', M[i], xyz[:,i:i+1,:]) for i in range(M.shape[0])],axis=1)
        else: # first dim of xyz must match dim of 1st of M and 1st dim of xyzw
            return np.concatenate([np.einsum('ij,lj->li', M[i], xyz[i:i+1,:]) for i in range(M.shape[0])],axis=0)
      

def _xyz_to_pqs(xyz):
    """
    Calculate pqs from xyz.
    """
    # calculate pqs from xyz:
    return _dot_M_xyz(_M_XYZ_TO_PQS,xyz)

def _transpose_02(C):
    if C.ndim == 3:
        C = np.transpose(C,(2,0,1))
    else:
        C = C[None,...]
    return C


def _cij_to_gij(xyz,C):
    """ Convert from matrix elements describing the discrimination ellipses from Cij (XYZ) to gij (Yxy)"""
    SIG = xyz[...,0] + xyz[...,1] + xyz[...,2]
    M1 = np.array([SIG, -SIG*xyz[...,0]/xyz[...,1], xyz[...,0]/xyz[...,1]])
    M2 = np.array([np.zeros_like(SIG), np.zeros_like(SIG), np.ones_like(SIG)])
    M3 = np.array([-SIG, -SIG*(xyz[...,1] + xyz[...,2])/xyz[...,1], xyz[...,2]/xyz[...,1]])
    M = np.array((M1,M2,M3))
    
    M = _transpose_02(M) # move stimulus dimension to axis = 0
    
    C = _transpose_02(C) # move stimulus dimension to axis = 0
    
    # convert Cij (XYZ) to gij' (xyY):
    AM = np.einsum('ij,kjl->kil', _M_XYZ_TO_PQS, M)
    CAM = np.einsum('kij,kjl->kil', C, AM) 
#    ATCAM = np.einsum('ij,kjl->kil', _M_XYZ_TO_PQS.T, CAM)
#    gij = np.einsum('kij,kjl->kil', np.transpose(M,(0,2,1)), ATCAM) # gij = M.T*A.T*C**A*M = (AM).T*C*A*M
    gij = np.einsum('kij,kjl->kil', np.transpose(AM,(0,2,1)), CAM) # gij = M.T*A.T*C**A*M = (AM).T*C*A*M

    # convert gij' (xyY) to gij (Yxy):
    gij = np.roll(np.roll(gij,1,axis=2),1,axis=1)
    
    return gij


def _get_gij_fmc_1(xyz, cspace = 'Yxy'):
    """
    Get gij matrices describing the discrimination ellipses for xyz using FMC-1.
    
    Reference:
        Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4), p.537-541
    """
    # Convert xyz to pqs coordinates:
    pqs = _xyz_to_pqs(xyz)
    
    # get FMC-1 Cij matrix (for X,Y,Z):
    D2 = (pqs[...,0]**2 + pqs[...,1]**2)
    b2 = 3.098e-4*(pqs[...,2]**2 + 0.2015*xyz[...,1]**2)
    A2 = 57780*(1 + 2.73*((pqs[...,0]*pqs[...,1])**2)/(pqs[...,0]**4 + pqs[...,1]**4))
    C11 = (A2*(0.0778*pqs[...,0]**2 + pqs[...,1]**2) + ((pqs[...,0]*pqs[...,2])**2)/b2)/D2**2
    C12 = (-0.9222*A2*pqs[...,0]*pqs[...,1] + (pqs[...,0]*pqs[...,1]*pqs[...,2]**2)/b2)/D2**2
    C22 = (A2*(pqs[...,0]**2 + 0.0778*pqs[...,1]**2) + ((pqs[...,1]*pqs[...,2])**2)/b2)/D2**2
    C13 = -pqs[...,0]*pqs[...,2]/(b2*D2) # or -PQ/b2*D2 ??
    C33 = 1/b2
    C23 = pqs[...,1]*C13/pqs[...,0]
    C = np.array([[C11, C12, C13],[C12, C22, C23], [C13, C23, C33]])
    if cspace == 'Yxy':
        return _cij_to_gij(xyz,C)
    else:
        return C
    
def _get_gij_fmc_2(xyz, cspace = 'Yxy'):
    """
    Get gij matrices describing the discrimination ellipses for xyz using FMC-1.
    
    Reference:
        Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1), p.118-122
    """
    # Convert xyz to pqs coordinates:
    pqs = _xyz_to_pqs(xyz)
    
    # get FMC-2 Cij matrix (for X,Y,Z):
    D = (pqs[...,0]**2 + pqs[...,1]**2)**0.5
    a = (17.3e-6*D**2/(1 + 2.73*((pqs[...,0]*pqs[...,1])**2)/(pqs[...,0]**4 + pqs[...,1]**4)))**0.5
    b = (3.098e-4*(pqs[...,2]**2 + 0.2015*xyz[...,1]**2))**0.5
    K1 = 0.55669 + xyz[...,1]*(0.049434 + xyz[...,1]*(-0.82575e-3 + xyz[...,1]*(0.79172e-5 - 0.30087e-7*xyz[...,1])))
    K2 = 0.17548 + xyz[...,1]*(0.027556 + xyz[...,1]*(-0.57262e-3 + xyz[...,1]*(0.63893e-5 - 0.26731e-7*xyz[...,1]))) 
    e1 = K1*pqs[...,2]/(b*D**2)
    e2 = K1/b
    e3 = 0.279*K2/(a*D)
    e4 = K1/(a*D)
    C11 = (e1**2 + e3**2)*pqs[...,0]**2 + e4**2*pqs[...,1]**2
    C12 = (e1**2 + e3**2 - e4**2)*pqs[...,0]*pqs[...,1]
    C22 = (e1**2 + e3**2)*pqs[...,1]**2 + e4**2*pqs[...,0]**2
    C13 = -e1*e2*pqs[...,0]
    C23 = -e1*e2*pqs[...,1]
    C33 = e2**2
    C = np.array([[C11, C12, C13],[C12, C22, C23], [C13, C23, C33]])
    if cspace == 'Yxy':
        return _cij_to_gij(xyz,C)
    else:
        return C
    

def get_gij_fmc(Yxy, etype = 'fmc2', ellipsoid = True, Y = None, cspace = 'Yxy'):
    """
    Get gij matrices describing the discrimination ellipses/ellipsoids for Yxy or xyz using FMC-1 or FMC-2.
    
    Args:
        :Yxy:
            | 2D ndarray with [Y,]x,y coordinate centers. 
            | If Yxy.shape[-1]==2: Y is added using the value from the Y-input argument.
        :etype:
            | 'fmc2', optional
            | Type of FMC color discrimination equations to use (see references below).
            | options: 'fmc1', fmc2'
        :Y:
            | None, optional
            | Only affects FMC-2 (see note below).
            | If not None: Y = 10.69 and overrides values in Yxy. 
        :ellipsoid:
            | True, optional
            | If True: return ellipsoids, else return ellipses (only if cspace == 'Yxy')!
        :cspace:
            | 'Yxy', optional
            | Return coefficients for Yxy-ellipses/ellipsoids ('Yxy') or XYZ ellipsoids ('xyz')
    
    Note:
        1. FMC-2 is almost identical to FMC-1 is Y = 10.69!; see [2]
    
    References:
        1. Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4), p.537-541
        2. Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1), p.118-122
    """
    if Yxy.shape[-1] == 2:
        Yxy = np.hstack((100*np.ones((Yxy.shape[0],1)),Yxy))
    if Y is not None:
        Yxy[...,0] = Y
    xyz = Yxy_to_xyz(Yxy)
    if etype == 'fmc2':
        gij = _get_gij_fmc_2(xyz, cspace = cspace)
    else:
        gij = _get_gij_fmc_1(xyz, cspace = cspace)
    if ellipsoid == True:
        return gij
    else:
        if cspace.lower()=='xyz':
            return gij
        else:
            return gij[:,1:,1:]

def get_fmc_discrimination_ellipse(Yxy = np.array([[100,1/3,1/3]]), etype = 'fmc2', Y = None, nsteps = 10):
    """
    Get discrimination ellipse(s) in v-format (R,r, xc, yc, theta) for Yxy using FMC-1 or FMC-2.
    
    Args:
        :Yxy:
            | 2D ndarray with [Y,]x,y coordinate centers. 
            | If Yxy.shape[-1]==2: Y is added using the value from the Y-input argument.
        :etype:
            | 'fmc2', optional
            | Type of FMC color discrimination equations to use (see references below).
            | options: 'fmc1', fmc2'
        :Y:
            | None, optional
            | Only affects FMC-2 (see note below).
            | If not None: Y = 10.69 and overrides values in Yxy. 
        :nsteps:
            | 10, optional
            | Set multiplication factor for ellipses 
            | (nsteps=1 corresponds to approximately 1 MacAdam step, 
            | for FMC-2, Y also has to be 10.69, see note below).
    
    Note:
        1. FMC-2 is almost identical to FMC-1 is Y = 10.69!; see [2]
    
    References:
        1. Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4), p.537-541
        2. Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1), p.118-122
    """
    # Get matrix elements for discrimination ellipses at Yxy:
    gij = get_gij_fmc(Yxy, etype = etype, ellipsoid = False, Y = Y, cspace = 'Yxy')
    
    # Convert cik (gij) to v-format (R,r,xc,yc, theta):
    if Yxy.shape[-1]==2:
        xyc = Yxy
    else:
        xyc = Yxy[...,1:]
    v = math.cik_to_v(gij, xyc = xyc)
    
    # convert to desired number of MacAdam-steps:
    v[:,0:2] = v[:,0:2]*nsteps
    
    return v
    
if __name__ == '__main__':
    from macadamellipses import get_macadam_ellipse
    Yxy1 = np.array([[100,1/3,1/3]])
    Yxy2 = np.array([[100,1/3, 1/3],[50,1/3,1/3]])
    gij_11 = get_gij_fmc(Yxy1,etype = 'fmc1', ellipsoid=False)
    gij_12 = get_gij_fmc(Yxy2,etype = 'fmc1', ellipsoid=False)
    
    # Get MacAdam ellipses:
    v_mac = get_macadam_ellipse(xy = None)
    xys = v_mac[:,2:4]
    
    # Get discrimination ellipses for MacAdam centers using macadam, FMC-1 & FMC-2:
    v_mac_1 = get_fmc_discrimination_ellipse(Yxy = xys, etype = 'fmc1', nsteps = 10)
    v_mac_2 = get_fmc_discrimination_ellipse(Yxy = xys, etype = 'fmc2', nsteps = 10, Y = 10.69)
    
    # Plot results:
    cspace = 'Yxy'
    #axh = plot_chromaticity_diagram_colors(cspace = cspace)
    axh = plotSL(cspace = cspace, cieobs = '1931_2', show = False, diagram_colors = False)
    axh = plotellipse(v_mac, show = True, axh = axh, cspace_in = None, cspace_out = cspace,plot_center = False, center_color = 'r', out = 'axh', line_style = ':', line_color ='r',line_width = 1.5)
    plotellipse(v_mac_1, show = True, axh = axh, cspace_in = None, cspace_out = cspace,line_color = 'b', line_style = ':', plot_center = True, center_color = 'k')
    plotellipse(v_mac_2, show = True, axh = axh, cspace_in = None, cspace_out = cspace,line_color = 'g', line_style = '--', plot_center = True, center_color = 'k')

    axh.set_xlim([0,0.75])
    axh.set_ylim([0,0.85])
    #plt.plot(Yxys[:,1],Yxys[:,2],'ro')
    

