# -*- coding: utf-8 -*-
"""
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np
import matplotlib.pyplot as plt
import luxpy as lx

from tc1_97.compute import compute_tabulated, VisualData, LMS_quantal, LMS_energy
import luxpy_individual_observer_cmf_model as ic


data = ic.init(wl = None, dsrc_std = 'matlab', dsrc_lms_odens = 'cietc197',
               lms_to_xyz_method = 'cietc197',
               use_sign_figs = True, use_my_round = True, use_chop = True, out = 1)


def get_out_197(fs, age, wlr3 = [390,830,1]):
    res, _ = compute_tabulated(fs, age, wlr3[0], wlr3[1], wlr3[2])
    keys_197 = ['LMS','LMS_base','XYZ','XYZ_N','trans_mat','trans_mat_N']
    out_197 = {}
    for key in keys_197:
        if (key == 'LMS') | (key == 'LMS_base') | (key == 'XYZ') | (key == 'XYZ_N'):
            out_197[key] = res[key].T.copy()
        else:
            out_197[key] = res[key].copy()
    return out_197

def plot_lms(out1, out2):
    fig, axs = plt.subplots(nrows = 2, ncols = 2)
    axs = axs.ravel()
    axs[0].plot(out1[:1,:].T,out1[1:,:].T,'b-')
    axs[0].plot(out2[:1,:].T,out2[1:,:].T,'r--')
    colors = ['r','g','b']
    for i in range(3):
        axs[i+1].plot(out1[:1,:].T,(out1[i+1:i+2,:] - out2[i+1:i+2,:]).T, colors[i]+'-')
        d = (out1[i+1,:] - out2[i+1,:])
        print('Max diff: {:1.4e} at {:1.1f}nm'.format(np.sign(d[np.abs(d).argmax()])*np.abs(d).max(),out1[0,np.abs(d).argmax()]))

def test_equality_input():
    # get TC197 data: 
    vd_197 = VisualData()
    in_197 ={'LMSa':vd_197.absorbance.T.copy(),
             'rmd':vd_197.macula_rel.T.copy(),
             'docul2':vd_197.docul2.T.copy(),
             'docul1_fine':vd_197.docul1_fine.T.copy(),
             'docul2_fine':vd_197.docul2_fine.T.copy(),
             'xyz1931':vd_197.XYZ31.T.copy(),
             'xyz1964':vd_197.XYZ64.T.copy()}
    
    # Get luxpy data:
    data = ic.init(wl = None, dsrc_std = 'matlab', dsrc_lms_odens = 'cietc197',
                   lms_to_xyz_method = 'cietc197',
                   use_sign_figs = True, use_my_round = True, use_chop = True, out = 1)
    in_lx = data['odata']
    wls = in_lx['wls']
    in_lx['docul1_fine'] = in_lx['docul'][1,:]
    in_lx['docul2_fine'] = in_lx['docul'][2,:]
    in_lx['xyz1931'] = lx.cie_interp(lx._CMF['1931_2']['bar'],wls,kind='cmf')
    in_lx['xyz1964'] = lx.cie_interp(lx._CMF['1964_10']['bar'],wls,kind='cmf')
    return in_197, in_lx


def test_equality_output(fs, age, wlr3 = [390,830,1]):
    # CIE1-97:
    out_197 = get_out_197(fs, age, wlr3 = wlr3)
    
    # luxpy:
    data = ic.init(wl = None, dsrc_std = 'matlab', dsrc_lms_odens = 'cietc197',
                   lms_to_xyz_method = 'cietc197',
                   use_sign_figs = True, use_my_round = True, use_chop = True, out = 1)
    LMS,XYZ,M = ic.compute_cmfs(fieldsize = fs, age = age, wl = wlr3,
                                out = 'LMS,XYZ,M', 
                                base = False, lms_to_xyz_method = 'cietc197')
    out_lx = {'LMS':LMS.copy(), 'XYZ':XYZ.copy(),'trans_mat':M.copy()}
    
    
    return out_197, out_lx
    
    
if __name__ == '__main__':
    
    fs = 10
    age = 40
    wlr3 = [390,830,5]
    
    data = ic.init(wl = None, dsrc_std = 'matlab', dsrc_lms_odens = 'cietc197',
               lms_to_xyz_method = 'cietc197',
               use_sign_figs = True, use_my_round = True, use_chop = True, out = 1)

    
    out_197, out_lx = test_equality_output(fs, age, wlr3 = wlr3)
    plot_lms(out_197['XYZ_N'],out_lx['XYZ'])
    plot_lms(out_197['LMS'],out_lx['LMS'])
    print('M_197:\n',out_197['trans_mat'])
    print('M_lx:\n',out_lx['trans_mat'])
    print('M-normalization tc197:', np.dot(out_197['trans_mat_N'],np.array([[100,100,100]]).T).T)
    print('M-normalization luxpy:', np.dot(out_lx['trans_mat'], np.array([[100,100,100]]).T).T)