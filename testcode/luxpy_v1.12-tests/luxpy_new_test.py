# -*- coding: utf-8 -*-
#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

#-----other imports-------------------------------------------------------
import numpy as np

#-----function definitions -----------------------------------------------

##------------------------------------------------------
if __name__ == '__main__':

    
    # get some data:
    spds = lx.utils.getdata(root+"luox/calculation_results/sample.csv", header='infer').T
    cmf2, K2 = lx.utils.getdata(root+"/data/cmfs/ciexyz_1931_2.dat").T, 683.002
    cmf2_lu = lx.utils.getdata(root+"/luox/data/ciexyz31_1.csv").T
    cmf10_lu = lx.utils.getdata(root+"/luox/data/ciexyz64_1.csv").T
    rfl = lx._RFL['cri']['cie-13.3-1995']['8'].copy()

    # test spd_to_xyz
    xyz2, xyzw2 = lx.spd_to_xyz(spds, cieobs=cmf2, K = K2, relative = True, out = 2)
    
    # for testing:
    cmf, K = lx.cie_interp(cmf2,spds[0], "cmf")[2:], K2
    spds2 = np.vstack((spds,spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:],spds[1:]))
    spds2_= spds[:2]
    s = spds2[2:]
    dl = lx.getwld(spds2[0])
    rfl = np.ones((2,spds.shape[-1]))
    rfl2 = np.vstack((spds2[0],rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl))
    rfl2_ = np.vstack((spds2[0],rfl[0]))
    
    # # speedtests
    import timeit
    ts1 = np.array([timeit.timeit('lx.spd_to_xyz(spds2,cieobs=cmf2,rfl=None)', setup='from __main__ import (spds2,spds2_,cmf2,rfl2,rfl2_,lx)', number=10) for i in range(1000)])/10
    ts2 = np.array([timeit.timeit('lx.spd_to_xyz(spds2,cieobs=cmf2,rfl=rfl2)', setup='from __main__ import (spds2,spds2_,cmf2,rfl2,rfl2_,lx)', number=10) for i in range(1000)])/10
    print('med(ts1), med(ts2): ',np.median(ts1),np.median(ts2))

    print("xyzw2:\n",xyzw2)

   


