
# -*- coding: utf-8 -*-
# created Dec. 25, 2018
# Author:  K.A.G. Smet

# Timing of various functions:
import numpy as np
import luxpy as lx
import timeit
import pickle

# first get test data
def get_test_spectra(M,N):
    spd=np.vstack((lx.getwlr([360,830,1]),np.ones((N,471))))
    rfl=np.vstack((lx.getwlr([360,830,1]),0.5*np.ones((M,471))))
    return spd, rfl


#-------------------------------------------------------------------------------
# Get test spectra:
MN = (1,2,4,10,100,100)
spd1,rfl1 = get_test_spectra(1,1)
spd2,rfl2 = get_test_spectra(2,2)
spd4,rfl4 = get_test_spectra(4,4)
spd10,rfl10 = get_test_spectra(10,10)
spd100,rfl100 = get_test_spectra(100,100)
spd1000,rfl1000 = get_test_spectra(1000,1000)

spds = (spd1,spd2,spd4,spd10,spd100,spd1000)
rfls = (rfl1,rfl2,spd4,rfl10,rfl100,rfl1000)

fcns = ["spd_to_xyz","xyz_to_Yxy","xyz_to_Yuv","xyz_to_wuv","xyz_to_lab","xyz_to_luv","xyz_to_ipt","xyz_to_Vrb_mb","xyz_to_Ydlep","xyz_to_xyz","xyz_to_lms"]
Nfunctions = 6*(len(fcns)-1*(not ("spd_to_xyz" in fcns))) + 4*(("spd_to_xyz" in fcns))
performance =  np.nan*np.ones((len(spds),len(rfls),Nfunctions))
perf_strs = lx.odict()

Ntimeit = 10000 # same as in Julia
def btime(f,number = 1000):
    return timeit.timeit(f, number = number)/number*1e6 #convert from sec to μs. 


def get_ctf_performance(performance, perf_strs, cspace, fwf, bwf, k, i, j, xyzi, xyzij, xyzwij, xyzw1,takes_wp = False):
    print("dimensions: size(xyzi):", xyzi.shape, ", size(xyzij):", xyzij.shape)
    if takes_wp == False:
        
        out = fwf(xyzi)
        xyzi_ = bwf(out)
        d = np.mean((xyzi-xyzi_)[:])
        print("   xyzi, mean diff: {:1.4f}".format(d))

        if performance is not None:
            performance[i,j,k+1] = btime(lambda: fwf(xyzi), number = Ntimeit)
            performance[i,j,k+2] = btime(lambda: bwf(out), number = Ntimeit)
            perf_strs[k+1] =  cspace+", xyzi, fwf"
            perf_strs[k+2] =  cspace+", xyzi, bwf"
            print("   xyzi, forward: {:1.4f} μs".format(performance[i,j,k+1]))
            print("   xyzi, backward: {:1.4f} μs".format(performance[i,j,k+2]))


        out = fwf(xyzij)
        xyzij_ = bwf(out)
        d = np.mean((xyzij-xyzij_)[:])
        print("   xyzij, mean diff: {:1.4f}".format(d))

        if performance is not None:
            performance[i,j,k+3] = btime(lambda: fwf(xyzij), number = Ntimeit)
            performance[i,j,k+4] = btime(lambda: bwf(out), number = Ntimeit)
            perf_strs[k+3] =  cspace+", xyzij, fwf"
            perf_strs[k+4] =  cspace+", xyzij, bwf"
            print("   xyzij, forward: {:1.4f} μs".format(performance[i,j,k+3]))
            print("   xyzij, backward: {:1.4f} μs".format(performance[i,j,k+4]))        
            
            perf_strs[k+5] =  cspace+", ..."
            perf_strs[k+6] =  cspace+", ..."

    else:
        out = fwf(xyzi, xyzw = xyzw1)
        xyzi_ = bwf(out, xyzw = xyzw1)
        d = np.mean((xyzi-xyzi_)[:])
        print("   xyzi + xyzw1, mean diff: {:1.4f}".format(d))

        if performance is not None:
            performance[i,j,k+1] = btime(lambda: fwf(xyzi, xyzw = xyzw1), number = Ntimeit)
            performance[i,j,k+2] = btime(lambda: bwf(out, xyzw = xyzw1), number = Ntimeit)
            perf_strs[k+1] =  cspace+", xyzi + xyzw1, fwf"
            perf_strs[k+2] =  cspace+", xyzi + xyzw1, bwf"
            print("   xyzi + xyzw1, forward: {:1.4f} μs".format(performance[i,j,k+1]))
            print("   xyzi + xyzw1, backward: {:1.4f} μs".format(performance[i,j,k+2]))

        out = fwf(xyzij, xyzw = xyzw1)
        xyzij_ = bwf(out, xyzw = xyzw1)
        d = np.mean((xyzij-xyzij_)[:])
        print("   xyzij + xyzw1, mean diff: {:1.4f}".format(d))

        if performance is not None:
            performance[i,j,k+3] = btime(lambda: fwf(xyzij, xyzw = xyzw1), number = Ntimeit)
            performance[i,j,k+4] = btime(lambda: bwf(out, xyzw = xyzw1), number = Ntimeit)
            perf_strs[k+3] =  cspace+", xyzij + xyzw1, fwf"
            perf_strs[k+4] =  cspace+", xyzij + xyzw1, bwf"
            print("   xyzij + xyzw1, forward: {:1.4f} μs".format(performance[i,j,k+3]))
            print("   xyzij + xyzw1, backward: {:1.4f} μs".format(performance[i,j,k+4]))
        
        out = fwf(xyzij, xyzw = xyzwij)
        xyzij_ = bwf(out, xyzw = xyzwij)
        d = np.mean((xyzij-xyzij_)[:])
        print("   xyzij + xyzwij, mean diff: {:1.4f}".format(d))

        if performance is not None:
            performance[i,j,k+5] = btime(lambda: fwf(xyzij, xyzw = xyzwij), number = Ntimeit)
            performance[i,j,k+6] = btime(lambda: bwf(out, xyzw = xyzwij), number = Ntimeit)
            perf_strs[k+5] =  cspace+", xyzij + xyzwij, fwf"
            perf_strs[k+6] =  cspace+", xyzij + xyzwij, bwf"
            print("   xyzij + xyzwij, forward: {:1.4f} μs".format(performance[i,j,k+5]))
            print("   xyzij + xyzwij, backward: {:1.4f} μs".format(performance[i,j,k+6]))

    return performance, perf_strs


xyzw1 = lx.spd_to_xyz(spd1)
for (i,spdi) in enumerate(spds):
    for (j,rflj) in enumerate(rfls):
        k = 0
    
        print("----------------------------------------------------")
        print("size(spdi):",spdi.shape, " / size(rflj):", rflj.shape)
        xyzi, xyzwi = lx.spd_to_xyz(spdi,out=2)
        xyzij, xyzwij = lx.spd_to_xyz(spdi, rfl = rflj, out=2)
    
        if "spd_to_xyz" in fcns:
            print("Testing spd_to_xyz:")
            performance[i,j,k+1] = btime(lambda: lx.spd_to_xyz(spdi,out=2), number = Ntimeit)
            perf_strs[k+1] = "spd_to_xyz, spdi"
            performance[i,j,k+2] = btime(lambda: lx.spd_to_xyz(spdi, rfl = rflj,out=2), number = Ntimeit)
            perf_strs[k+2] = "spd_to_xyz, spdi, rflj"
            performance[i,j,k+3] = btime(lambda: lx.spd_to_xyz(spdi, relative = False,out=2), number = Ntimeit)
            perf_strs[k+3] = "spd_to_xyz, spdi, relative=false"
            performance[i,j,k+4] = btime(lambda: lx.spd_to_xyz(spdi, rfl = rflj, relative = False,out=2), number = Ntimeit)
            perf_strs[k+4] = "spd_to_xyz, spdi, rflj, relative=false"
            for kk in np.arange(4):
                print("   case {:1.0f}: {:1.4f} μs", kk+1, performance[i,j,k+kk]*1e6)
            k = k + 4
        
    
        if "xyz_to_Yxy" in fcns:
            print("Testing xyz_to_Yxy:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "Yxy", lx.xyz_to_Yxy, lx.Yxy_to_xyz, k, i, j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6
        
    
        if "xyz_to_Yuv" in fcns:
            print("Testing xyz_to_Yuv:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "Yuv", lx.xyz_to_Yuv, lx.Yuv_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6
        
    
        if "xyz_to_lab" in fcns:
            print("Testing xyz_to_lab:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "lab", lx.xyz_to_lab, lx.lab_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = True)
            k = k + 6
        
    
        if "xyz_to_luv" in fcns:
            print("Testing xyz_to_luv:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "luv", lx.xyz_to_luv, lx.luv_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = True)
            k = k + 6
        
    
        if "xyz_to_wuv" in fcns:
            print("Testing xyz_to_wuv:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "wuv", lx.xyz_to_wuv, lx.wuv_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = True)
            k = k + 6
        
    
        if "xyz_to_ipt" in fcns:
            print("Testing xyz_to_ipt:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "ipt", lx.xyz_to_ipt, lx.ipt_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = True)
            k = k + 6
        
    
        if "xyz_to_Vrb_mb" in fcns:
            print("Testing xyz_to_Vrb_mb:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "Vrb_mb", lx.xyz_to_Vrb_mb, lx.Vrb_mb_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6
        
    
        if "xyz_to_Ydlep" in fcns:
            print("Testing xyz_to_Ydlep:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "Ydlep", lx.xyz_to_Ydlep, lx.Ydlep_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6
        
    
        if "xyz_to_xyz" in fcns:
            print("Testing xyz_to_xyz:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "xyz", lx.xyz_to_xyz, lx.xyz_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6
        
    
        if "xyz_to_lms" in fcns:
            print("Testing xyz_to_lms:")
            performance, perf_strs = get_ctf_performance(performance, perf_strs, "lms", lx.xyz_to_lms, lx.lms_to_xyz, k, i,j, xyzi, xyzij, xyzwij, xyzw1, takes_wp = False)
            k = k + 6

#------------------------------------------------------------------------------       
        
def analyze_performance(write_to_xls=False):
    # Python: store in dict:
    n = list(perf_strs.keys())
    performance_=performance[:,:,n]
    perf_dict_py=lx.odict()
    for i in range(len(perf_strs.values())):
        perf_dict_py[list(perf_strs.values())[i]] = performance_[:,:,i]
        
    # Julia: get data from tmp folder:
    tmppath = "D:/Documents/JULIALANG/Julia-0.7.0/jlux/tmp/"
    f = lambda x: lx.getdata(tmppath+list(perf_dict_py.keys())[x]+'.dat')
    perf_dict_jl=lx.odict()
    for i in range(len(perf_strs.values())):
        perf_dict_jl[list(perf_strs.values())[i]] = f(i)
    
    # Calculate ratio of py/jl:
    perf_dict_pydivjl=lx.odict()
    for i in range(len(perf_strs.values())):
        key = list(perf_strs.values())[i]
        perf_dict_pydivjl[key] = perf_dict_py[key]/perf_dict_jl[key]
        
    with open('perf_dicts_py_jl.pickle', 'wb') as handle:
        pickle.dump((perf_dict_py,perf_dict_jl,perf_dict_pydivjl), handle, protocol=pickle.HIGHEST_PROTOCOL)
     
        
    getkv = lambda i: (list(perf_dict_pydivjl.keys())[i],perf_dict_pydivjl[list(perf_dict_pydivjl.keys())[i]])
    
    # write to excel file
    spdsize = ["{:1.0f}".format(MNi) for MNi in MN]
    for i in range(len(perf_dict_py.keys())):
        lx.write_to_excel("performance_comp_py1e2_jl1e4.xlsx",lx.pd.DataFrame(getkv(i)[1],columns = spdsize,index=spdsize),getkv(i)[0])
      
    return perf_dict_pydivjl, getkv

# Perform analysis:
perf_dict_pydivjl, getkv = analyze_performance(write_to_xls=False)