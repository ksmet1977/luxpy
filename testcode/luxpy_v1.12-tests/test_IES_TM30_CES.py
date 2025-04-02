# -*- coding: utf-8 -*-
# test_IES_TM30_CES

#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

import numpy as np
import matplotlib.pyplot as plt

#----function definitions-------------------------------------------------

fDElab = lambda a,b : (((a - b)**2).sum(axis = -1)**0.5)
fDEl = lambda a,b : (((a - b)**2)[...,0]**0.5)
fDEab = lambda a,b : (((a - b)**2)[...,1:].sum(axis=-1)**0.5)

fDij = lambda a,b,fD: np.array([[fD(a[i],b[j]) if (i>j) else np.nan for j in range(a.shape[0])] for i in range(b.shape[0])])

#------------------------
def spectral_uniformity(rfls, dl = None):
    if rfls.ndim == 2:
        wl = rfls[0]
    else:
        wl = rfls[0,:,0]
        
    if dl is not None:
        wl_new = lx.getwlr([wl[0],wl[-1],dl])
        if rfls.ndim == 2:
            rfls = lx.cie_interp(rfls, wl_new, kind = 'rfl', negative_values_allowed = True)
        else:
            rfls = np.transpose(np.array([lx.cie_interp(rfls[...,i], wl_new, kind = 'rfl', negative_values_allowed = True) for i in range(rfls.shape[-1])]),(1,2,0))
        wl = wl_new 
        
    d1 = np.gradient(rfls[1:], axis = 1)
    d2 = np.gradient(d1, axis = 1)
    su_i = (3*d1**2+(100*d2)**2)**0.5
    su = su_i.sum(axis = 0)/su_i.shape[0]
    if su.ndim == 2:
        su = su.sum(axis = -1)/su.shape[-1]
    return np.vstack((wl,su))


#==================================================================================
if __name__ == '__main__':

    run_calculations = False

    CEStype = 4880

    cieobs = '1964_10'
    
    if run_calculations:
        
        spd = lx._CIE_E
        spd = spd[:,(spd[0]>=380) & (spd[0]<=780)]
        rfl = lx._RFL['cri']['ies-tm30'][f'{CEStype}']['1nm']
        #rfl = rfl[:,(rfl[0]>=400) & (rfl[0]<=700)]

        xyz, xyzw = lx.spd_to_xyz(spd, cieobs = cieobs, rfl = rfl, out = 2)
        xyz = xyz[:,0,:]
        lab = lx.xyz_to_jab_cam02ucs(xyz, xyzw = xyzw)
      
        DElab = fDij(lab,lab,fDElab)
        RMSE = fDij(rfl[1:],rfl[1:], fDElab)
        np.save(f'IESTM30_{CEStype}_test.npy',{'data':{'spd':spd,'rfl':rfl,'xyz':xyz,'xyzw':xyzw,'lab':lab,'DElab':DElab,'RMSE':RMSE}})
    else:
        data = np.load(f'IESTM30_{CEStype}_test.npy', allow_pickle=True)[()]['data']
        DElab, RMSE, lab, rfl, spd, xyz, xyzw = [data[k] for k in sorted(list(data.keys()))]

    print('Min DElab: ', np.nanmin(DElab))
    print('Min RMSE: ', np.nanmin(RMSE))

    DElab_min = 0.0 # 0.25 used to determine 2685, 0.0 used for 2696 
    if np.nanmin(DElab) <= DElab_min: # indicative of same rfls
                    
        # Get unique rfls if there are any doubles:
        #------------------------------------------
        # 1. Get DE <= DElab_min rfl comparions -> these are doubles per column (or row):
        n = xyz.shape[0]
        DElab_ = DElab.copy()
        DElab_[np.isnan(DElab_)] = 0.0
        DElab__ = (DElab_ + DElab_.T) + np.diag(np.ones((n,))*np.nan)
        a = np.argwhere(DElab__ <= DElab_min)
        u = np.unique(a[:,0]) 
        same_rfl_idxs = [np.unique(a[a[:,0] == u[i],:]) for i in range(u.shape[0])] # a list of same rfls (but rows are not necessarily unique (yet)!)
        Ns = np.array([np.shape(x) for x in same_rfl_idxs]).ravel() # number of same rfls per group
        same_rfl_groups_idxs = np.unique(np.array([np.hstack((-1*np.ones((Ns.max()-Ns[i],)),same_rfl_idxs[i])) if ((Ns.max()-Ns[i])>0) else same_rfl_idxs[i] for i in range(len(same_rfl_idxs))]),axis=0) # create a stackable array with groups of same rfls (fill with -1, and then take unqiue to ensure unique groups!)
        same_rfls_group_examples_idxs = np.array([same_rfl_groups_idxs[i,same_rfl_groups_idxs[i,:]>=0].astype(int)[0] for i in range(same_rfl_groups_idxs.shape[0])]) # take first example rfl of each group as new standard rfl
        unique_rfls_idxs_original = np.setdiff1d(np.arange(n),np.unique(same_rfl_groups_idxs)[1:]) # find rfls in the original set that don't have any copies
        unique_rfls_idxs = np.unique(np.hstack((same_rfls_group_examples_idxs, unique_rfls_idxs_original))).astype(int) # join original uniques with new unique standard rfls to create new set
        unique_rfls = np.vstack((rfl[0],rfl[1:][unique_rfls_idxs])) # create new set of actual unique spectral reflectance functions
        doubled_rfls = np.vstack((rfl[0],rfl[1:][same_rfls_group_examples_idxs])) # create a set of doubled spectral reflectance functions
        
        i,j = same_rfl_groups_idxs[0,same_rfl_groups_idxs[0]>-1].astype(int) # take a single double rfl set to plot as example
        print(f'\nMin check: {DElab[j,i]:.2} --> Some rfls have doubles !!!!')
        print(f'\nNumber of groups of identical rfls in old set: {doubled_rfls.shape[0]-1:.0f}')
        print(f'Number of identical rfls in largest group: {same_rfl_groups_idxs.shape[1]:.0f}')
        print(f'Total number of unique rfls in old set: {unique_rfls_idxs_original.shape[0]} ({100*(unique_rfls_idxs_original.shape[0])/n:.1f}%)')
        print(f'Total number of non-unique rfls in old set: {(same_rfl_groups_idxs>-1).sum()} ({100*(same_rfl_groups_idxs>-1).sum()/n:.1f}%)')
        print(f'Total number of unique rfls in new set: {unique_rfls.shape[0]-1} ({100*(unique_rfls.shape[0]-1)/n:.1f}%)')
        
        rfl = unique_rfls.copy()
        xyz, xyzw = lx.spd_to_xyz(spd, cieobs = cieobs, rfl = rfl, out = 2)
        xyz = xyz[:,0,:]
        n = xyz.shape[0]
        lab = lx.xyz_to_jab_cam02ucs(xyz, xyzw = xyzw)
        DElab = fDij(lab,lab,fDElab)
        print('Min DElab: ', np.nanmin(DElab))
        i,j = np.unravel_index(np.nanargmin(DElab), (DElab.shape[0],DElab.shape[0]))
        print(f'\nMin check: {DElab[i,j]:.2} --> No copies of rfls present !!!!')


    else:
        procentage_same_rfls = 0.0
        i,j = np.unravel_index(np.nanargmin(DElab), (DElab.shape[0],DElab.shape[0]))
        print(f'\nMin check: {DElab[i,j]:.2} --> No copies of rfls present !!!!')


    plt.figure()
    plt.plot(lab[:,1],lab[:,2],'b.')
    plt.plot(lab[i,1],lab[i,2],'r^')
    plt.plot(lab[j,1],lab[j,2],'gv')
    plt.xlabel('a*')
    plt.ylabel('b*')

    plt.figure()
    plt.plot(rfl[0],rfl[i+1],'r-')
    plt.plot(rfl[0],rfl[j+1],'g--')
    plt.xlabel('Wavelengths (nm)');

    # Check spectral uniformity:
    dl = 5
    rfl4880 = lx._RFL['cri']['ies-tm30']['4880']['1nm']
    rfl99 = lx._RFL['cri']['ies-tm30']['99']['1nm']
    rfl2696 = lx._RFL['cri']['ies-tm30']['2696']['1nm']
    suCIE8 = spectral_uniformity(lx._RFL['cri']['cie-13.3-1995']['8'], dl = dl)
    su4880 = spectral_uniformity(rfl4880, dl = dl)
    su99 = spectral_uniformity(rfl99, dl = dl)
    su2696 = spectral_uniformity(rfl2696, dl = dl)
    su = spectral_uniformity(rfl, dl = dl)
    plt.figure()
    plt.plot(suCIE8[0],suCIE8[1],'b-', label = 'CIE8')
    plt.plot(su4880[0],su4880[1],'r-', label = '4880')
    plt.plot(su99[0],su99[1],'g--', label = '99')
    plt.plot(su2696[0],su2696[1],'k--', label = '2696')
    plt.plot(su[0],su[1],'k:', label = f'{CEStype}')
    plt.xlabel('Wavelengths (nm)');
    plt.legend()

    # Check Rf values (same scaling factor):
    spds = lx._IESTM3015['S']['data']
    Rf99 = lx.cri.spd_to_cri(spds, cri_type = 'ies-tm30')
    Rf4880 = lx.cri.spd_to_cri(spds, cri_type = 'ies-tm30', sampleset = rfl4880)
    Rf2696 = lx.cri.spd_to_cri(spds, cri_type = 'ies-tm30', sampleset = rfl2696)
    Rf = lx.cri.spd_to_cri(spds, cri_type = 'ies-tm30', sampleset = rfl)

    print('percentile([50, 95, 99, 100]) for 99 vs 4880: ', np.percentile(np.abs(Rf99 - Rf4880), [25,95,99,100]))
    print(f'percentile([50, 95, 99, 100]) for 4880 vs {CEStype}: ', np.percentile(np.abs(Rf - Rf4880), [25,95,99,100]))
    print(f'percentile([50, 95, 99, 100]) for 99 vs {CEStype}: ', np.percentile(np.abs(Rf99 - Rf), [25,95,99,100]))
    print(f'percentile([50, 95, 99, 100]) for 99 vs 2696: ', np.percentile(np.abs(Rf99 - Rf2696), [25,95,99,100]))
    print(f'percentile([50, 95, 99, 100]) for {CEStype} vs 2696: ', np.percentile(np.abs(Rf - Rf2696), [25,95,99,100]))
    
    plt.figure()
    plt.plot(Rf99.T,Rf.T,'b.',label = f'99 vs {CEStype}');
    plt.plot(Rf99.T,Rf2696.T,'c.', label = f'99 vs 2696');
    plt.plot(Rf99.T,Rf4880.T,'r.', label = '99 vs 4880');
    plt.plot(Rf4880.T,Rf.T,'g.', label = f'4880 vs {CEStype}');

    plt.legend()

