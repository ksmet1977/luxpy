# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:11:05 2020

@author: u0032318
"""
import numpy as np

from luxpy import cam
from luxpy.color.cri.utils.helpers import gamut_slicer
from luxpy.color.cri.iestm30.ies_tm30_metrics import spd_to_ies_tm30_metrics as s
from luxpy.color.cri2.utils.helpers import _get_hue_bin_data
from luxpy.color.cri2.iestm30.ies_tm30_metrics import spd_to_ies_tm30_metrics as s2
import luxpy as lx

start_hue = 0
nhbins = 16
normalize_gamut = True
normalized_chroma_ref = 100

SPDs = lx._IESTM3015['S']['data']
SPD = SPDs[:2]
SPD= lx._CIE_F4

data =   s(SPD, cri_type = 'ies-tm30', hbins = 16, start_hue = 0, scalef = 100)
data2 = s2(SPD, cri_type = 'ies-tm30', hbins = 16, start_hue = 0, scalef = 100)

out = 'jab_test,jab_ref'
out2_s = 'jabt,jabr'.split(',')
out = gamut_slicer(data['jabt_'],data['jabr_'],start_hue=0,nhbins=16,normalize_gamut=True,out=out)
out2_ = _get_hue_bin_data(data2['hue_bin_data']['jabt'],data2['hue_bin_data']['jabr'],start_hue=0,nhbins=16)
out2 = (out2_[out2_s[0]],out2_[out2_s[1]])

def positive_arctan(x,y, htype = 'deg'):

    if htype == 'deg':
        r2d = 180.0/np.pi
        h360 = 360.0
    else:
        r2d = 1.0
        h360 = 2.0*np.pi
    h = np.atleast_1d((np.arctan2(y,x)*r2d))
    h[np.where(h<0)] = h[np.where(h<0)] + h360
    return h


jabt,jabr=out2
# jabt2,jabr2=out2
# ii=0
# ht = positive_arctan(jabt[:,ii,1],jabt[:,ii,2], htype='rad')
# ht2 = positive_arctan(jabt2[...,1], jabt2[...,2], htype = 'rad')
# ht = np.arctan2(jabt[:,ii,2],jabt[:,ii,1])
# ht2 = np.arctan2(jabt2[...,2], jabt2[...,1])

#-----------------------------------------------------------
dh = 360/nhbins 
hue_bin_edges = np.arange(start_hue, 360 + 1, dh)*np.pi/180

# get hues of jabt, jabr:
ht = cam.hue_angle(jabt[...,1], jabt[...,2], htype = 'rad')
hr = cam.hue_angle(jabr[...,1], jabr[...,2], htype = 'rad')

# Get chroma of jabt, jabr:
Ct = ((jabt[...,1]**2 + jabt[...,2]**2))**0.5
Cr = ((jabr[...,1]**2 + jabr[...,2]**2))**0.5


# Calculate DEi between jabt, jabr:
DEi = ((jabt - jabr)**2).sum(axis = -1, keepdims = True)**0.5

# calculate hue-bin averages for jabt, jabr:
jabt_hj = np.ones((nhbins,ht.shape[1],3))*np.nan
jabr_hj = np.ones((nhbins,hr.shape[1],3))*np.nan
DE_hj = np.ones((nhbins,hr.shape[1]))*np.nan
ht_idx = np.ones_like((ht))*np.nan
hr_idx = np.ones_like((hr))*np.nan
n = hr_idx.shape[-1]

for j in range(nhbins):
    cndt_hj = (ht>=hue_bin_edges[j]) & (ht<hue_bin_edges[j+1])
    cndr_hj = (hr>=hue_bin_edges[j]) & (hr<hue_bin_edges[j+1])

    ht_idx[cndt_hj] = j # store hue bin indices for all samples
    hr_idx[cndr_hj] = j
    #wt = np.sum(cndt_hj,axis=0,keepdims=True).astype(np.float)
    wr = np.nansum(cndr_hj,axis=0,keepdims=True).astype(np.float)

    #wt[wt==0] = np.nan
    wr[wr==0] = np.nan

    jabt_hj[j,...] = np.sum((jabt * cndr_hj[...,None]), axis=0)/wr.T # must use ref. bins !!!
    jabr_hj[j,...] = np.sum((jabr * cndr_hj[...,None]), axis=0)/wr.T
    DE_hj[j,...] = np.nansum((DEi * cndr_hj[...,None])/wr.T, axis = 0).T # local color difference is average of DEi per hue bin !!

# calculate normalized hue-bin averages for jabt, jabr:
ht_hj = cam.hue_angle(jabt_hj[...,1],jabt_hj[...,2],htype='rad')
hr_hj = cam.hue_angle(jabr_hj[...,1],jabr_hj[...,2],htype='rad')
Ct_hj = ((jabt_hj[...,1]**2 + jabt_hj[...,2]**2))**0.5
Cr_hj = ((jabr_hj[...,1]**2 + jabr_hj[...,2]**2))**0.5
Ctn_hj = normalized_chroma_ref*Ct_hj/(Cr_hj + 1e-308) # calculate normalized chroma for samples under test
Ctn_hj[Cr_hj == 0.0] = np.inf
jabtn_hj = jabt_hj.copy()
jabrn_hj = jabr_hj.copy()
jabtn_hj[...,1], jabtn_hj[...,2] = Ctn_hj*np.cos(ht_hj), Ctn_hj*np.sin(ht_hj)
jabrn_hj[...,1], jabrn_hj[...,2] = normalized_chroma_ref*np.cos(hr_hj), normalized_chroma_ref*np.sin(hr_hj)

# calculate normalized versions of jabt, jabr:
jabtn = jabt.copy()
jabrn = jabr.copy()
Ctn = np.zeros((jabt.shape[0],jabt.shape[1]))
Crn = Ctn.copy()
for j in range(nhbins):
    Ctn = Ctn + (Ct/Cr_hj[j,...])*(hr_idx==j)
    Crn = Crn + (Cr/Cr_hj[j,...])*(hr_idx==j)
Ctn*=normalized_chroma_ref
Crn*=normalized_chroma_ref
jabtn[...,1] = (Ctn*np.cos(ht))
jabtn[...,2] = (Ctn*np.sin(ht))
jabrn[...,1] = (Crn*np.cos(hr))
jabrn[...,2] = (Crn*np.sin(hr))


# closed jabt_hj, jabr_hj for Rg:
jabt_hj_closed = np.vstack((jabt_hj,jabt_hj[:1,...]))
jabr_hj_closed = np.vstack((jabr_hj,jabr_hj[:1,...]))

# closed jabtn_hj, jabrn_hj for plotting:
jabtn_hj_closed = np.vstack((jabtn_hj,jabtn_hj[:1,...]))
jabrn_hj_closed = np.vstack((jabrn_hj,jabrn_hj[:1,...]))

#---------------------------------------------------------------
jabt,jabr=out
# jabt,jabr = data['bjabt'],data['bjabr']
_jab_test, _jab_ref = jabt.copy(), jabr.copy()
close_gamut = True

# make 3d for easy looping:
_test_original_shape = _jab_test.shape

if len(_test_original_shape)<3:
    _jab_test = _jab_test[:,None]
    _jab_ref = _jab_ref[:,None]

#initialize Jabt, Jabr, binnr, DEi;
_test_shape = list(_jab_test.shape)
if nhbins is not None:
    _nhbins = np.int(nhbins)
    _test_shape[0] = _nhbins + close_gamut*1
else:
    _nhbins = nhbins
    _test_shape[0] = _test_shape[0] + close_gamut*1
_jabt = np.zeros(_test_shape)
_jabr = _jabt.copy()
_binnr = _jab_test[...,0].copy()
_DEi = _jabt[...,0].copy()

# Store all samples (for output of potentially scaled coordinates):
# if ('jabti' in out) | ('jabri' in out):
_jabti = _jab_test.copy()
_jabri = _jab_ref.copy()

# Loop over axis 1:
for ii in range(_jab_test.shape[1]):
      
    # calculate hue angles:
    _ht = cam.hue_angle(_jab_test[:,ii,1],_jab_test[:,ii,2], htype='rad')
    _hr = cam.hue_angle(_jab_ref[:,ii,1],_jab_ref[:,ii,2], htype='rad')

    #divide huecircle/data in n hue slices:
    _hbins = np.floor(((_hr - start_hue*np.pi/180)/2/np.pi) * nhbins) # because of start_hue bin range can be different from 0 : n-1
    _hbins[_hbins>=_nhbins] = _hbins[_hbins>=_nhbins] - _nhbins # reset binnumbers to 0 : n-1 range
    _hbins[_hbins < 0] = (_nhbins - 2) - _hbins[_hbins < 0] # reset binnumbers to 0 : n-1 range

    _jabtii = np.zeros((_nhbins,3))
    _jabrii = np.zeros((_nhbins,3))
    for i in range(nhbins):
        if i in _hbins:
            _jabtii[i,:] = _jab_test[_hbins==i,ii,:].mean(axis = 0)
            _jabrii[i,:] = _jab_ref[_hbins==i,ii,:].mean(axis = 0)
            _DEi[i,ii] =  np.sqrt(np.power((_jab_test[_hbins==i,ii,:] - _jab_ref[_hbins==i,ii,:]),2).sum(axis = _jab_test[_hbins==i,ii,:].ndim -1)).mean(axis = 0)
    _jabt_hj = _jabtii.copy()
    _jabr_hj = _jabrii.copy()
    
    if normalize_gamut == True:
        
        #renormalize jab_test, jab_ref using jabrii:
        # if ('jabti' in out) | ('jabri' in out):
        _Cti = np.sqrt(_jab_test[:,ii,1]**2 + _jab_test[:,ii,2]**2)
        _Cri = np.sqrt(_jab_ref[:,ii,1]**2 + _jab_ref[:,ii,2]**2)
        _hti = _ht.copy()
        _hri = _hr.copy()
            
        #renormalize jabtii using jabrii:
        _Ct_hj = np.sqrt(_jabtii[:,1]**2 + _jabtii[:,2]**2)
        _Cr_hj = np.sqrt(_jabrii[:,1]**2 + _jabrii[:,2]**2)
        _ht_hj = cam.hue_angle(_jabtii[:,1],_jabtii[:,2], htype = 'rad')
        _hr_hj = cam.hue_angle(_jabrii[:,1],_jabrii[:,2], htype = 'rad')
    
        # calculate rescaled chroma of test:
        _C_hj = normalized_chroma_ref*(_Ct_hj/_Cr_hj) 
    
        # calculate normalized cart. co.: 
        _jabtn_hj = _jabtii.copy()
        _jabrn_hj = _jabrii.copy()
        _jabtn_hj[:,1] = _C_hj*np.cos(_ht_hj)
        _jabtn_hj[:,2] = _C_hj*np.sin(_ht_hj)
        _jabrn_hj[:,1] = normalized_chroma_ref*np.cos(_hr_hj)
        _jabrn_hj[:,2] = normalized_chroma_ref*np.sin(_hr_hj)
        
        # generate scaled coordinates for all samples:
        # if ('jabti' in out) | ('jabri' in out):
        _Ctn = _Cti.copy()
        _Crn = _Cri.copy()
        _jabtn = _jabti.copy()
        _jabrn = _jabri.copy()
        for i in range(_nhbins):
            if i in _hbins:
                _Ctn[_hbins==i] = normalized_chroma_ref*(_Ctn[_hbins==i]/_Cr_hj[i]) 
                _Crn[_hbins==i] = normalized_chroma_ref*(_Crn[_hbins==i]/_Cr_hj[i]) 
        _jabtn[:,ii,1] = _Ctn*np.cos(_hti)
        _jabtn[:,ii,2] = _Ctn*np.sin(_hti)
        _jabrn[:,ii,1] = _Crn*np.cos(_hri)
        _jabrn[:,ii,2] = _Crn*np.sin(_hri)
    
    if close_gamut == True:
        _jabtn_hj= np.vstack((_jabtn_hj,_jabtn_hj[0,:])) # to create closed curve when plotting
        _jabrn_hj = np.vstack((_jabrn_hj,_jabrn_hj[0,:])) # to create closed curve when plotting


    _jabt[:,ii,:] = _jabtn_hj
    _jabr[:,ii,:] = _jabrn_hj
    _binnr[:,ii] = _hbins



