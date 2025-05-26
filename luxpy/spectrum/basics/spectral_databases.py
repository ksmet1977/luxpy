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
Module for loading light source (spd) and reflectance (rfl) spectra databases
=============================================================================

 :_S_PATH: Path to light source spectra data.

 :_R_PATH: Path to with spectral reflectance data

 :_IESTM3015: Database with spectral reflectances related to and light source 
            spectra contained excel calculator of IES TM30-15 publication.
            
 :_IESTM3018: Database with spectral reflectances related to and light source 
            spectra contained excel calculator of IES TM30-18 publication.

 :_IESTM3015_S: Database with only light source spectra contained in the 
              IES TM30-15 excel calculator.
              
 :_IESTM3018_S: Database with only light source spectra contained in the 
              IES TM30-18 excel calculator.

 :_CIE_ILLUMINANTS: | Database with CIE illuminants: 
                    | * 'E', 'D65', 'A', 'C',
                    | * 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                      'F7', 'F8', 'F9', 'F10', 'F11', 'F12',...
                      
 :_CIE_E, _CIE_D65, _CIE_A, ',_CIE_B', _CIE_C, _CIE_F4, _CIE_L41: Some CIE illuminants for easy use.

 :_CRI_RFL: | Database with spectral reflectance functions for various 
              color rendition calculators:
            | * `CIE 13.3-1995 (8, 14 & 15 munsell samples) <http://www.cie.co.at/index.php/index.php?i_ca_id=303>`_
            | * `CIE 224:2017 (99 set) <http://www.cie.co.at/index.php?i_ca_id=1027>`_
            | * `CRI2012 (HL17 & HL1000 spectrally uniform and 210 real samples) <http://journals.sagepub.com/doi/abs/10.1177/1477153513481375>`_
            | * `IES TM30 (99, 4880, 2696 spectrally uniform samples) <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition>`_
            | * `MCRI (10 familiar object set) <http://www.sciencedirect.com/science/article/pii/S0378778812000837>`_
            | * `CQS (v7.5 and v9.0 sets) <http://spie.org/Publications/Journal/10.1117/1.3360335>`_

 :_MUNSELL: Database (dict) with 1269 Munsell spectral reflectance functions 
            and Value (V), Chroma (C), hue (h) and (ab) specifications.
           
 :_RFL: | Database (dict) with RFLs, including:
        | * all those in _CRI_RFL, 
        | * the 1269 Matt Munsell samples (see also _MUNSELL),
        | * the 24 Macbeth ColorChecker samples,
        | * the 215 samples proposed by Opstelten, J.J. , 1983, The establishment of a representative set of test colours
        |   for the specification of the colour rendering properties of light sources, CIE-20th session, Amsterdam. 
        | * the 114120 RFLs from `(capbone.com/spectral-reflectance-database/)<114120 RFLs from https://capbone.com/spectral-reflectance-database/>`_
        
 :_CIE_GLASS_ID: CIE spectral transmission to convert illuminants to indoor variants.
 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import copy
import numpy as np

from luxpy.utils import _PKG_PATH, _SEP, getdata

__all__ = ['_R_PATH','_S_PATH', 
           '_CIE_ILLUMINANTS', '_CIE_E', '_CIE_D65', '_CIE_A', '_CIE_B', '_CIE_C', '_CIE_F4', '_CIE_L41',
           '_CIE_F_SERIES', '_CIE_F3_SERIES','_CIE_HP_SERIES','_CIE_LED_SERIES',
           '_IESTM3015','_IESTM3015_S','_IESTM3018','_IESTM3018_S','_IESTM3020','_IESTM3020_S',
           '_CIE_GLASS_ID','_CRI_RFL','_RFL', '_MUNSELL']

_S_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'spds' + _SEP #folder with spd data
_R_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'rfls' + _SEP #folder with rfl data

###############################################################################
# spectral power distributions:

#------------------------------------------------------------------------------
# Illuminant library: set some typical CIE illuminants:
E = np.array([np.linspace(360,830,471),np.ones(471)])
_CIE_E = E

D65 = getdata(_S_PATH + 'CIE_D65.csv').T
_CIE_D65 = D65

C = getdata(_S_PATH + 'CIE_C.csv').T
_CIE_C = C

A = getdata(_S_PATH + 'CIE_A.csv').T
_CIE_A = A

B = getdata(_S_PATH + 'CIE_B.csv').T
_CIE_B = B

L41 = getdata(_S_PATH + 'CIE_L41.csv').T # illuminant for spectral mismatch calculations (cfr. CIE TC2-90)
_CIE_L41 = L41

_CIE_F_SERIES = getdata(_S_PATH + 'CIE_F_1to12_1nm.csv').T
_CIE_F_SERIES_dict = {'F{:1.0f}'.format(i+1):np.vstack((_CIE_F_SERIES[0],_CIE_F_SERIES[i+1])) for i in range(12)}
_CIE_F4 = _CIE_F_SERIES_dict['F4']
  
_CIE_F3_SERIES = getdata(_S_PATH + 'CIE_F3_1to15.csv').T
_CIE_F3_SERIES_dict = {'F3.{:1.0f}'.format(i+1):np.vstack((_CIE_F3_SERIES[0],_CIE_F3_SERIES[i+1])) for i in range(15)}

_CIE_HP_SERIES = getdata(_S_PATH + 'CIE_HP_1to5.csv').T
_CIE_HP_SERIES_dict = {'HP{:1.0f}'.format(i+1):np.vstack((_CIE_HP_SERIES[0],_CIE_HP_SERIES[i+1])) for i in range(5)}

_CIE_LED_SERIES = getdata(_S_PATH + 'CIE_LED_B1toB5_BH1_RGB1_V1_V2.csv').T
_CIE_LED_types = ['B1','B2','B3','B4','B5','BH1','RGB1','V1','V2']
_CIE_LED_SERIES_dict = {'LED_{:s}'.format(_CIE_LED_types[i]):np.vstack((_CIE_LED_SERIES[0],_CIE_LED_SERIES[i+1])) for i in range(len(_CIE_LED_types))}

_CIE_ILLUMINANTS = {'E':E,'D65':D65,'A':A,'B':B,'C':C,'F4':_CIE_F4,'L41':_CIE_L41}
_CIE_ILLUMINANTS.update(_CIE_F_SERIES_dict)
_CIE_ILLUMINANTS.update(_CIE_F3_SERIES_dict)
_CIE_ILLUMINANTS.update(_CIE_HP_SERIES_dict)
_CIE_ILLUMINANTS.update(_CIE_LED_SERIES_dict)
_CIE_ILLUMINANTS['types'] = list(_CIE_ILLUMINANTS.keys())
       
_CIE_ILLUMINANTS['series'] = {}
_CIE_ILLUMINANTS['series']['F'] = _CIE_F_SERIES
_CIE_ILLUMINANTS['series']['F3'] = _CIE_F3_SERIES
_CIE_ILLUMINANTS['series']['HP'] = _CIE_HP_SERIES
_CIE_ILLUMINANTS['series']['LED'] = _CIE_LED_SERIES
_CIE_ILLUMINANTS['types'].append('series')


# load TM30 spd data base:
_IESTM3015 = {'S': {'data': getdata(_S_PATH + 'IESTM30_15_Sspds.dat').transpose()}}
_IESTM3015['S']['info'] = getdata(_S_PATH + 'IESTM30_15_Sinfo.txt',header='infer',verbosity = False, dtype = None)
_IESTM3015_S = _IESTM3015['S']

_IESTM3018 = {'S': {'data': getdata(_S_PATH + 'IESTM30_15_Sspds.dat').transpose()}}
_IESTM3018['S']['info'] = getdata(_S_PATH + 'IESTM30_15_Sinfo.txt',header='infer',verbosity = False, dtype = None)
_IESTM3018_S = _IESTM3018['S']
_IESTM3020 = _IESTM3018
_IESTM3020_S = _IESTM3020['S']
_IESTM3024 = _IESTM3018
_IESTM3024_S = _IESTM3024['S']
    
###############################################################################
# spectral reflectance/transmission functions:

# Glass spectral mission for indoor illuminants:
_CIE_GLASS_ID = {'T': getdata(_R_PATH + 'GlassSpecTrans_indoor_illuminants.csv').T}

#------------------------------------------------------------------------------
# CIE 13.3-1995 color rendering index:
_CIE133_1995 = {'14': {'5nm' : getdata(_R_PATH + 'CIE_13_3_1995_R14.dat').T}}
_CIE133_1995['8'] = {'5nm' : _CIE133_1995['14']['5nm'][0:9].copy()}
_JISZ8726_R15 = getdata(_R_PATH + 'JIS-Z-8726-R15.dat').T # = R15 J-Z-8726 sample (asian skin)
_CIE133_1995['15'] = {'5nm' : np.vstack((_CIE133_1995['14']['5nm'].copy(),_JISZ8726_R15[1:]))}

# CIE13.3-1995 requires linear interpolation from 5 nm to smaller intervals, contrary
# to the CIE015 recommended method for interpolation for spectral reflectances. Hence,
# we pre-interpolate here so these are available with te correct interpolation method applied.
wln = np.arange(360,831,1)
for key in list(_CIE133_1995.keys()):
    _CIE133_1995[key]['1nm'] = np.vstack((wln,np.array([np.interp(wln,_CIE133_1995[key]['5nm'][0],_CIE133_1995[key]['5nm'][i+1]) for i in range(_CIE133_1995[key]['5nm'].shape[0]-1)])))

#------------------------------------------------------------------------------  
# IES TM30-15 color fidelity and color gamut indices:
# (note that wavelength range of rfls has been extended from [380-780] nm using flat-extrapolation to [360-830] nm.)
_IESTM3015['R'] = {'4880' : {'1nm': np.load(_R_PATH + 'IESTM30_15_R4880.npz')['_IESTM30_R4880']}}
# _IESTM3015['R'] = {'4880' : {'1nm': getdata(_R_PATH + 'IESTM30_15_R4880.csv',kind='np')}}
_IESTM3015['R']['99'] = {'1nm' : getdata(_R_PATH + 'IESTM30_15_R99_1nm.dat').T}
_IESTM3015['R']['99']['5nm'] = getdata(_R_PATH + 'IESTM30_15_R99_5nm.dat').T
temp = getdata(_R_PATH + 'IESTM30_15_R99info.dat')[0]
ies99categories = ['nature','skin','textiles','paints','plastic','printed','color system']
_IESTM3015['R']['99']['info'] = [ies99categories[int(i-1)] for i in temp]

# selection from TM30 (2015) 4880 set by removing duplicate rfls:
rfl4880 = _IESTM3015['R']['4880']['1nm']
unique_rfls_idxs = np.asarray(getdata(_R_PATH + 'IESTM30-R4880-subset-with-unique-rfls--2696.dat'),dtype = int)[:,0] # Jan 2025: selection from 4880 set by removing duplicate rfls (duplicate if DEjab == 0)!
_IESTM3015['R']['2696'] = {'1nm': np.vstack((rfl4880[0],rfl4880[1:][unique_rfls_idxs]))}


#------------------------------------------------------------------------------
# cie 224:2017 (color fidelity index based on IES TM-30-15):
# (note that wavelength range of rfls has been extended from [380-780] nm using flat-extrapolation to [360-830] nm.)
_CIE224_2017 = {'99': {'1nm' : getdata(_R_PATH + 'CIE224_2017_R99_1nm.dat').T}} # 25/02/25: same as 1 nm data from TM30-18 (used to be from calculator [see '...-from excelcalculator.dat' file], but doesn't match linear, cubic, sprague5, spragie_CIE224_2017 interpolation, although cubic is closed)
_CIE224_2017['99']['5nm'] = getdata(_R_PATH + 'CIE224_2017_R99_5nm.dat').T # from 5 nm calculator (luxpy.math.interp1_sprague_cie224_2017 to 1 nm has max diff of order 1e-16 with TM30-24 1 nm data! )
_CIE224_2017['99']['info'] = _IESTM3015['R']['99']['info']


#------------------------------------------------------------------------------  
# IES TM30-18 and TM30-20 and TM30-24 color fidelity and color gamut indices:
# (note that wavelength range of rfls has been extended from [380-780] nm using flat-extrapolation to [360-830] nm.)
_IESTM3018['R'] = copy.deepcopy(_IESTM3015['R'])
_IESTM3018['R']['99']['1nm'] = getdata(_R_PATH + 'IESTM30_18_R99_1nm.dat').T
_IESTM3018['R']['99']['5nm'] = _IESTM3018['R']['99']['1nm'][:,::5].copy() # [::5] is equivalent to linear or cubic interpolation as data points are kept on upsampling
_IESTM3020['R']['99']['1nm'] = _IESTM3018['R']['99']['1nm']
_IESTM3020['R']['99']['5nm'] = _IESTM3018['R']['99']['5nm'] 
_IESTM3024['R']['99']['1nm'] = _IESTM3018['R']['99']['1nm']
_IESTM3024['R']['99']['5nm'] = _IESTM3018['R']['99']['5nm'] 



#------------------------------------------------------------------------------
# CRI2012 spectrally uniform mathematical sampleset:
_CRI2012 = {'HL17' : getdata(_R_PATH + 'CRI2012_HL17.dat').T}
_CRI2012['HL1000'] = getdata(_R_PATH +'CRI2012_Hybrid14_1000.dat').T
_CRI2012['Real210'] = getdata(_R_PATH +'CRI2012_R210.dat').T


#------------------------------------------------------------------------------
# MCRI (memory color rendition index, Rm) sampleset:
_MCRI= {'R' : getdata(_R_PATH + 'MCRI_R10.dat').T}
_MCRI['info'] = ['apple','banana','orange','lavender','smurf','strawberry yoghurt','sliced cucumber', 'cauliflower','caucasian skin','N4'] # familiar objects, N4: neutral (approx. N4) gray sphere 


#------------------------------------------------------------------------------
# CQS versions 7.5 and 9.0:
_CQS = {'v7.5': getdata(_R_PATH + 'CQSv7dot5.dat').T}
_CQS['v9.0'] =  getdata(_R_PATH + 'CQSv9dot0.dat').T


#------------------------------------------------------------------------------
# FCI (Feeling of Contrast Index) sampleset:
_FCI= {'R' : getdata(_R_PATH + 'FCI_RFL4.csv').T}


#------------------------------------------------------------------------------
# collect in one dict:
_CRI_RFL = {'cie-13.3-1995': _CIE133_1995}
_CRI_RFL['cie-224-2017'] = _CIE224_2017
_CRI_RFL['cri2012'] = _CRI2012
_CRI_RFL['ies-tm30-15'] = _IESTM3015['R']
_CRI_RFL['ies-tm30-18'] = _IESTM3018['R']
_CRI_RFL['ies-tm30-20'] = _IESTM3020['R']
_CRI_RFL['ies-tm30-24'] = _IESTM3024['R']
_CRI_RFL['ies-tm30'] = _IESTM3024['R']
_CRI_RFL['mcri'] = _MCRI['R']
_CRI_RFL['cqs'] = _CQS
_CRI_RFL['fci'] = _FCI['R']

#------------------------------------------------------------------------------
# 1269 Munsell spectral reflectance functions:
_MUNSELL = {'cieobs':'1931_2', 'Lw' : 400.0, 'Yb': 0.2}
_MUNSELL['R'] = getdata(_R_PATH + 'Munsell1269.dat').T
temp = getdata(_R_PATH + 'Munsell1269NotationInfo.dat',header = 'infer',verbosity=0, dtype = None)
_MUNSELL['H'] = temp[:,1,None]
_MUNSELL['V'] = temp[:,2,None].astype(float)
_MUNSELL['C'] = temp[:,3,None].astype(float)
_MUNSELL['h'] = temp[:,4,None].astype(float)
_MUNSELL['ab'] = temp[:,5:7].astype(float)

del temp, ies99categories


# Initialize _RFL and set non-essential sets to None:
_RFL = {'cri' : _CRI_RFL,
        'munsell': _MUNSELL, 
        'macbeth': None,
        'capbone': None,
        'opstelten': None}


#------------------------------------------------------------------------------
# 215 samples proposed by Opstelten, J.J. , 1983, The establishment of a representative set of test colours
# for the specification of the colour rendering properties of light sources, CIE-20th session, Amsterdam. 
_OPSTELTEN215 = {'R' : getdata(_R_PATH + 'Opstelten1983_215.dat').T}


#------------------------------------------------------------------------------
# 24 MacBeth ColorChecker RFLs
_MACBETH_RFL = {'CC': {'R': getdata(_R_PATH + 'MacbethColorChecker.dat').T}}

#------------------------------------------------------------------------------
# 114120 RFLs from https://capbone.com/spectral-reflectance-database/
try:
    _CAPBONE_100K_RFL = {'R': np.load(_R_PATH + 'capbone_100k_rfls.npz')['_CAPBONE_100K_RFL']}
    # _CAPBONE_100K_RFL = {'R': getdata(_R_PATH + 'capbone_100k_rfls.csv',kind='np')}
    _CAPBONE_100K_RFL['file'] = _R_PATH + 'capbone_100k_rfls.npz'
except:
    _CAPBONE_100K_RFL = {'R': None}
    _CAPBONE_100K_RFL['file']  = _R_PATH + 'capbone_100k_rfls.npz'
finally:
    _CAPBONE_RFL = {'100k': _CAPBONE_100K_RFL}
    
# Add to _RFL:
_RFL['macbeth'] = _MACBETH_RFL
_RFL['capbone'] = _CAPBONE_RFL
_RFL['opstelten'] = _OPSTELTEN215
