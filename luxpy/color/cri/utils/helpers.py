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
Module with color rendition, fidelity and gamut area helper functions
=====================================================================

:gamut_slicer(): Slices the gamut in nhbins slices and provides normalization 
                  of test gamut to reference gamut.

 :jab_to_rg(): Calculates gamut area index, Rg.

 :jab_to_rhi(): | Calculate hue bin measures: 
                |   Rfhi (local (hue bin) color fidelity)
                |   Rcshi (local chroma shift) 
                |   Rhshi (local hue shift)

 :spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated
                    with test source and its reference illuminant.

 :spd_to_rg(): Calculates the color gamut index of spectral data 
               for a sample set illuminated with test source (data) 
               with respect to some reference illuminant.

 :spd_to_DEi(): Calculates color difference (~fidelity) of spectral data 
                between sample set illuminated with test source (data) 
                and some reference illuminant.

 :optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type 
                           such that average Rf for a set of light sources is 
                           the same as that of a target-cri (default: 'ciera')

 :spd_to_cri(): Calculates the color rendering fidelity index 
                (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
                Can also output Rg, Rfhi, Rcshi, Rhshi, cct, duv, ...

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import (np, _S_INTERP_TYPE, _CRI_RFL, _IESTM3015, math, cam, cat,
                minimize, asplit, np2d, spd, put_args_in_db, 
                colortf, spd_to_xyz, cri_ref, xyz_to_cct)

from .DE_scalers import linear_scale, log_scale, psy_scale

from .init_cri_defaults_database import _CRI_TYPE_DEFAULT, _CRI_DEFAULTS, process_cri_type_input

__all__ = ['gamut_slicer','jab_to_rg', 'jab_to_rhi', 'jab_to_DEi',
           'spd_to_DEi', 'spd_to_rg', 'spd_to_cri']

#------------------------------------------------------------------------------
def gamut_slicer(jab_test,jab_ref, out = 'jabt,jabr', nhbins = None, \
                 start_hue = 0.0, normalize_gamut = True, \
                 normalized_chroma_ref = 100, close_gamut = False):
    """
    Slices the gamut in hue bins.
    
    Args:
        :jab_test: 
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the test SPD
        :jab_ref:
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the reference SPD
        :out: 
            | 'jabt,jabr' or str, optional
            | Specifies which variables to output as ndarray
        :nhbins:
            | None or int, optional
            |   - None: defaults to using the sample hues themselves as 'bins'. 
            |           In other words, the number of bins will be equal to the 
            |           number of samples.
            |   - float: number of bins to slice the sample gamut in.
        :start_hue:
            | 0.0 or float, optional
            | Hue angle to start bin slicing
        :normalize_gamut:
            | True or False, optional
            | True normalizes the gamut of test to that of ref.
              (perfect agreement results in circle).
        :normalized_chroma_ref:
            | 100.0 or float, optional
            | Controls the size (chroma/radius) of the normalization circle/gamut.
        :close_gamut:
            | False or True, optional
            | True appends the first jab coordinates to the end of the output 
              (for plotting closed gamuts)
    
    Returns:
        :returns:
            | ndarray with average jabt,jabr of each hue bin. 
            |  (.shape = (number of hue bins, 3))
            | 
            |  (or outputs whatever is specified in :out:) 
    """

    # make 3d for easy looping:
    test_original_shape = jab_test.shape

    if len(test_original_shape)<3:
        jab_test = jab_test[:,None]
        jab_ref = jab_ref[:,None]
    
    #initialize Jabt, Jabr, binnr, DEi;
    test_shape = list(jab_test.shape)
    if nhbins is not None:
        nhbins = np.int(nhbins)
        test_shape[0] = nhbins + close_gamut*1
    else:
        test_shape[0] = test_shape[0] + close_gamut*1
    jabt = np.zeros(test_shape)
    jabr = jabt.copy()
    binnr = jab_test[...,0].copy()
    DEi = jabt[...,0].copy()
    
    # Loop over axis 1:
    for ii in range(jab_test.shape[1]):
          
        # calculate hue angles:
        ht = cam.hue_angle(jab_test[:,ii,1],jab_test[:,ii,2], htype='rad')
        hr = cam.hue_angle(jab_ref[:,ii,1],jab_ref[:,ii,2], htype='rad')

        if nhbins is None:
            Ir = np.argsort(hr)
            jabtii = jab_test[Ir,ii,:]
            jabrii = jab_ref[Ir,ii,:]
            nhbins = (jabtii.shape[0])
            DEi[...,ii] =  np.sqrt(np.power((jabtii - jabtii),2).sum(axis = jabtii.ndim -1))
        else:
            
            #divide huecircle/data in n hue slices:
            hbins = np.floor(((hr - start_hue*np.pi/180)/2/np.pi) * nhbins) # because of start_hue bin range can be different from 0 : n-1
            hbins[hbins>=nhbins] = hbins[hbins>=nhbins] - nhbins # reset binnumbers to 0 : n-1 range
            hbins[hbins < 0] = (nhbins - 2) - hbins[hbins < 0] # reset binnumbers to 0 : n-1 range

            jabtii = np.zeros((nhbins,3))
            jabrii = np.zeros((nhbins,3))
            for i in range(nhbins):
                if i in hbins:
                    jabtii[i,:] = jab_test[hbins==i,ii,:].mean(axis = 0)
                    jabrii[i,:] = jab_ref[hbins==i,ii,:].mean(axis = 0)
                    DEi[i,ii] =  np.sqrt(np.power((jab_test[hbins==i,ii,:] - jab_ref[hbins==i,ii,:]),2).sum(axis = jab_test[hbins==i,ii,:].ndim -1)).mean(axis = 0)

        if normalize_gamut == True:
            #renormalize jabtii using jabrii:
            Ct = np.sqrt(jabtii[:,1]**2 + jabtii[:,2]**2)
            Cr = np.sqrt(jabrii[:,1]**2 + jabrii[:,2]**2)
            ht = cam.hue_angle(jabtii[:,1],jabtii[:,2], htype = 'rad')
            hr = cam.hue_angle(jabrii[:,1],jabrii[:,2], htype = 'rad')
        
            # calculate rescaled chroma of test:
            C = normalized_chroma_ref*(Ct/Cr) 
        
            # calculate normalized cart. co.: 
            jabtii[:,1] = C*np.cos(ht)
            jabtii[:,2] = C*np.sin(ht)
            jabrii[:,1] = normalized_chroma_ref*np.cos(hr)
            jabrii[:,2] = normalized_chroma_ref*np.sin(hr)
        
        if close_gamut == True:
            jabtii = np.vstack((jabtii,jabtii[0,:])) # to create closed curve when plotting
            jabrii = np.vstack((jabrii,jabrii[0,:])) # to create closed curve when plotting

        jabt[:,ii,:] = jabtii
        jabr[:,ii,:] = jabrii
        binnr[:,ii] = hbins

    # circle coordinates for plotting:
    hc = np.arange(360.0)*np.pi/180.0
    jabc = np.ones((hc.shape[0],3))*100
    jabc[:,1] = normalized_chroma_ref*np.cos(hc)
    jabc[:,2] = normalized_chroma_ref*np.sin(hc)

    if len(test_original_shape) == 2:
        jabt = jabt[:,0]
        jabr = jabr[:,0]

    if out == 'jabt,jabr':
        return jabt, jabr
    elif out == 'jabt,jabr,DEi':
        return jabt, jabr, DEi
    elif out == 'jabt,jabr,DEi,binnr':
        return jabt, jabr, DEi, binnr
    else:
        return eval(out)        
 
#------------------------------------------------------------------------------
def jab_to_rg(jabt,jabr, max_scale = 100, ordered_and_sliced = False, \
              nhbins = None, start_hue = 0.0, normalize_gamut = True, \
              normalized_chroma_ref = 100, out = 'Rg,jabt,jabr'):
    """
    Calculates gamut area index, Rg.
    
    Args:
        :jabt:  
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the test SPD
        :jabr:
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the reference SPD
        :max_scale:
            | 100.0, optional
            | Value of Rg when Rf = max_scale (i.e. DEavg = 0)
        :ordered_and_sliced: 
            | False or True, optional
            |   - False: Hue ordering will be done with lux.cri.gamut_slicer().
            |   - True: user is responsible for hue-ordering and closing gamut 
                  (i.e. first element in :jab: equals the last).
        :nhbins: 
            | None or int, optional
            |   - None: defaults to using the sample hues themselves as 'bins'. 
            |           In other words, the number of bins will be equal to the 
            |           number of samples.
            |   - float: number of bins to slice the sample gamut in.
        :start_hue:
            | 0.0 or float, optional
            | Hue angle to start bin slicing
        :normalize_gamut:
            | True or False, optional
            | True normalizes the gamut of test to that of ref.
              (perfect agreement results in circle).
        :normalized_chroma_ref:
            | 100.0 or float, optional
            | Controls the size (chroma/radius) of the normalization circle/gamut
        :out: 
            | 'Rg,jabt,jabr' or str, optional
            | Specifies which variables to output as ndarray

    Returns: 
        :Rg: 
            | float or ndarray with gamut area indices Rg.
    """    
    # slice, order and normalize jabt and jabr:
    if ordered_and_sliced == False: 
        jabt, jabr, DEi = gamut_slicer(jabt,jabr, out = 'jabt,jabr,DEi', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)
 
    # make 3d:
    test_original_shape = jabt.shape
    if len(test_original_shape)<3:
        jabt = jabt[None] # expand 2-array to 3-array by adding '0'-axis
        jabr = jabt[None] # expand 2-array to 3-array by adding '0'-axis
    
    # calculate Rg for each spd:
    Rg = np.zeros((1,jabt.shape[1]))

    for ii in range(jabt.shape[1]):
        Rg[:,ii] = max_scale*math.polyarea(jabt[:,ii,1],jabt[:,ii,2])/math.polyarea(jabr[:,ii,1],jabr[:,ii,2]) # calculate Rg =  gamut area ratio of test and ref
    
    if out == 'Rg':
        return Rg
    elif (out == 'Rg,jabt,jabr'):
        return Rg, jabt, jabr
    elif (out == 'Rg,jabt,jabr,DEi'):
        return Rg, jabt, jabr, DEi
    else:
        return eval(out)


#------------------------------------------------------------------------------
def jab_to_rhi(jabt, jabr, DEi, cri_type = _CRI_TYPE_DEFAULT, start_hue = None,\
               nhbins = None, scale_factor = None, scale_fcn = None, \
               use_bin_avg_DEi = True):
    """
    Calculate hue bin measures: Rfhi, Rcshi and Rhshi.
    
    |   Rfhi: local (hue bin) color fidelity  
    |   Rcshi: local chroma shift
    |   Rhshi: local hue shift
    |
    |   (See IES TM30)
    
    Args:
        :jabt: 
            | ndarray with jab coordinates under test SPD
        :jabr: 
            | ndarray with jab coordinates under reference SPD
        :DEi: 
            | ndarray with DEi (from gamut_slicer()).
        :use_bin_avg_DEi: 
            | True, optional
            | Note that following IES-TM30 DEi from gamut_slicer() is obtained by
              averaging the DEi per hue bin (True), and NOT by averaging the 
              jabt and jabr per hue  bin and then calculating the DEi (False).
        :nhbins:
            | int, number of hue bins to slice gamut 
              (None use the one specified in :cri_type: dict).
        :start_hue: 
            | float (°), hue at which to start slicing
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :scale_factor:
            | factors used in scaling function
        
    Returns:
        :returns: 
            | ndarrays of Rfhi, Rcshi and Rhshi
        
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    """
    if isinstance(cri_type, str): 
        args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
        cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.jab_to_rhi')
    
    if jabt.ndim < 3:
        jabt = jabt[:,None,:]
    if jabr.ndim < 3:
        jabr = jabr[:,None,:]

    # Get scale factor and function:
    if (scale_factor is None):
        scale_factor = cri_type['scale']['cfactor']
    if (scale_fcn is None):
        scale_fcn = cri_type['scale']['fcn']
    if (start_hue is None):
        start_hue = cri_type['rg_pars']['start_hue']
    if (nhbins is None):
        nhbins = cri_type['rg_pars']['nhbins']
     
    # A. Local Color Fidelity, Rfhi:
    if use_bin_avg_DEi == False:
        DEi = np.power((jabt - jabr), 2).sum(axis = len(jabt.shape)-1,keepdims = False)**0.5
    Rfhi = scale_fcn(DEi,scale_factor)
    
    # B.Local chroma shift and hue shift, [Rcshi, Rhshi]:
    # B.1 relative paths:
    Cr = np.sqrt((jabr[...,1:3]**2).sum(axis = jabr[...,1:3].ndim-1))
    da = np.atleast_2d((jabt[...,1] - jabr[...,1])/Cr)
    db = np.atleast_2d((jabt[...,2] - jabr[...,2])/Cr)

    # B.2 Reference unit circle:
    dhbins = 2*np.pi/nhbins
    
    hbincenters = np.arange(start_hue + dhbins/2, 2*np.pi, dhbins)[...,None]
    arc = np.cos(hbincenters)
    brc = np.sin(hbincenters)

    # B.3 calculate local chroma shift, Rcshi:
    Rcshi = da * arc + db * brc
    
    # B.4 calculate local hue shift, Rcshi:
    Rhshi = db * arc - da * brc
    
    return Rfhi, Rcshi, Rhshi 


#------------------------------------------------------------------------------
def jab_to_DEi(jabt, jabr, out = 'DEi', avg = None):
    """
    Calculates color differences (~fidelity), DEi, of Jab input.
    
    Args:
        :jabt: 
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the test SPD
        :jabr:
            | ndarray with Cartesian color coordinates (e.g. Jab) 
              of the samples under the reference SPD
        :avg: 
            | None, optional
            | If None: don't calculate average, else: avg must be function handle
        :out: 
            | 'DEi' or str, optional
            | Specifies requested output (e.g. 'DEi,DEa') 

    Returns:
        :returns:
            | float or ndarray with DEi for :out: 'DEi'
            | Other output is also possible by changing the :out: str value.
    """
      
    # E. calculate DEi
    DEi = np.power((jabt - jabr),2).sum(axis = len(jabt.shape)-1,keepdims = False)**0.5
    if avg is not None:
        DEa = avg(DEi, axis = 0) 
        DEa = np2d(DEa)
    else:
        out = 'DEi' #override any requested output if avg is not supplied and DEa has not been calculated.
  
     # output:
    if (out == 'DEi,DEa'):
        return DEi, DEa
    else:
        return  DEi


#------------------------------------------------------------------------------
def spd_to_jab_t_r(SPD, cri_type = _CRI_TYPE_DEFAULT, out = 'jabt,jabr', wl = None,\
                   sampleset = None, ref_type = None, cieobs  = None, cspace = None,\
                   catf = None, cri_specific_pars = None):
    """
    Calculates jab color values for a sample set illuminated with test source 
    SPD and its reference illuminant.
        
    Args:
        :SPD: 
            | ndarray with spectral data 
              (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'jabt,jabr' or str, optional
            | Specifies requested output (e.g.'jabt,jabr' or 'jabt,jabr,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
              override default values in cri_type dict.
            
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
    
    Returns:
        :returns: 
            | (ndarray, ndarray) 
            | with jabt and jabr data for :out: 'jabt,jabr'
            | 
            | Other output is also possible by changing the :out: str value.
    """
   
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_jab_t_r')

    avg, catf, cieobs, cri_specific_pars, cspace, ref_type, rg_pars, sampleset, scale = [cri_type[x] for x in sorted(cri_type.keys())] 

    # make SPD atleast_2d:
    SPD = np2d(SPD)

    if wl is not None: 
        SPD = spd(data = SPD, interpolation = _S_INTERP_TYPE, kind = 'np', wl = wl)
      
    # obtain sampleset:
    if isinstance(sampleset,str):
        sampleset = eval(sampleset)
    
    # A. calculate reference illuminant:
    # A.a. get xyzw:
    xyztw = spd_to_xyz(SPD, cieobs = cieobs['cct'], rfl = None, out = 1)

    # A.b. get cct:
    cct, duv = xyz_to_cct(xyztw, cieobs = cieobs['cct'], out = 'cct,duv',mode = 'lut')
    
    # A.c. get reference ill.:
    if isinstance(ref_type,np.ndarray):
        Sr = cri_ref(ref_type, ref_type = 'spd', cieobs = cieobs['cct'], wl3 = SPD[0])
    else:
        Sr = cri_ref(cct, ref_type = ref_type, cieobs = cieobs['cct'], wl3 = SPD[0])

    # B. calculate xyz and xyzw of data (spds) and Sr:
    xyzti, xyztw = spd_to_xyz(SPD, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)
    xyzri, xyzrw = spd_to_xyz(Sr, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)

    # C. apply chromatic adaptation for non-cam/lab cspaces:
    if catf is not None:
        D_cat, Dtype_cat, La_cat, catmode_cat, cattype_cat, mcat_cat, xyzw_cat = [catf[x] for x in sorted(catf.keys())]
        
        #if not isinstance(D_cat,list): D_cat = [D_cat]
        if xyzw_cat is None: #transform from xyzwt --> xyzwr
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
        
        else: # transform both xyzwr and xyzwt to xyzw_cat
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)

    # D. convert xyz to colorspace, cam or chromaticity co. lab (i.e. lab, ipt, Yuv, jab, wuv,..):
    # D.a. broadcast xyzw to shape of xyzi:
    xyztw = xyztw[None] 
    xyzrw = xyzrw[None] 

    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyztw # enter test whitepoint
    jabt = colortf(xyzti, tf = cspace['type'], fwtf = cspace_pars)
    
    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyzrw # enter ref. whitepoint
    jabr = colortf(xyzri, tf = cspace['type'], fwtf = cspace_pars)    
    del cspace_pars


    # E. Regulate output:
    if out == 'jabt,jabr':
        return jabt, jabr
    elif out == 'jabt,jabr,cct,duv':
        return jabt,jabr,cct,duv
    else:
        eval(out)


#------------------------------------------------------------------------------
def spd_to_DEi(SPD, cri_type = _CRI_TYPE_DEFAULT, out = 'DEi', wl = None, \
               sampleset = None, ref_type = None, cieobs = None, avg = None, \
               cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates color differences (~fidelity), DEi, of spectral data.
    
    Args:
        :SPD: 
            | ndarray with spectral data 
              (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'DEi' or str, optional
            | Specifies requested output (e.g. 'DEi,DEa,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
              override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
    
    Returns:
        :returns: 
            | float or ndarray with DEi for :out: 'DEi'
            | 
            | Other output is also possible by changing the :out: str value.

    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_DEi')

    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    jabt, jabr, cct, duv = spd_to_jab_t_r(SPD, cri_type = cri_type, out = 'jabt,jabr,cct,duv', wl = wl)
      
    # E. calculate DEi, DEa:
    DEi, DEa = jab_to_DEi(jabt,jabr, out = 'DEi,DEa', avg = cri_type['avg'])
  
     # output:
    if (out != 'DEi'):
        return  eval(out)
    else:
        return DEi

      
#------------------------------------------------------------------------------
def optimize_scale_factor(cri_type, opt_scale_factor, scale_fcn, avg) :
    """
    Optimize scale_factor of cri-model in cri_type 
    such that average Rf for a set of light sources is the same as that 
    of a target-cri (default: 'ciera').
    
    Args:
        :cri_type: 
            | str or dict
            |   -'str: specifies dict with default cri model parameters 
            |       (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |       (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |       for required structure)
        :opt_scale:
            | True or False
            | True: optimize scaling-factor, else do nothing and use value of 
              scaling-factor in :scale: dict.   
        :scale_fcn:
            | function handle to type of cri scale, 
            | e.g. 
            |   * linear()_scale --> (100 - scale_factor*DEi), 
            |   * log_scale --> (cfr. Ohno's CQS), 
            |   * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :avg: 
            | None or fcn handle
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.

    Returns:
        :scaling_factor: 
            | ndarray

    """
    if  np.any(opt_scale_factor):
        if 'opt_cri_type' not in cri_type['scale'].keys(): 
            opt_cri_type = _CRI_DEFAULTS['ciera'] # use CIE Ra-13.3-1995 as target
        if 'opt_spd_set' not in cri_type['scale'].keys(): 
            opt_spd_set = _IESTM3015['S']['data'][0:13] # use CIE F1-F12
        scale_fcn_opt = opt_cri_type ['scale']['fcn']
        scale_factor_opt = opt_cri_type ['scale']['cfactor']
        avg_opt = opt_cri_type ['avg']
        DEa_opt = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = opt_cri_type) # DEa using target cri
        Rf_opt = avg(scale_fcn_opt(DEa_opt,scale_factor_opt))
        
        DEa = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = cri_type) # DEa using current cri

        
        # optimize scale_factor to minimize rms difference:
        sf = cri_type['scale']['cfactor'] # get scale_factor of cri_type to determine len and non-optimized factors

        if (isinstance(sf,float)): #(isinstance(1.0*sf,float))
            sf = [sf]
        if isinstance(opt_scale_factor, bool):
            opt_scale_factor = [opt_scale_factor] 
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,x)) - Rf_opt,axis=1) # optimize the only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[1:]) ))) - Rf_opt,axis=1) # optimize the first cfactor (for scale_factor input of len = 1)
        else:
            x0 = np.ones(np.sum(opt_scale_factor))
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[np.invert(opt_scale_factor)]) ))) - Rf_opt,axis=1) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)
        
        optresult = minimize(fun = optfcn, x0 = x0, args=(), method = 'Nelder-Mead')
        scale_factor = optresult['x']
        
        #Reconstruct 'scale_factor' from optimized and fixed parts:
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            pass #only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            scale_factor = np.hstack( (scale_factor,sf[1:]) )
        else:
          scale_factor = np.hstack( (scale_factor,sf[np.invert(opt_scale_factor)]) ) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)

    else:
        scale_factor = cri_type['scale']['cfactor']
    return scale_factor


#------------------------------------------------------------------------------
def spd_to_rg(SPD, cri_type = _CRI_TYPE_DEFAULT, out = 'Rg', wl = None, \
              sampleset = None, ref_type = None, cieobs  = None, avg = None, \
              cspace = None, catf = None, cri_specific_pars = None, rg_pars = None):
    """
    Calculates the color gamut index, Rg, of spectral data. 
    
    Args:
        :SPD: 
            | ndarray with spectral data 
              (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'Rg' or str, optional
            | Specifies requested output (e.g. 'Rg,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
              override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
        :rg_pars: 
            | None or dict, optional
            | Dict containing specifying parameters for slicing the gamut.
            | Dict structure: 
            |     {'nhbins' : None, 'start_hue' : 0, 'normalize_gamut' : True}
            |    - key: 'nhbins': int, number of hue bins to slice gamut 
            |                 (None use the one specified in :cri_type: dict).
            |    - key: 'start_hue': float (°), hue at which to start slicing
            |    - key: 'normalize_gamut': True or False: 
            |                normalize gamut or not before calculating a gamut 
            |                area index Rg. 
            |    - key: 'normalized_chroma_ref': 100.0 or float, optional
            |                Controls the size (chroma/radius) 
            |                of the normalization circle/gamut.
        :avg: 
            | None or fcn handle, optional
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.
        :scale:
            | None or dict, optional
            | Specifies scaling of color differences to obtain CRI.
            |     - None use the one specified in :cri_type: dict.
            |     - dict: user specified dict with scaling parameters.
            |         - key: 'fcn': function handle to type of cri scale, 
            |                 e.g. 
            |                 * linear()_scale --> (100 - scale_factor*DEi), 
            |                 * log_scale --> (cfr. Ohno's CQS), 
            |                 * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
            |        - key: 'cfactor': factors used in scaling function, 
            |              If None: 
            |                     Scaling factor value(s) will be optimized to 
            |                     minimize the rms between the Rf's of the 
            |                     requested metric and the target metric specified
            |                     in:
            |
            |                  - key: 'opt_cri_type':  str 
            |                      * str: one of the preset _CRI_DEFAULTS
            |                      * dict: user speciied 
            |                      (dict must contain all keys as normal)
            |                     Note that if key not in :scale: dict, 
            |                     then 'opt_cri_type' is added with default 
            |                     setting = 'ciera'.
            |                 - key: 'opt_spd_set': ndarray with set of light 
            |                     source spds used to optimize cfactor. 
            |                     Note that if key not in :scale: dict, 
            |                     then default = 'F1-F12'.

    Returns:
        :returns:
            | float or ndarray with Rg for :out: 'Rg'
            | Other output is also possible by changing the :out: str value.
            
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_rg')

    #avg, catf, cieobs, cieobs_cct, cri_specific_pars, cspace, cspace_pars, ref_type, rg_pars, sampleset, scale_factor, scale_fcn = [cri_type[x] for x in sorted(cri_type.keys())] 

       
    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    jabt, jabr,cct,duv = spd_to_jab_t_r(SPD, cri_type = cri_type, out = 'jabt,jabr,cct,duv', wl = wl) 

    
    # calculate gamut area index:
    rg_pars = cri_type['rg_pars']
    #rg_pars = put_args_in_db(cri_type['rg_pars'],rg_pars)#{'nhbins':nhbins,'start_hue':start_hue,'normalize_gamut':normalize_gamut}) #override with not-None input from function
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue  = [rg_pars[x] for x in sorted(rg_pars.keys())]
    
    Rg, jabt_binned, jabr_binned, DEi_binned = jab_to_rg(jabt,jabr, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, out = 'Rg,jabt,jabr,DEi')
    Rg = np2d(Rg)
    
   
    if (out == 'Rg'):
        return Rg
    elif (out == 'Rg,jabt,jabr'):
        return Rg, jabt_binned,jabr_binned
    elif (out == 'Rg,jabt,jabr,DEi'):
        return Rg, jabt_binned,jabr_binned,DEi_binned
    else:
        return eval(out)



#------------------------------------------------------------------------------
def spd_to_cri(SPD, cri_type = _CRI_TYPE_DEFAULT, out = 'Rf', wl = None, \
               sampleset = None, ref_type = None, cieobs = None, avg = None, \
               scale = None, opt_scale_factor = False, cspace = None, catf = None,\
               cri_specific_pars = None, rg_pars = None):
    """
    Calculates the color rendering fidelity index, Rf, of spectral data. 
    
    Args:
        :SPD: 
            | ndarray with spectral data 
              (can be multiple SPDs, first axis are the wavelengths)
        :out: 
            | 'Rf' or str, optional
            | Specifies requested output (e.g. 'Rf,cct,duv') 
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :cri_type:
            | _CRI_TYPE_DEFAULT or str or dict, optional
            |   -'str: specifies dict with default cri model parameters 
            |     (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            |   - dict: user defined model parameters 
            |     (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] 
            |     for required structure)
            | Note that any non-None input arguments to the function will 
              override default values in cri_type dict.
        :sampleset:
            | None or ndarray or str, optional
            | Specifies set of spectral reflectance samples for cri calculations.
            |     - None defaults to standard set for metric in cri_type.
            |     - ndarray: user defined set of spectral reflectance functions 
            |       (.shape = (N+1, number of wavelengths); 
            |        first axis are wavelengths)
        :ref_type:
            | None or str or ndarray, optional
            | Specifies type of reference illuminant type.
            |     - None: defaults to metric_specific reference illuminant in 
            |             accordance with cri_type.
            |     - str: 'BB' : Blackbody radiatiors, 
            |            'DL': daylightphase, 
            |            'ciera': used in CIE CRI-13.3-1995, 
            |            'cierf': used in CIE 224-2017, 
            |            'iesrf': used in TM30-15, ...
            |     - ndarray: user defined reference SPD
        :cieobs:
            | None or dict, optional
            | Specifies which CMF sets to use for the calculation of the sample 
            | XYZs and the CCT (for reference illuminant calculation).
            | None defaults to the one specified in :cri_type: dict.    
            |     - key: 'xyz': str specifying CMF set for calculating xyz 
            |                   of samples and white 
            |     - key: 'cct': str specifying CMF set for calculating cct
        :cspace:
            | None or dict, optional
            | Specifies which color space to use.
            | None defaults to the one specified in  :cri_type: dict.  
            |     - key: 'type': str specifying color space used to calculate 
            |                    color differences in.
            |     - key: 'xyzw': None or ndarray with white point of color space
            |            If None: use xyzw of test / reference (after chromatic 
            |                     adaptation, if specified)
            |     - other keys specify other possible parameters needed for color
            |       space calculation, 
            |       see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf:
            | None or dict, optional
            | Perform explicit CAT before converting to color space coordinates.
            |    - None: don't apply a cat (other than perhaps the one built 
            |            into the colorspace) 
            |    - dict: with CAT parameters:
            |        - key: 'D': ndarray with degree of adaptation
            |        - key: 'mcat': ndarray with sensor matrix specification
            |        - key: 'xyzw': None or ndarray with white point
            |              None: use xyzw of reference otherwise transform both 
            |                    test and ref to xyzw
        :cri_specific_pars:
            | None or dict, optional
            | Specifies other parameters specific to type of cri 
            | (e.g. maxC for CQS calculations)
            |     - None: default to the one specified in  :cri_type: dict. 
            |     - dict: user specified parameters. 
            |         For its use, see for example:
            |             luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars']
        :rg_pars: 
            | None or dict, optional
            | Dict containing specifying parameters for slicing the gamut.
            | Dict structure: 
            |     {'nhbins' : None, 'start_hue' : 0, 'normalize_gamut' : True}
            |    - key: 'nhbins': int, number of hue bins to slice gamut 
            |                 (None use the one specified in :cri_type: dict).
            |    - key: 'start_hue': float (°), hue at which to start slicing
            |    - key: 'normalize_gamut': True or False: 
            |                normalize gamut or not before calculating a gamut 
            |                area index Rg. 
            |    - key: 'normalized_chroma_ref': 100.0 or float, optional
            |                Controls the size (chroma/radius) 
            |                of the normalization circle/gamut.
        :avg: 
            | None or fcn handle, optional
            | Averaging function (handle) for color differences, DEi 
            | (e.g. numpy.mean, .math.rms, .math.geomean)
            | None use the one specified in :cri_type: dict.
        :scale:
            | None or dict, optional
            | Specifies scaling of color differences to obtain CRI.
            |     - None use the one specified in :cri_type: dict.
            |     - dict: user specified dict with scaling parameters.
            |         - key: 'fcn': function handle to type of cri scale, 
            |                 e.g. 
            |                 * linear()_scale --> (100 - scale_factor*DEi), 
            |                 * log_scale --> (cfr. Ohno's CQS), 
            |                 * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
            |        - key: 'cfactor': factors used in scaling function, 
            |              If None: 
            |                     Scaling factor value(s) will be optimized to 
            |                     minimize the rms between the Rf's of the 
            |                     requested metric and the target metric specified
            |                     in:
            |
            |                  - key: 'opt_cri_type':  str 
            |                      * str: one of the preset _CRI_DEFAULTS
            |                      * dict: user speciied 
            |                      (dict must contain all keys as normal)
            |                     Note that if key not in :scale: dict, 
            |                     then 'opt_cri_type' is added with default 
            |                     setting = 'ciera'.
            |                 - key: 'opt_spd_set': ndarray with set of light 
            |                     source spds used to optimize cfactor. 
            |                     Note that if key not in :scale: dict, 
            |                     then default = 'F1-F12'.
        :opt_scale: 
            | True or False, optional
            | True: optimize scaling-factor, else do nothing and use value of 
              scaling-factor in :scale: dict.   
    
    Returns:
        :returns: 
            | float or ndarray with Rf for :out: 'Rf'
            | Other output is also possible by changing the :out: str value.
            
    References:
        1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `CIE224:2017. CIE 2017 Colour Fidelity Index for accurate scientific use. 
        Vienna, Austria: CIE. (2017).
        <http://www.cie.co.at/index.php?i_ca_id=1027>`_
        
        4. `Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
        CRI2012: A proposal for updating the CIE colour rendering index. 
        Lighting Research and Technology, 45, 689–709. 
        <http://lrt.sagepub.com/content/45/6/689>`_
        
        5. `CIE13.3-1995. Method of Measuring and Specifying 
        Colour Rendering Properties of Light Sources 
        (Vol. CIE13.3-19). Vienna, Austria: CIE. (1995).
        <http://www.cie.co.at/index.php/index.php?i_ca_id=303>`_
                    

    """
    outlist = out.split(',')
    
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)

    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_cri')
    
    # unpack some keys:
    if (opt_scale_factor is None) | (opt_scale_factor is False):
        scale_factor = cri_type['scale']['cfactor']
    scale_fcn = cri_type['scale']['fcn']
    avg = cri_type['avg']  
    
    # Input parsing: optimize scale_factor for input based on F1-F12 (default) if scale_factor is NaN or None:
    scale_factor = optimize_scale_factor(cri_type,opt_scale_factor, scale_fcn, avg)

    if np.isnan(scale_factor).any():
        raise Exception ('Unable to optimize scale_factor.')

    # A. get DEi of for ciera and of requested cri metric for spds in or specified by scale_factor_optimization_spds':
    DEi, jabt, jabr, cct, duv = spd_to_DEi(SPD, out = 'DEi,jabt,jabr,cct,duv', cri_type = cri_type)
    
    # B. convert DEi to color rendering index:
    Rfi = scale_fcn(DEi,scale_factor)
    Rf = np2d(scale_fcn(avg(DEi,axis = 0),scale_factor))
    
    # C. get binned jabt jabr and DEi:
    if ('Rg' in outlist) | ('Rfhi' in outlist) | ('Rhshi' in outlist) | ('Rcshi' in outlist):
        # calculate gamut area index:
        rg_pars = cri_type['rg_pars'] 
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
        Rg, jabt_binned, jabr_binned, DEi_binned = jab_to_rg(jabt,jabr, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, out = 'Rg,jabt,jabr,DEi')
    else:
        jabt_binned, jabr_binned, DEi_binned = None, None, None

    # D. Calculate Rfhi, Rhshi and Rcshi:
    if ('Rfhi' in outlist) | ('Rhshi' in outlist) | ('Rcshi' in outlist):
        Rfhi, Rcshi, Rhshi = jab_to_rhi(jabt = jabt_binned[:-1,...], jabr = jabr_binned[:-1,...], DEi = DEi_binned[:-1,...], cri_type = cri_type, scale_factor = scale_factor, scale_fcn = scale_fcn, use_bin_avg_DEi = True) # [:-1,...] removes last row from jab as this was added to close the gamut. 


    if (out == 'Rf'):
        return Rf
    elif (out == 'Rg'):
        return Rg
    else:
        return eval(out)

