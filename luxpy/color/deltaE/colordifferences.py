# -*- coding: utf-8 -*-
"""
Module for color difference calculations
========================================

 :deltaH(): Compute a hue difference, dH = 2*C1*C2*sin(dh/2)
 
 :_process_DEi(): Process color difference input DEi for output (helper fnc).

 :DE_camucs(): Calculate color appearance difference DE using camucs type model.

 :DE_2000(): Calculate DE2000 color difference.

 :DE_cspace():  Calculate color difference DE in specific color space.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import cam, _CSPACE, colortf, xyz_to_lab
from luxpy.utils import np, np2d


__all__ = ['deltaH', 'DE_camucs', 'DE2000','DE_cspace']

def deltaH(h1, C1, h2 = None, C2 = None, htype = 'deg'):
    """
    Compute a hue difference, dH = 2*C1*C2*sin(dh/2)
    
    Args:
        :h1:
            | hue for sample 1 (or hue difference if h2 is None)
        :C1: 
            | chroma of sample 1 (or prod C1*C2 if C2 is None)
        :h2: 
            | hue angle of sample 2 (if None, then h1 contains a hue difference)
        :C2: 
            | chroma of sample 2
        :htype: 
            | 'deg' or 'rad', optional
            |   - 'deg': hue angle between 0° and 360°
            |   - 'rad': hue angle between 0 and 2pi radians
    
    Returns:
        :returns:
            | ndarray of deltaH values.
    """
    return cam.deltaH(h1, C1, h2 = h2, C2 = C2, htype = htype)

def _process_DEi(DEi, DEtype = 'jab', avg = None, avg_axis = 0, out = 'DEi'):
    """
    Process color difference input DEi for output (helper function).
    
    Args:
        :DEi: 
            | tuple(J ndarray, ab ndarray).
        :DEtype:
            | 'jab' or str, optional
            | Options: 
            |    - 'jab' : calculates full color difference over all 3 dimensions.
            |    - 'ab'  : calculates chromaticity difference.
            |    - 'j'   : calculates lightness or brightness difference 
            |             (depending on :out:).
            |    - 'j,ab': calculates both 'j' and 'ab' options 
            |              and returns them as a tuple.
        :avg:
            | None, optional
            | None: don't calculate average DE, 
            |       otherwise use function handle in :avg:.
        :avg_axis:
            | axis to calculate average over, optional
        :out: 
            | 'DEi' or str, optional
            | Requested output.
        
    Note:
        For the other input arguments, see specific color space used.
        
    Returns:
        :returns: 
            | ndarray with DEi [, DEa] or other as specified by :out:
    """

    if (DEi[0].shape[-1] == 1) & (DEi[0].ndim==3):
        DEi = tuple((map(lambda x: np.squeeze(x, axis = x.ndim-1),DEi)))
    
    # Calculate correct type of DE:
    if DEtype == 'jab':
        DEi = np.sqrt(DEi[0] + DEi[1])
    elif DEtype == 'ab':
        DEi = np.sqrt(DEi[1])
    elif DEtype == 'j':
        DEi = np.sqrt(DEi[0])
    
    # Calculate average when requested:
    if (avg is not None) & ('DEa' in out.split(',')):
        if isinstance(DEi, tuple):
            DEa = (avg(DEi[0],axis = avg_axis, keepdims = True), avg(DEi[1],axis = avg_axis, keepdims = True))
        else:
            DEa = avg(DEi,axis = avg_axis, keepdims = True)


    if out == 'DEi':
        return DEi
    elif out == 'DEi,DEa':
        return DEi, DEa
    else:
        return eval(out)


def DE_camucs(xyzt, xyzr, DEtype = 'jab', avg = None, avg_axis = 0, out = 'DEi',
              xyzwt = cam._CAM_DEFAULT_WHITE_POINT, xyzwr = cam._CAM_DEFAULT_WHITE_POINT, \
              Ywt = None, conditionst = cam._CAM_DEFAULT_CONDITIONS,\
              Ywr = None, conditionsr = cam._CAM_DEFAULT_CONDITIONS,\
              camtype = cam._CAM_DEFAULT_TYPE, ucstype = 'ucs', mcat = None, \
              outin = 'J,aM,bM', yellowbluepurplecorrect = False, **kwargs):
    
    """
    Calculate color appearance difference DE using camucs type model.
    
    Args:
        :xyzt: 
            | ndarray with tristimulus values of test data.
        :xyzr:
            | ndarray with tristimulus values of reference data.
        :DEtype:
            | 'jab' or str, optional
            | Options: 
            |    - 'jab' : calculates full color difference over all 3 dimensions.
            |    - 'ab'  : calculates chromaticity difference.
            |    - 'j'   : calculates lightness or brightness difference 
            |             (depending on :outin:).
            |    - 'j,ab': calculates both 'j' and 'ab' options 
            |              and returns them as a tuple.
        :avg:
            | None, optional
            | None: don't calculate average DE, 
            |       otherwise use function handle in :avg:.
        :avg_axis:
            | axis to calculate average over, optional
        :out: 
            | 'DEi' or str, optional
            | Requested output.
        :camtype: 
            | luxpy.cam._CAM_DEFAULT_TYPE, optional
            | Str specifier for CAM type to use, options: 'ciecam02' or 'ciecam16'.
        :ucstype:
            | 'ucs' or 'lcd' or 'scd', optional
            | Str specifier for which type of color attribute compression 
            |  parameters to use:
            |   -'ucs': uniform color space, 
            |   -'lcd': large color differences, 
            |   -'scd': small color differences

    Note:
        For the other input arguments, see ?luxpy.cam.camucs_structure.
        
    Returns:
        :returns: 
            | ndarray with DEi [, DEa] or other as specified by :out:
    """
    
    
    jabt = cam.camXucs(xyzt, xyzw = xyzwt, Yw = Ywt, conditions = conditionst,\
                       camtype = camtype, ucstype = ucstype, mcat = mcat, forward = True, \
                       outin = outin, yellowbluepurplecorrect = yellowbluepurplecorrect)
    
    jabr = cam.camXucs(xyzr, xyzw = xyzwr, Yw = Ywr, conditions = conditionsr,\
                       camtype = camtype, ucstype = ucstype, mcat = mcat, forward = True, \
                       outin = outin, yellowbluepurplecorrect = yellowbluepurplecorrect)
    
    
    KL, c1, c2 = [cam._CAM_UCS_PARAMETERS[camtype][ucstype][x] for x in sorted(cam._CAM_UCS_PARAMETERS[camtype][ucstype].keys())]

    # Calculate color difference and take account of KL:
    DEi = ((((jabt[...,0:1]-jabr[...,0:1])/KL)**2).sum(axis = jabt[...,0:1].ndim - 1, keepdims = True),\
               ((jabt[...,1:3]-jabr[...,1:3])**2).sum(axis = jabt[...,1:3].ndim - 1, keepdims = True))
    
    return _process_DEi(DEi, DEtype = DEtype, avg = avg, avg_axis = avg_axis, out = out)
    



def DE2000(xyzt, xyzr, dtype = 'xyz', DEtype = 'jab', avg = None, avg_axis = 0, out = 'DEi',
              xyzwt = None, xyzwr = None, KLCH = None):
    
    """
    Calculate DE2000 color difference.
    
    Args:
        :xyzt: 
            | ndarray with tristimulus values of test data.
        :xyzr:
            | ndarray with tristimulus values of reference data.
        :dtype:
            | 'xyz' or 'lab', optional
            | Specifies data type in :xyzt: and :xyzr:.
        :xyzwt:
            | None or ndarray, optional
            |   White point tristimulus values of test data
            |   None defaults to the one set in lx.xyz_to_lab()
        :xyzwr:
            | None or ndarray, optional
            |    Whitepoint tristimulus values of reference data
            |    None defaults to the one set in lx.xyz_to_lab()
        :DEtype:
            | 'jab' or str, optional
            | Options: 
            |    - 'jab' : calculates full color difference over all 3 dimensions.
            |    - 'ab'  : calculates chromaticity difference.
            |    - 'j'   : calculates lightness or brightness difference 
            |             (depending on :outin:).
            |    - 'j,ab': calculates both 'j' and 'ab' options 
            |              and returns them as a tuple.
        :KLCH: 
            | None, optional
            | Weigths for L, C, H 
            | None: default to [1,1,1] 
        :avg:
            | None, optional
            | None: don't calculate average DE, 
            |       otherwise use function handle in :avg:.
        :avg_axis:
            | axis to calculate average over, optional
        :out: 
            | 'DEi' or str, optional
            | Requested output.
        
    Note:
        For the other input arguments, see specific color space used.
        
    Returns:
        :returns: 
            | ndarray with DEi [, DEa] or other as specified by :out:
            
    References:
        1. `Sharma, G., Wu, W., & Dalal, E. N. (2005). 
        The CIEDE2000 color‐difference formula: Implementation notes, 
        supplementary test data, and mathematical observations. 
        Color Research & Application, 30(1), 21–30. 
        <https://doi.org/10.1002/col.20070>`_
    """
    
    if KLCH is None:
        KLCH = [1,1,1]
    
    if dtype == 'xyz':
        labt = xyz_to_lab(xyzt, xyzw = xyzwt)
        labr = xyz_to_lab(xyzr, xyzw = xyzwr)
    else:
        labt = xyzt
        labr = xyzr
 
    Lt = labt[...,0:1]
    at = labt[...,1:2]
    bt = labt[...,2:3]
    Ct = np.sqrt(at**2 + bt**2)
    #ht = cam.hue_angle(at,bt,htype = 'rad')
    
    Lr = labr[...,0:1]
    ar = labr[...,1:2]
    br = labr[...,2:3]
    Cr = np.sqrt(ar**2 + br**2)
    #hr = cam.hue_angle(at,bt,htype = 'rad')
    
    # Step 1:
    Cavg = (Ct + Cr)/2
    G = 0.5*(1 - np.sqrt((Cavg**7.0)/((Cavg**7.0) + (25.0**7))))
    apt = (1 + G)*at
    apr = (1 + G)*ar
    
    Cpt = np.sqrt(apt**2 + bt**2)
    Cpr = np.sqrt(apr**2 + br**2)
    Cpprod = Cpt*Cpr


    hpt = cam.hue_angle(apt,bt, htype = 'deg')
    hpr = cam.hue_angle(apr,br, htype = 'deg')
    hpt[(apt==0)*(bt==0)] = 0
    hpr[(apr==0)*(br==0)] = 0
    
    # Step 2:
    dL = np.abs(Lr - Lt)
    dCp = np.abs(Cpr - Cpt)
    dhp_ = hpr - hpt  

    dhp = dhp_.copy()
    dhp[np.where(np.abs(dhp_) > 180)] = dhp[np.where(np.abs(dhp_) > 180)] - 360
    dhp[np.where(np.abs(dhp_) < -180)] = dhp[np.where(np.abs(dhp_) < -180)] + 360
    dhp[np.where(Cpprod == 0)] = 0

    #dH = 2*np.sqrt(Cpprod)*np.sin(dhp/2*np.pi/180)
    dH = deltaH(dhp, Cpprod, htype = 'deg')

    # Step 3:
    Lp = (Lr + Lt)/2
    Cp = (Cpr + Cpt)/2
    
    hps = hpt + hpr
    hp = (hpt + hpr)/2
    hp[np.where((np.abs(dhp_) > 180) & (hps < 360))] = hp[np.where((np.abs(dhp_) > 180) & (hps < 360))] + 180
    hp[np.where((np.abs(dhp_) > 180) & (hps >= 360))] = hp[np.where((np.abs(dhp_) > 180) & (hps >= 360))] - 180
    hp[np.where(Cpprod == 0)] = 0

    T = 1 - 0.17*np.cos((hp - 30)*np.pi/180) + 0.24*np.cos(2*hp*np.pi/180) +\
        0.32*np.cos((3*hp + 6)*np.pi/180) - 0.20*np.cos((4*hp - 63)*np.pi/180)
    dtheta = 30*np.exp(-((hp-275)/25)**2)
    RC = 2*np.sqrt((Cp**7)/((Cp**7) + (25**7)))
    SL = 1 + ((0.015*(Lp-50)**2)/np.sqrt(20 + (Lp - 50)**2))
    SC = 1 + 0.045*Cp
    SH = 1 + 0.015*Cp*T
    RT = -np.sin(2*dtheta*np.pi/180)*RC

    kL, kC, kH = KLCH
    
    DEi = ((dL/(kL*SL))**2 , (dCp/(kC*SC))**2 + (dH/(kH*SH))**2 + RT*(dCp/(kC*SC))*(dH/(kH*SH)))
    

    return _process_DEi(DEi, DEtype = DEtype, avg = avg, avg_axis = avg_axis, out = out)

def DE_cspace(xyzt, xyzr, dtype = 'xyz', tf = _CSPACE, DEtype = 'jab', avg = None, avg_axis = 0, out = 'DEi',
              xyzwt = None, xyzwr = None, fwtft = {}, fwtfr = {}, KLCH = None,\
              camtype = cam._CAM_DEFAULT_TYPE, ucstype = 'ucs'):
    
    """
    Calculate color difference DE in specific color space.
    
    Args:
        :xyzt: 
            | ndarray with tristimulus values of test data.
        :xyzr:
            | ndarray with tristimulus values of reference data.
        :dtype:
            | 'xyz' or 'jab', optional
            | Specifies data type in :xyzt: and :xyzr:.
        :xyzwt:
            | None or ndarray, optional
            |   White point tristimulus values of test data
            |   None defaults to the one set in :fwtft: 
            |   or else to the default of cspace.
        :xyzwr:
            | None or ndarray, optional
            |   Whitepoint tristimulus values of reference data
            |    None defaults to the one set in non-empty :fwtfr: 
            |    or else to default of cspace.
        :tf:
            | _CSPACE, optional
            | Color space to use for color difference calculation.
        :fwtft:
            | {}, optional
            | Dict with parameters for forward transform 
              from xyz to cspace for test data.
        :fwtfr: 
            | {}, optional 
            | Dict with parameters for forward transform 
            | from xyz to cspace for reference data.
        :KLCH:
            | None, optional
            | Weigths for L, C, H 
            | None: default to [1,1,1] 
            | KLCH is not used when tf == 'camucs'.
        :DEtype:
            | 'jab' or str, optional
            | Options: 
            |    - 'jab' : calculates full color difference over all 3 dimensions.
            |    - 'ab'  : calculates chromaticity difference.
            |    - 'j'   : calculates lightness or brightness difference 
            |             (depending on :outin:).
            |    - 'j,ab': calculates both 'j' and 'ab' options 
            |              and returns them as a tuple.
        :avg:
            | None, optional
            | None: don't calculate average DE, 
            |       otherwise use function handle in :avg:.
        :avg_axis:
            | axis to calculate average over, optional
        :out: 
            | 'DEi' or str, optional
            | Requested output.
        :camtype: 
            | luxpy.cam._CAM_DEFAULT_TYPE, optional
            | Str specifier for CAM type to use, options: 'ciecam02' or 'ciecam16'.
            | Only when DEtype == 'camucs'.
        :ucstype:
            | 'ucs' or 'lcd' or 'scd', optional
            | Str specifier for which type of color attribute compression 
            | parameters to use:
            |     -'ucs': uniform color space,
            |     -'lcd', large color differences,
            |     -'scd': small color differences
            | Only when DEtype == 'camucs'.
        
    Note:
        For the other input arguments, see specific color space used.
        
    Returns:
        :returns: 
            | ndarray with DEi [, DEa] or other as specified by :out:
    """
    
    # Get xyzw from dict if xyzw is None & dict is Not None
    if xyzwr is not None:
        fwtfr['xyzw'] = xyzwr
    else:
        if bool(fwtfr):
            xyzwr = fwtfr['xyzw']
    if xyzwt is not None:
        fwtft['xyzw'] = xyzwt
    else:
        if bool(fwtft):
            xyzwt = fwtft['xyzw']
    
    
    if tf == 'camucs':
        if dtype == 'xyz':
            if fwtfr['xyzw'] is None:
                fwtfr['xyzw'] = cam._CAM_DEFAULT_WHITE_POINT
            if fwtft['xyzw'] is None:
                fwtft['xyzw'] = cam._CAM_DEFAULT_WHITE_POINT
            jabt = cam.camXucs(xyzt, camtype = camtype, ucstype = ucstype, **fwtft)
            jabr = cam.camXucs(xyzr, camtype = camtype, ucstype = ucstype, **fwtfr)

        
        else:
            jabt = xyzt
            jabr = xyzr
            
        KL, c1, c2 = [cam._CAM_UCS_PARAMETERS[camtype][ucstype][x] for x in sorted(cam._CAM_UCS_PARAMETERS[camtype][ucstype].keys())]

        # Calculate color difference and take account of KL:
        DEi = ((((jabt[...,0:1]-jabr[...,0:1])/KL)**2).sum(axis = jabt[...,0:1].ndim - 1, keepdims = True),\
                   ((jabt[...,1:3]-jabr[...,1:3])**2).sum(axis = jabt[...,1:3].ndim - 1, keepdims = True))
    
    elif (tf == 'DE2000') | (tf == 'DE00'):
        return DE2000(xyzt, xyzr, dtype = 'xyz', DEtype = DEtype, avg = avg,\
               avg_axis = avg_axis, out = out,
              xyzwt = xyzwt, xyzwr = xyzwr, KLCH = KLCH)

        
    else:
        if dtype == 'xyz':
            # Use colortf:
            jabt = colortf(xyzt, tf = tf, fwtf = fwtft)
            jabr = colortf(xyzr, tf = tf, fwtf = fwtfr)  
        else:
            jabt = xyzt
            jabr = xyzr
    
        if (KLCH == None) | (KLCH == [1,1,1]):
            # Calculate color difference and take account of KL:
            DEi = (((jabt[...,0:1]-jabr[...,0:1])**2).sum(axis = jabt[...,0:1].ndim - 1, keepdims = True),\
                   ((jabt[...,1:3]-jabr[...,1:3])**2).sum(axis = jabt[...,1:3].ndim - 1, keepdims = True))
        
        else: #using LCH specification for use with KLCH weights:
            Jt = jabt[...,0:1]
            at = jabt[...,1:2]
            bt = jabt[...,2:3]
            Ct = np.sqrt(at**2 + bt**2)
            ht = cam.hue_angle(at,bt,htype = 'rad')
            
            Jr = jabr[...,0:1]
            ar = jabr[...,1:2]
            br = jabr[...,2:3]
            Cr = np.sqrt(ar**2 + br**2)
            hr = cam.hue_angle(at,bt,htype = 'rad')
            
            dJ = Jt - Jr
            dC = Ct - Cr
            dH = ht - hr
            DEab2 = ((at-ar)**2 + (bt-br)**2)
            dH = np.sqrt(DEab2 - dC**2)
            
            DEi = ((dJ/KLCH[0])**2, (dC/KLCH[1])**2 + (dH/KLCH[2])**2)
    
    return _process_DEi(DEi, DEtype = DEtype, avg = avg, avg_axis = avg_axis, out = out)


