# -*- coding: utf-8 -*-
"""
Module for some photometric conversions and other functions
-----------------------------------------------------------

 :L_to_td(): convert luminance to trolands 
 
 :td_to_L(): convert trolands to luminance 
 
 :L_to_Er(): convert luminance to retinal illuminance
 
 :Er_to_L(): convert retinal illuminance to luminance
 
 :get_pupil_diameter_watson2012(): Get pupil diameter from adapting luminance, adaptingArea (or Diameter) and observer age

Created on Sat Mar  4 13:38:47 2023

@author: u0032318
"""
import numpy as np 

__all__ = ['L_to_td', 'td_to_L', 'L_to_Er', 'Er_to_L',
           'get_pupil_diameter_watson2012']

def L_to_td(L, pa = 1, pd = None):
    """
    Luminance to Trolands: Td [td = µcd] = L[cd/m²] * pupil area [mm²]
    
    Args: 
        :L:
            | luminance [cd/m²]
        :pa: 
            | None, optional
            | pupil area in mm²
        :pd: 
            | None, optional
            | pupil diameter in mm
            | If None: pa is used.
            
    Returns:
        :td:
            | trolands [micro-cd]
    """
    if pd is not None: pa = np.pi*(pd/2)**2
    return L * pa 

def td_to_L(td, pa = 1, pd = None):
    """
    Trolands to Luminance to : L[cd/m²] = Td [td = µcd] / pupil area [mm²]
    
    Args: 
        :td:
            | trolands [micro-cd]
        :pa: 
            | None, optional
            | pupil area in mm²
        :pd: 
            | None, optional
            | pupil diameter in mm
            | If None: pa is used.
            
    Returns:
        :L:
            | luminance [cd/m²]
    """
    if pd is not None: pa = np.pi*(pd/2)**2
    return td / pa 

def L_to_Er(L, pa = 1, pd = None, EFL = 16.7, beta = 0):
    """
    Luminance to retinal illuminance: Er [lx] = pi/4 * L[cd/m²] * (pupil diameter [mm] / EyeFocalLength [mm])^2 * cos^4(beta)
    
    Args: 
        :L:
            | luminance [cd/m²]
        :pa: 
            | None, optional
            | pupil area in mm²
        :pd: 
            | None, optional
            | pupil diameter in mm
            | If None: pa is used.
        :EFL:
            | 16.7, optional
            | Eye-Focal-Length [mm] 
        :beta:
            | 0, optional
            | off-axis angle [°]
            
    Returns:
        :Er:
            | retinal illuminance [lux]
    """
    return (L_to_td(L, pa = pa, pd = pd)/(EFL**2))*np.cos(np.pi/180*beta)**4

def Er_to_L(Er, pa = 1, pd = None, EFL = 16.7, beta = 0):
    """
    Retinal illuminance to Luminance: L[cd/m²] = 4/pi* Er [lx] * (EyeFocalLength [mm] / pupil diameter [mm])^2 / cos^4(beta) 
    
    Args: 
        :Er:
            | retinal illuminance [lux]
        :pa: 
            | None, optional
            | pupil area in mm²
        :pd: 
            | None, optional
            | pupil diameter in mm
            | If None: pa is used.
        :EFL:
            | 16.7, optional
            | Eye-Focal-Length [mm] 
        :beta:
            | 0, optional
            | off-axis angle [°]
            
    Returns:
        :L:
            | luminance [cd/m²]
    """
    return td_to_L(Er*(EFL**2)/np.cos(np.pi/180*beta)**4, pa = pa, pd = pd)
            

#------------------------------------------------------------------------------
def get_pupil_diameter_watson2012(La, age = 32, adapArea = 10, adapDiam = None, nEyes = 2):
    """
    Get pupil diameter from adapting luminance, adaptingArea (or Diameter) and observer age.
    
    Args:
        :La: 
            | adapting luminance [cd/m²]
        :adapArea: 
            | 10, optional
            | adapting area [degree²]
        :adapDiam:
            | None, optional
            | adapting diameter [degrees]
            | If None: use adapArea else calculate from adapDiam
        :nEyes:
            | 2, optional
            | Correction for 1 or 2 eye viewing.
            
    Returns:
        :DU:
            | Pupil diameter in mm.
            
    References:
        1. `Watson, A. & Elliot, J. (2012) "A unified formula for light-adapted pupil size"
        Journal of Vision (2012) 12(10):12, 1–16 
        <http://www.journalofvision.org/content/12/10/12>`
    """
    Me = 0.1 if nEyes == 1 else 1
    age = np.atleast_2d(age).T
    La = np.atleast_2d(La)
    # get corneal flux density = product of luminance, area (° squared), and the monocular effect, F=LaM(e)
    a = adapArea if adapDiam is None else np.pi*(adapDiam/2)**2
    F = La*a*Me

    DSD = 7.75 - 5.75*((F/846)**0.41) / (((F/846)**0.41) + 2)
    S = 0.021323-0.0095623*DSD
    A = (age - 28.58)*S 
    DU = DSD + A
    
    # simplified forms (with age<20 extension):
    f = F**0.41
    DU = (18.5172 + 0.122165*f - 0.105569*age + 0.000138645*f*age) / (2 + 0.0630635*f)
    DU2 = (16.4674 + np.exp(-0.208269*age+(-3.96868 + 0.00521209 *f)) + 0.124857 * f) / (2 + 0.0630635*f)
    DU[age[:,0]<20] = DU2[age[:,0]<20]
    return DU
    