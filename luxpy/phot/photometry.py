# -*- coding: utf-8 -*-
"""
Module for some photometric conversions and other functions
-----------------------------------------------------------

 :L_to_td(): convert luminance to trolands 
 
 :td_to_L(): convert trolands to luminance 
 
 :L_to_Er(): convert luminance to retinal illuminance
 
 :Er_to_L(): convert retinal illuminance to luminance

Created on Sat Mar  4 13:38:47 2023

@author: u0032318
"""
import numpy as np 

__all__ = ['L_to_td', 'td_to_L', 'L_to_Er', 'Er_to_L']

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
            

    