# -*- coding: utf-8 -*-
"""
Module with color utility functions
===================================

 :naka_rushton(): apply forward or inverse Naka-Rushton compression.

 :hue_angle(): Calculate positive hue angle (0°-360° or 0 - 2*pi rad.) from opponent signals a and b.

 :naka_rushton(): Apply a Naka-Rushton response compression (n) and an adaptive shift (sig)
 
 :hue_quadrature(): Get hue quadrature H from hue h.
 
     
Created on Wed Sep 30 15:43:49 2020

@author: ksmet1977 at gmail.com
"""
import numpy as np

from luxpy import math

__all__ = ['naka_rushton', 'hue_angle', 'deltaH', 'hue_quadrature']


def hue_angle(a,b, htype = 'deg'):
    """
    Calculate positive hue angle (0°-360° or 0 - 2*pi rad.) from opponent signals a and b.
    
    Args:
        :a: 
            | ndarray of a-coordinates
        :b: 
            | ndarray of b-coordinates
        :htype: 
            | 'deg' or 'rad', optional
            |   - 'deg': hue angle between 0° and 360°
            |   - 'rad': hue angle between 0 and 2pi radians
    Returns:
        :returns:
            | ndarray of positive hue angles.
    """
    return math.positive_arctan(a,b, htype = htype)


def naka_rushton(data, sig = 2.0, n = 0.73, scaling = 1.0, noise = 0.0, forward = True):
    """
    Apply a Naka-Rushton response compression (n) and an adaptive shift (sig).
    
    | NK(x) = sign(x) * scaling * ((abs(x)**n) / ((abs(x)**n) + (sig**n))) + noise
    
    Args:
        :data:
            | float or ndarray
        :sig: 
            | 2.0, optional
            | Semi-saturation constant. Value for which NK(:data:) is 1/2
        :n: 
            | 0.73, optional
            | Compression power.
        :scaling:
            | 1.0, optional
            | Maximum value of NK-function.
        :noise:
            | 0.0, optional
            | Cone excitation noise.
        :forward:
            | True, optional
            | True: do NK(x) 
            | False: do NK(x)**(-1).
    
    Returns:
        :returns: 
            | float or ndarray with NK-(de)compressed input :x:        
    """
    if forward:
        return np.sign(data)*scaling * ((np.abs(data)**n) / ((np.abs(data)**n) + (sig**n))) + noise
    elif forward == False:
        Ip =  sig*(((np.abs(np.abs(data)-noise))/(scaling-np.abs(np.abs(data)-noise))))**(1/n)
        if not np.isscalar(Ip):
            p = np.where(np.abs(data) < noise)
            Ip[p] = -Ip[p]
        else:
            if np.abs(data) < noise:
                Ip = -Ip
        return Ip
    
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
    if htype == 'deg':
        r2d = np.pi/180
    else:
        r2d = 1.0
    if h2 is not None:
        deltah = h1 - h2
    else:
        deltah = h1
    if C2 is not None:
        Cprod = C1 * C2
    else:
        Cprod = C1
    return 2*(Cprod)**0.5*np.sin(r2d*deltah/2)   

def hue_quadrature(h, unique_hue_data = None, forward = True):
    """
    Get hue quadrature H from hue h.
    
    Args:
        :h: 
            | float or ndarray [(N,) or (N,1)] with: 
            |   - hue angle data in degrees (!) if forward == True.
            |   - Hue quadrature data if forward = False
        :unique_hue data:
            | None or dict, optional
            |   - None: defaults to:
            |         {'hues': 'red yellow green blue red'.split(), 
            |        'i': np.arange(5.0), 
            |        'hi':[20.14, 90.0, 164.25,237.53,380.14],
            |        'ei':[0.8,0.7,1.0,1.2,0.8],
            |        'Hi':[0.0,100.0,200.0,300.0,400.0]}
            |   - dict: user specified unique hue data  
            |           (same structure as above)
        :forward:
            | True, optional
            | If true: input h is hue angle, else it is Hue quadrature
    
    Returns:
        :H: 
            | ndarray of Hue quadrature value(s) (forward == True) or of hue angle values(s) (foward == False).
    """
    
    if unique_hue_data is None:
        unique_hue_data = {'hues': 'red yellow green blue red'.split(), 
                           'i': [0,1,2,3,4], 
                           'hi':[20.14, 90.0, 164.25,237.53,380.14],
                           'ei':[0.8,0.7,1.0,1.2,0.8],
                           'Hi':[0.0,100.0,200.0,300.0,400.0]}
    
    ndim = np.array(h).ndim

    hi = unique_hue_data['hi']
    Hi = unique_hue_data['Hi']
    ei = unique_hue_data['ei']
    
    
    if forward == True:
        h = np.atleast_2d(h)
        h[h>360] -= 360.0
        h[h<hi[0]] += 360.0
        if h.shape[0] == 1:
            h = h.T
        
        H = np.zeros_like(h)
        for j in range(h.shape[1]):
            h_j = h[...,j:j+1]
            h_hi = np.repeat(h_j, repeats = len(hi), axis = 1)
            hi_h = np.repeat(np.atleast_2d(hi),repeats = h.shape[0], axis = 0)
            d = (h_hi - hi_h)
            d[d<0] = 100000.0
            p = d.argmin(axis = 1)
            p[p == (len(hi)-1)] = 0 # make sure last unique hue data is not selected
            H_j = np.array([Hi[pi] + (100.0/(1.0 + ((hi[pi+1] - h_j[i])/ei[pi+1])/(1e-308 + ((h_j[i]-hi[pi])-360.0*((h_j[i]-hi[pi])==360.0))/ei[pi]))) for (i,pi) in enumerate(p)])
            
            # for (i,pi) in enumerate(p):
            #     print('i,pi',i,pi)
            #     print('Hi[pi]',Hi[pi])
            #     print('h_j[i]-hi[pi]',h_j[i]-hi[pi])
            #     print('with bool',((h_j[i]-hi[pi])-360.0*((h_j[i]-hi[pi])==360)))
            #     print('ei[pi]',ei[pi])
            #     print('1/ei[pi]',1/ei[pi])
            #     print('(h_j[i]-hi[pi])/ei[pi]',(h_j[i]-hi[pi])/ei[pi])
            #     print('ei[pi+1]',ei[pi+1])
            #     print('1/ei[pi+1]',1/ei[pi+1])
            #     print('(hi[pi+1] - h_j[i])',(hi[pi+1] - h_j[i]))
            #     print('(hi[pi+1] - h_j[i])/ei[pi+1]',(hi[pi+1] - h_j[i])/ei[pi+1])
            #     print('((h_j[i]-hi[pi])/ei[pi] + (hi[pi+1] - h_j[i])/ei[pi+1])',((h_j[i]-hi[pi])/ei[pi] + (hi[pi+1] - h_j[i])/ei[pi+1]))
            #     print('100*...',(100.0/(1.0 + ((hi[pi+1] - h_j[i])/ei[pi+1])/((h_j[i]-hi[pi])/ei[pi]))))
            #     print('p1',1/(1e-308+(((h_j[i]-hi[pi])-360.0*((h_j[i]-hi[pi])==360.0))/ei[pi])))
            #     print(Hi[pi] + (100.0/(1.0 + ((hi[pi+1] - h_j[i])/ei[pi+1])/(1e-308 + ((h_j[i]-hi[pi])-360.0*((h_j[i]-hi[pi])==360.0))/ei[pi]))))
            
            H[...,j:j+1] = H_j
    
        if ndim == 0:
            return H[0][0]
        elif ndim == 1:
            return H[:,0]
        else:
            return H

    else:
        H = np.atleast_2d(h)
        H[H>=400] = H[H>=400] - 400
        if H.shape[0] == 1:
            H = H.T
        h = np.zeros_like(H)
        for j in range(H.shape[1]):
            H_j = H[...,j:j+1]
            H_Hi = np.repeat(H_j, repeats = len(Hi), axis = 1)
            Hi_H = np.repeat(np.atleast_2d(Hi),repeats = H.shape[0], axis = 0)
            d = (H_Hi - Hi_H)
            d[d<0] = 100000.0
            p = d.argmin(axis = 1)
            p[p == (len(Hi)-1)] = 0 # make sure last unique hue data is not selected
            h_j = np.array([((H_j[i] - Hi[pi])*(ei[pi+1]*hi[pi] - ei[pi]*hi[pi+1]) - 100*ei[pi+1]*hi[pi])/((H_j[i] - Hi[pi])*(ei[pi+1] - ei[pi]) - 100*ei[pi+1]) for (i,pi) in enumerate(p)])
            h[...,j:j+1] = h_j
        h[h > 360 - hi[0]*1] -= 360.0
    
        if ndim == 0:
            return h[0][0]
        elif ndim == 1:
            return h[:,0]
        else:
            return h
        
if __name__ == '__main__':
    h = np.array([10.0,30,110,280,370, 380.13, 380.14, 381.15, 390])
    # h = np.array([380.14])
    print('h',h)
    H = hue_quadrature(h, unique_hue_data = None, forward = True)
    print('H',H)
    h2 = hue_quadrature(H, unique_hue_data = None, forward = False)
    print('h2',h2)
    H2 = hue_quadrature(h2, unique_hue_data = None, forward = True)
    print('H2',H2)
    
