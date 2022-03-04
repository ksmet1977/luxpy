# -*- coding: utf-8 -*-
"""
Define munsell angles so that 5Y is at 90° and red is in first quadrant
Created on Wed Nov  3 18:56:01 2021

@author: u0032318
"""
import luxpy as lx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

V = lx._RFL['munsell']['V'].astype(np.float)
C = lx._RFL['munsell']['C'].astype(np.float)
H = lx._RFL['munsell']['H']
h = lx._RFL['munsell']['h'].astype(np.float)
Hu = unique(H)
p5Y = np.where(Hu=='5Y')[0][0]
h5Y = 90
hu = np.array([h5Y - (p5Y-i)*p5Y for i in range(len(Hu))])
for i in range(len(Hu)):
    h[H==Hu[i]] = hu[i]

ab = np.round(np.hstack((C*np.cos(h*np.pi/180), C*np.sin(h*np.pi/180))),5)

array = np.hstack((H,V,C,h,ab))

pd.DataFrame(array,columns = ['H','V','C','h','a','b']).to_csv('Munsell1269NotationInfo2.dat',header = True,index=True)

colors = ['R','Y','G','B']
for color in colors:
    c = [bool(('5'+color in H[i][0])*(len(H[i][0]) == 2)) for i in range(len(H))]
    plt.plot(ab[c,0],ab[c,1],color.lower()+'o')