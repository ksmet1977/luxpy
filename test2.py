# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:44:24 2017

@author: kevin.smet

"""
import luxpy as lx
import timeit
import numpy as np

def func(n):
    for i in range(1):
        lx.spd_to_xyz(lx._iestm30['S']['data'][:(n+1)],rfl = rfl)
    return True


def wrapper(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)
    return wrapper

xi=[]
for i in range(300): 
    n=i; 
    wrapped = wrapper(func, n);
    xj = []
    for j in range(10): 
        x = timeit.timeit(wrapped, number=1);
        xj = np.hstack((xj,float(x)));
    xi = np.hstack((xi,xj.min()))