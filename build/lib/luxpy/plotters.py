# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:44:45 2017

@author: kevin.smet
"""

###################################################################################################
# functions related to plotting 
###################################################################################################
#
# plot_color_data(): Plot color data (local helper function)
#
# plotDL(): Plot daylight locus (for 'ccts', default = 4000 K to 1e19 K) for 'cieobs' in 'cspace'.
#
# plotBB(): Plot blackbody locus (for 'ccts', default = 4000 K to 1e19 K) for 'cieobs' in 'cspace'.
#
# plotSL(): Plot spectrum locus for 'cieobs' in 'cspace'. plotBB and plotDL are also called, but can be turned off.
#
#--------------------------------------------------------------------------------------------------
from luxpy import *
import matplotlib.pyplot as plt 
__all__ = ['plotSL','plotDL','plotBB','plot_color_data']



def plot_color_data(x,y,z=None, axh=None, show = True, cieobs =_cieobs, cspace = _cspace,  formatstr = 'k-', **kwargs):
    """
    Plot data.
    """

    if 'grid' in kwargs.keys():
        plt.grid(kwargs['grid']);kwargs.pop('grid')
    if z is not None:
        plt.plot(x,y,z,formatstr, linewidth = 2)
        plt.xlabel(_cspace_axes[cspace][0], kwargs)
    else:
        plt.plot(x,y,formatstr,linewidth = 2)
        
    plt.xlabel(_cspace_axes[cspace][1], kwargs)
    plt.ylabel(_cspace_axes[cspace][2], kwargs)

    if show == True:
        plt.show()
    else:
        return plt.gca()



def plotDL(ccts = None, cieobs =_cieobs, cspace = _cspace, axh = None, show = True, force_daylight_below4000K = False, cspace_pars = {}, formatstr = 'k-',  **kwargs):
    """
    Plot daylight locus (for ccts, default = 4000 K to 1e19 K) for cieobs in cspace.
    """
    if ccts is None:
        ccts = 10**np.linspace(np.log10(4000.0),np.log10(10.0**19),100)
        
    xD,yD = daylightlocus(ccts, force_daylight_below4000K = force_daylight_below4000K)
    Y = 100*np.ones(xD.shape)
    DL =  Yxy_to_xyz(np.vstack((Y, xD,yD)).T)
    DL = colortf(DL, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(DL)
    
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    
    
    if show == False:
        return axh
    
def plotBB(ccts = None, cieobs =_cieobs, cspace = _cspace, axh = None, cctlabels = True, show = True, cspace_pars = {}, formatstr = 'k-',  **kwargs):  
    """
    Plot blackbody locus (for ccts) for cieobs in cspace.
    """
    if ccts is None:
        ccts1 = np.array([1000,1500,2000,2500,3000,3500,4000,5000,6000,8000,10000])
        ccts2 = 10**np.linspace(np.log10(15000.0),np.log10(10.0**19),100)
        ccts = np.hstack((ccts1,ccts2))
    else:
        ccts1 = None
    
    BB = cri_ref(ccts,ref_type='BB')
    xyz = spd_to_xyz(BB,cieobs = cieobs)
    Yxy = colortf(xyz, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(Yxy)
   
    axh = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show=show, formatstr=formatstr, **kwargs)    

    if (cctlabels == True) & (ccts1 is not None):
        for i in range(ccts1.shape[0]):
            if ccts1[i]>= 1000:
                if i%2 == 0:
                    plt.plot(x[i],y[i],'k+', color = '0.5')
                    plt.text(x[i]*1.05,y[i]*0.95,'{:1.0f}K'.format(ccts1[i]), color = '0.5')
        plt.plot(x[-1],y[-1],'k+', color = '0.5')
        plt.text(x[-1]*1.05,y[-1]*0.95,'{:1.3e}K'.format(ccts[-1]), color = '0.5')    
    if show == False:
        return axh
    
def plotSL(cieobs =_cieobs, cspace = _cspace,  DL = True, BBL = True, D65 = False, EEW = False, axh = None, show = True, cspace_pars = {}, formatstr = 'k-', **kwargs):
    """
    Plot spectrum locus for cieobs in cspace.
    """
    SL = _cmf['bar'][cieobs][1:4].T
    SL = np.vstack((SL,SL[0]))
    SL = 100*SL/SL[:,1,None]
    SL = colortf(SL, tf = cspace, tfa0 = cspace_pars)
    Y,x,y = asplit(SL)
    
    showcopy = show
    if np.any([DL,BBL,D65,EEW]):
        show = False

        
    axh_ = plot_color_data(x,y,axh = axh, cieobs = cieobs, cspace = cspace, show = show, formatstr=formatstr,  **kwargs)
    
    if DL == True:
        plotDL(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, formatstr = 'b:',  **kwargs)
    if BBL == True:
        plotBB(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, formatstr = 'r-.',  **kwargs)
    
    if D65 == True:
        YxyD65 = colortf(spd_to_xyz(_cie_illuminants['D65']), tf = cspace, tfa0 = cspace_pars)
        plt.plot(YxyD65[...,1],YxyD65[...,2],'bo')
    if EEW == True:
        YxyEEW = colortf(spd_to_xyz(_cie_illuminants['E']), tf = cspace, tfa0 = cspace_pars)
        plt.plot(YxyEEW[...,1],YxyEEW[...,2],'ko')
        
    if showcopy == False:
        return axh_
    else:
        plt.show()