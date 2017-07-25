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
            if ccts1[i]>= 3000:
                if i%2 == 0:
                    plt.plot(x[i],y[i],'k+', color = '0.5')
                    plt.text(x[i]*1.05,y[i]*0.95,'{:1.0f}K'.format(ccts1[i]), color = '0.5')
        plt.plot(x[-1],y[-1],'k+', color = '0.5')
        plt.text(x[-1]*1.05,y[-1]*0.95,'{:1.3e}K'.format(ccts[-1]), color = '0.5')    
    if show == False:
        return axh
    
def plotSL(cieobs =_cieobs, cspace = _cspace,  DL = True, BBL = True, D65 = False, EEW = False, cctlabels = False, axh = None, show = True, cspace_pars = {}, formatstr = 'k-', **kwargs):
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
        plotBB(ccts = None, cieobs = cieobs, cspace = cspace, axh = axh, show = show, cspace_pars = cspace_pars, cctlabels = cctlabels, formatstr = 'r-.',  **kwargs)
    
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
        
        
def plotceruleanline(cieobs = lx._cieobs, cspace = 'Yuv', axh = None,formatstr = 'ko-'):
    """
    Plot cerulean (yellow (577 nm) - blue (472 nm)) line (Kuehni, CRA, 2013: Table II: spectral lights).
    """
    cmf = _cmf['bar'][cieobs]
    p_y = cmf[0] == 577 #Kuehni, CRA 2013 (mean, table IV)
    p_b = cmf[0] == 472 #Kuehni, CRA 2013 (mean, table IV)
    xyz_y = cmf[1:,p_y].T
    xyz_b = cmf[1:,p_b].T
    lab = colortf(np.vstack((xyz_b,xyz_y)),cspace)
    if axh is None:
        axh = plt.gca()
    hcerline = axh.plot(lab[:,1],lab[:,2],formatstr,label = 'Cerulean line')    
    return hcerline

    
def plotUH(xyz0 = None, uhues = [0,1,2,3], cieobs = _cieobs, cspace = 'Yuv', axh = None,formatstr = ['yo-.','bo-.','ro-.','go-.'], excludefromlegend = ''):
    """ 
    Plot unique hue line from centerpoint xyz0 (Kuehni, CRA, 2013: uY,uB,uG: Table II: spectral lights; uR: Table IV: Xiao data)
    """
    hues = ['yellow','blue','red','green']
    cmf = _cmf['bar'][cieobs]
    p_y = cmf[0] == 577 #unique yellow,#Kuehni, CRA 2013 (mean, table IV: spectral data)
    p_b = cmf[0] == 472 #unique blue,Kuehni, CRA 2013 (mean, table IV: spectral data)
    p_g = cmf[0] == 514 #unique green, Kuehni, CRA 2013 (mean, table II: spectral data)
    p_r = cmf[0] == 650 #unique red, Kuehni, CRA 2013 (Xiao data, table IV: display data)
    xyz_y = 100*cmf[1:,p_y].T
    xyz_b = 100*cmf[1:,p_b].T
    xyz_g = 100*cmf[1:,p_g].T
    xyz_r = 100*cmf[1:,p_r].T
    xyz_uh = np.vstack((xyz_y,xyz_b,xyz_r,xyz_g))
    huniquehues = []
    if xyz0 is None:
        xyz0 = np.array([100,100,100])
    if axh is None:
        axh = plt.gca()
    for huenr in uhues:
        lab = colortf(np.vstack((xyz0,xyz_uh[huenr])),cspace)
        huh = axh.plot(lab[:,1],lab[:,2],formatstr[huenr],label = excludefromlegend + 'Unique '+ hues[huenr])
        huniquehues = [huniquehues,huh]
    return  huniquehues