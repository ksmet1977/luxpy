# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:52:28 2017

@author: kevin.smet
"""
###############################################################################
# functions related to correlated color temperature calculations
###############################################################################
#
# _cct_lut_dir: Folder with Look-Up-Tables (LUT) for correlated color temperature calculation followings Ohno's method.
#
# _cct_LUT: Dict with LUT.
#
# xyz_to_cct(): Calculates CCT,Duv from XYZ, wrapper for ..._ohno() & ..._search()
#
# xyz_to_duv(): Calculates Duv, (CCT) from XYZ, wrapper for ..._ohno() & ..._search()
#
# cct_to_xyz(): Calculates xyz from CCT, Duv [100 K < CCT < 10**20]
#
# xyz_to_cct_mcamy(): Calculates CCT from XYZ using Mcamy model:
#                   *[McCamy, Calvin S. (April 1992). "Correlated color temperature as an explicit function of chromaticity coordinates". Color Research & Application. 17 (2): 142–144.](http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract)
#
# xyz_to_cct_HA(): Calculate CCT from XYZ using Hernández-Andrés et al. model .
#                  * [Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). Calculating Correlated Color Temperatures Across the Entire Gamut of Daylight and Skylight Chromaticities. Applied Optics. 38 (27): 5703–5709. PMID 18324081. doi:10.1364/AO.38.005703](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703)
#
# xyz_to_cct_ohno(): Calculates CCT,Duv from XYZ using LUT following:
#                   * [Ohno Y. Practical use and calculation of CCT and Duv. Leukos. 2014 Jan 2;10(1):47-55.](http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020)
#
# xyz_to_cct_search(): Calculates CCT,Duv from XYZ using brute-force search algorithm (between 1e2 K - 1e20 K on a log scale)
#
#cct_to_mired(): Converts from CCT to Mired scale (or back)
#
#------------------------------------------------------------------------------

from luxpy import *
__all__ = ['_cct_LUT','xyz_to_cct','cct_to_xyz','cct_to_mired','xyz_to_cct_ohno','xyz_to_cct_search','xyz_to_cct_HA','xyz_to_cct_mcamy']

#------------------------------------------------------------------------------
_cct_lut_dir = _pckg_dir + _sep + 'data'+ _sep + 'cctluts' + _sep #folder with cct lut data

#--------------------------------------------------------------------------------------------------
# load CCT LUT:
_cct_LUT = dictkv(keys = sorted(_cmf['types']), values = [getdata('{}cct_lut_{}.dat'.format(_cct_lut_dir,sorted(_cmf['types'])[i]),kind='np') for i in range(len(_cmf['types']))],ordered = False)



def xyz_to_cct_mcamy(data):
    """
	 Convert data = np.array([[x,y,z]]) tristimulus values to correlated color temperature (cct)
	 using the mcamy approximation for CCT (only valid for approx. 3000 < T < 9000, if < 6500, error < 2 K)
	 """
    Yxy = xyz_to_Yxy(data)
    axis_of_v3 = len(data.shape)-1
    n = (Yxy[:,1]-0.3320)/(Yxy[:,2]-0.1858)
    return  np2d(-449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33).T


def xyz_to_cct_HA(data):
    """
	 Convert data = np.array([[x,y,z]]) tristimulus values to correlated color temperature (cct) using:
	 Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
    Calculating Correlated Color Temperatures Across the Entire Gamut of Daylight and Skylight Chromaticities.
    Applied Optics. 38 (27): 5703–5709. PMID 18324081. doi:10.1364/AO.38.005703.
    According to paper small error from 3000 - 800 000 K, but a test with Planckians showed
    errors up to 20% around 500 000 K; e>0.05: T>200 000, e>0.1: e>300 000, ...
    """
    if len(data.shape)>2:
        raise Exception('xyz_to_cct_HA(): Input data.shape must be <= 2 !')
        
    out_of_range_code = np.nan
    xe = [0.3366, 0.3356]
    ye = [0.1735, 0.1691]
    A0 = [-949.86315, 36284.48953]
    A1 = [6253.80338, 0.00228]
    t1 = [0.92159, 0.07861]
    A2 = [28.70599, 5.4535*1e-36]
    t2 = [0.20039, 0.01543]
    A3 = [0.00004, 0.0]
    t3 = [0.07125,1.0]
    cct_ranges = np.array([[3000.0,50000.0],[50000.0,800000.0]])
    
    Yxy = xyz_to_Yxy(data)
    CCT = np.ones((1,Yxy.shape[0]))*out_of_range_code
    for i in range(2):
        n = (Yxy[:,1]-xe[i])/(Yxy[:,2]-ye[i])
        CCT_i = np2d(np.array(A0[i] + A1[i]*np.exp(np.divide(-n,t1[i])) + A2[i]*np.exp(np.divide(-n,t2[i])) + A3[i]*np.exp(np.divide(-n,t3[i]))))
        p = (CCT_i >= (1.0-0.05*(i == 0))*cct_ranges[i][0]) & (CCT_i < (1.0+0.05*(i == 0))*cct_ranges[i][1])
        CCT[p] = CCT_i[p]
        p = (CCT_i < (1.0-0.05)*cct_ranges[0][0]) #smaller than smallest valid CCT value
        CCT[p] = -1
   
    if (np.isnan(CCT.sum()) == True) | (np.any(CCT == -1)):
        print("Warning: xyz_to_cct_HA(): one or more CCTs out of range! --> (CCT < 3 kK,  CCT >800 kK) coded as (-1, NaN) 's")
    return CCT.T



def xyz_to_cct_search(data, cieobs = _cieobs, out = 'cct',wl = None, accuracy = 0.1, upper_cct_max = 10.0**20, approx_cct_temp = True):
    """
	 Convert data = np.array([[x,y,z]]) tristimlus values to cct (correlated color temperature) 
	 and Duv (distance above (>0) or below (<0) the Planckian locus) by a brute-force search:
	 The algorithm uses an approximate cct_temp (HA approx., see xyz_to_cct_HA) as starting point
	 or uses the middle of the allowed cct-range (1e2 K - 1e20 K, higher causes overflow) on a log-scale,
	 then constructs a 4-step section of the blackbody locus on which to find min distance to the 
    1960 uv chromaticity of the test source. 
    This program is more accurate, but slower than xyz_to_cct_ohno!
    Note that cct must be between 1e3 K - 1e20 K (very large cct take a long time!!!)
    """

    data = np2d(data)   
    
    if len(data.shape)>2:
        raise Exception('xyz_to_cct_search(): Input data.shape must be <= 2 !')
       
    # get 1960 u,v of test source:
    Yuvt = xyz_to_Yuv(np.squeeze(data)) # remove possible 1-dim + convert data to CIE 1976 u',v'
    axis_of_v3t = len(Yuvt.shape)-1 # axis containing color components
    ut = Yuvt[:,1,None] #.take([1],axis = axis_of_v3t) # get CIE 1960 u
    vt = (2/3)*Yuvt[:,2,None] #.take([2],axis = axis_of_v3t) # get CIE 1960 v

    # Initialize arrays:
    ccts = np.ones((data.shape[0],1))*np.nan
    duvs = ccts.copy()
    
    #calculate preliminary solution(s):
    if (approx_cct_temp == True):
        ccts_est = xyz_to_cct_HA(data)
        procent_estimates = np.array([[3000.0, 100000.0,0.05],[100000.0,200000.0,0.1],[200000.0,300000.0,0.25],[300000.0,400000.0,0.4],[400000.0,600000.0,0.4],[600000.0,800000.0,0.4],[800000.0,np.inf,0.25]])
    else:
        upper_cct = np.array(upper_cct_max)
        lower_cct = np.array(10.0**2)
        cct_scale_fun = lambda x: np.log10(x)
        cct_scale_ifun = lambda x: np.power(10.0,x)
        dT = (cct_scale_fun(upper_cct) - cct_scale_fun(lower_cct))/2
        ccttemp = np.array([cct_scale_ifun(cct_scale_fun(lower_cct) + dT)])
        ccts_est = np2d(ccttemp*np.ones((data.shape[0],1)))
        dT_approx_cct_False = dT.copy()

    
    # Loop through all ccts:        
    for i in range(data.shape[0]):

        #initialize CCT search parameters:
        cct = np.nan
        duv = np.nan
        ccttemp = ccts_est[i].copy()
        
        # Take care of (-1, NaN)'s from xyz_to_cct_HA signifying (CCT < lower, CCT > upper) bounds:
        approx_cct_temp_temp = approx_cct_temp
        if (approx_cct_temp == True):
            cct_scale_fun = lambda x: x
            cct_scale_ifun = lambda x: x
            if (ccttemp != -1) & (np.isnan(ccttemp) == False): # within validity range of CCT estimator-function
                for ii in range(procent_estimates.shape[0]):
                    if (ccttemp >= (1.0-0.05*(ii == 0))*procent_estimates[ii,0]) & (ccttemp < (1.0+0.05*(ii == 0))*procent_estimates[ii,1]):
                        procent_estimate = procent_estimates[ii,2]
                        break

                dT = np.multiply(ccttemp,procent_estimate) # determines range around CCTtemp (25% around estimate) or 100 K
            elif (ccttemp == -1) & (np.isnan(ccttemp) == False):
                ccttemp = np.array([procent_estimates[0,0]/2])
                procent_estimate = 1 # cover 0 K to min_CCT of estimator
                dT = np.multiply(ccttemp,procent_estimate)
            elif (np.isnan(ccttemp) == True):
                upper_cct = np.array(upper_cct_max)
                lower_cct = np.array(10.0**2)
                cct_scale_fun = lambda x: np.log10(x)
                cct_scale_ifun = lambda x: np.power(10.0,x)
                dT = (cct_scale_fun(upper_cct) - cct_scale_fun(lower_cct))/2
                ccttemp = np.array([cct_scale_ifun(cct_scale_fun(lower_cct) + dT)])
                approx_cct_temp = False

                
        else:
            dT = dT_approx_cct_False
      
        nsteps = 3 
        signduv = 1.0 
        ccttemp = ccttemp[0]
        delta_cct = dT
        while ((delta_cct > accuracy)):# keep converging on CCT 

            #generate range of ccts:
            ccts_i = cct_scale_ifun(np.linspace(cct_scale_fun(ccttemp)-dT,cct_scale_fun(ccttemp)+dT,nsteps+1))
            
            ccts_i[ccts_i < 100.0] = 100.0 # avoid nan's in calculation

            # Generate BB:
            BB = cri_ref(ccts_i,wl3 = wl,ref_type = ['BB'],cieobs = cieobs)
            
            # Calculate xyz:
            xyz = spd_to_xyz(BB,cieobs = cieobs)
    
            # Convert to CIE 1960 u,v:
            Yuv = xyz_to_Yuv(np.squeeze(xyz)) # remove possible 1-dim + convert data to CIE 1976 u',v'
            axis_of_v3 = len(Yuv.shape)-1 # axis containing color components
            u = Yuv[:,1,None] # get CIE 1960 u
            v = (2.0/3.0)*Yuv[:,2,None] # get CIE 1960 v
            
            # Calculate distance between list of uv's and uv of test source:
            dc = ((ut[i] - u)**2 + (vt[i] - v)**2)**0.5
            if np.isnan(dc.min()) == False:
                eps = _eps
                q = dc.argmin()
    
                if np.size(q) > 1: #to minimize calculation time: only calculate median when necessary
                    cct = np.median(ccts[q])
                    duv = np.median(dc[q])
                    q = np.median(q)
                    q = int(q) #must be able to serve as index
    
                else:
                     cct = ccts_i[q]
                     duv = dc[q]
                    
                
                if (q == 0):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun([cct])) + 2*dT/nsteps)
                    #dT = 2.0*dT/nsteps
                    continue # look in higher section of planckian locus
                    
                if (q == np.size(ccts_i)):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun([cct])) - 2*dT/nsteps)
                    #dT = 2.0*dT/nsteps
                    continue # look in lower section of planckian locus
                    
                if (q > 0) & (q < np.size(ccts_i)-1):
                    dT = 2*dT/nsteps
                    # get Duv sign:
                    d_p1m1 = ((u[q+1] - u[q-1])**2.0 + (v[q+1] - v[q-1])**2.0)**0.5
    
                    x = (dc[q-1]**2.0 - dc[q+1]**2.0 + d_p1m1**2.0)/2.0*d_p1m1
                    vBB = v[q-1] + ((v[q+1] - v[q-1]) * (x / d_p1m1))
                    signduv =np.sign(vt[i]-vBB)

                
                #calculate difference with previous intermediate solution:
                delta_cct = abs(cct - ccttemp)
                
                ccttemp = np.array([cct]) #%set new intermediate CCT
                approx_cct_temp = approx_cct_temp_temp
            else:
                ccttemp = np.nan 
                cct = np.nan
                duv = np.nan
              

        duvs[i] = signduv*abs(duv)
        ccts[i] = cct
    
    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2d(ccts)
    elif (out == 'duv') | (out == -1):
        return np2d(duvs)
    elif (out == 'cct,duv') | (out == 2):
        return np2d(ccts), np2d(duvs)
    elif (out == "[cct,duv]") | (out == -2):
        return np.vstack((ccts,duvs)).T

def xyz_to_cct_ohno(data, cieobs = _cieobs, out = 'cct', wl = None, accuracy = 0.1, force_out_of_lut = True, upper_cct_max = 10.0**20, approx_cct_temp = True):
    """
    Convert data = np.array([[x,y,z]]) tristimlus values to cct (correlated color temperature) 
	 and Duv (distance above (>0) or below (<0) the Planckian locus) according to:
    Ohno Y. Practical use and calculation of CCT and Duv. Leukos. 2014 Jan 2;10(1):47-55.
    """

    data = np2d(data)  

    if len(data.shape)>2:
        raise Exception('xyz_to_cct_ohno(): Input data.shape must be <= 2 !')
      
    # get 1960 u,v of test source:
    Yuv = xyz_to_Yuv(data) # remove possible 1-dim + convert data to CIE 1976 u',v'
    axis_of_v3 = len(Yuv.shape)-1 # axis containing color components
    u = Yuv[:,1,None] # get CIE 1960 u
    v = (2.0/3.0)*Yuv[:,2,None] # get CIE 1960 v

    uv = np2d(np.concatenate((u,v),axis = axis_of_v3))
    
    # load cct & uv from LUT:
    cct_LUT = _cct_LUT[cieobs][:,0,None] 
    uv_LUT = _cct_LUT[cieobs][:,1:3] 
    
    # calculate CCT of each uv:
    CCT = np.ones(uv.shape[0])*np.nan # initialize with NaN's
    Duv = CCT.copy() # initialize with NaN's
    idx_m = 0
    idx_M = uv_LUT.shape[0]-1
    for i in range(uv.shape[0]):
        out_of_lut = False
        delta_uv = (((uv_LUT - uv[i])**2.0).sum(axis = 1))**0.5 # calculate distance of uv with uv_LUT
        idx_min = delta_uv.argmin() # find index of minimum distance 

        # find Tm, delta_uv and u,v for 2 points surrounding uv corresponding to idx_min:
        if idx_min == idx_m:
            idx_min_m1 = idx_min
            out_of_lut = True
        else:
            idx_min_m1 = idx_min - 1
        if idx_min == idx_M:
            idx_min_p1 = idx_min
            out_of_lut = True
        else:
            idx_min_p1 = idx_min + 1
        

        if (out_of_lut == True) & (force_out_of_lut == True): # calculate using search-function
            cct_i, Duv_i = xyz_to_cct_search(data[i], cieobs = cieobs, wl = wl, accuracy = accuracy,out = 'cct,duv',upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp)
            CCT[i] = cct_i
            Duv[i] = Duv_i
            continue
        elif (out_of_lut == True) & (force_out_of_lut == False):
            CCT[i] = np.nan
            Duv[i] = np.nan
            
            
        cct_m1 = cct_LUT[idx_min_m1] # - 2*_eps
        delta_uv_m1 = delta_uv[idx_min_m1]
        uv_m1 = uv_LUT[idx_min_m1]
        cct_p1 = cct_LUT[idx_min_p1] 
        delta_uv_p1 = delta_uv[idx_min_p1]
        uv_p1 = uv_LUT[idx_min_p1]

        cct_0 = cct_LUT[idx_min]
        delta_uv_0 = delta_uv[idx_min]

        # calculate uv distance between Tm_m1 & Tm_p1:
        delta_uv_p1m1 = ((uv_p1[0] - uv_m1[0])**2.0 + (uv_p1[1] - uv_m1[1])**2.0)**0.5

        # Triangular solution:
        x = ((delta_uv_m1**2)-(delta_uv_p1**2)+(delta_uv_p1m1**2))/(2*delta_uv_p1m1)
        Tx = cct_m1 + ((cct_p1 - cct_m1) * (x / delta_uv_p1m1))
        uBB = uv_m1[0] + (uv_p1[0] - uv_m1[0]) * (x / delta_uv_p1m1)
        vBB = uv_m1[1] + (uv_p1[1] - uv_m1[1]) * (x / delta_uv_p1m1)

        Tx_corrected_triangular = Tx*0.99991
        signDuv = np.sign(uv[i][1]-vBB)
        Duv_triangular = signDuv*np.atleast_1d(((delta_uv_m1**2.0) - (x**2.0))**0.5)

                                
        # Parabolic solution:   
        a = delta_uv_m1/(cct_m1 - cct_0 + _eps)/(cct_m1 - cct_p1 + _eps)
        b = delta_uv_0/(cct_0 - cct_m1 + _eps)/(cct_0 - cct_p1 + _eps)
        c = delta_uv_p1/(cct_p1 - cct_0 + _eps)/(cct_p1 - cct_m1 + _eps)
        A = a + b + c
        B = -(a*(cct_p1 + cct_0) + b*(cct_p1 + cct_m1) + c*(cct_0 + cct_m1))
        C = (a*cct_p1*cct_0) + (b*cct_p1*cct_m1) + (c*cct_0*cct_m1)
        Tx = -B/(2*A+_eps)
        Tx_corrected_parabolic = Tx*0.99991
        Duv_parabolic = signDuv*(A*np.power(Tx_corrected_parabolic,2) + B*Tx_corrected_parabolic + C)

        Threshold = 0.002
        if Duv_triangular < Threshold:
            CCT[i] = Tx_corrected_triangular
            Duv[i] = Duv_triangular
        else:
            CCT[i] = Tx_corrected_parabolic
            Duv[i] = Duv_parabolic
    
    
    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2dT(CCT)
    elif (out == 'duv') | (out == -1):
        return np2dT(Duv)
    elif (out == 'cct,duv') | (out == 2):
        return np2dT(CCT), np2dT(Duv)
    elif (out == "[cct,duv]") | (out == -2):
        return np.vstack((CCT,Duv)).T


#---------------------------------------------------------------------------------------------------
def cct_to_xyz(data, duv = None, cieobs = _cieobs, wl = None, mode = 'lut', out = None, accuracy = 0.1, force_out_of_lut = True, upper_cct_max = 10.0*20, approx_cct_temp = True):
    """
	 Convert data = np.array([[cct]]) or data = np.array([[cct,duv]]) to np.array([[x,y,z]]) tristimulus values.
	 If duv is not supplied source is assumed to be on the Planckian locus.
	 """
    # make data a min. 2d np.array:
    if isinstance(data,list):
        data = np2dT(np.array(data))
    else:
        data = np2d(data) 
    
    if len(data.shape)>2:
        raise Exception('cct_to_xyz(): Input data.shape must be <= 2 !')
    
    # get cct and duv arrays from data:
    cct = np2d(data[:,0,None])


    if (duv is None) & (data.shape[1] == 2):
        duv = np2d(data[:,1,None])
    elif duv is not None:
        duv = np2d(duv)

    #get estimates of approximate xyz values in case duv = None:
    BB = cri_ref(ccts = cct, wl3 = wl, ref_type = ['BB'])
    xyz_est = spd_to_xyz(data = BB, cieobs = cieobs, out = 1)
    results = np.ones([data.shape[0],3])*np.nan 

    if duv is not None:
        
        # optimization/minimization setup:
        def objfcn(uv_offset, uv0, cct,duv, out = 1):#, cieobs = cieobs, wl = wl, mode = mode):
            uv0 = np2d(uv0 + uv_offset)
            Yuv0 = np.concatenate((np2d([100.0]), uv0),axis=1)
            cct_min, duv_min = xyz_to_cct(Yuv_to_xyz(Yuv0),cieobs = cieobs, out = 'cct,duv',wl = wl, mode = mode, accuracy = accuracy, force_out_of_lut = force_out_of_lut, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp)
            F = np.sqrt(((100.0*(cct_min[0] - cct[0])/(cct[0]))**2.0) + (((duv_min[0] - duv[0])/(duv[0]))**2.0))
            if out == 'F':
                return F
            else:
                return np.concatenate((cct_min, duv_min, np2d(F)),axis = 1) 
            
        # loop through each xyz_est:
        for i in range(xyz_est.shape[0]):
            xyz0 = xyz_est[i]
            cct_i = cct[i]
            duv_i = duv[i]
            cct_min, duv_min =  xyz_to_cct(xyz0,cieobs = cieobs, out = 'cct,duv',wl = wl, mode = mode, accuracy = accuracy, force_out_of_lut = force_out_of_lut, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp)
            
            if np.abs(duv[i]) > _eps:
                # find xyz:
                Yuv0 = xyz_to_Yuv(xyz0)
                uv0 = Yuv0[0] [1:3]

                OptimizeResult = minimize(fun = objfcn,x0 = np.zeros((1,2)), args = (uv0,cct_i, duv_i, 'F'), method = 'Nelder-Mead',options={"maxiter":np.inf, "maxfev":np.inf, 'xatol': 0.000001, 'fatol': 0.000001})
                betas = OptimizeResult['x']
                #betas = np.zeros(uv0.shape)
                if out is not None:
                    results[i] = objfcn(betas,uv0,cct_i, duv_i, out = 3)
                
                uv0 = np2d(uv0 + betas)
                Yuv0 = np.concatenate((np2d([100.0]),uv0),axis=1)
                xyz_est[i] = Yuv_to_xyz(Yuv0)
            
            else:
                xyz_est[i] = xyz0
      
    if (out is None) | (out == 1):
        return xyz_est
    else:
        # Also output results of minimization:
        return np.concatenate((xyz_est,results),axis = 1)  


#-------------------------------------------------------------------------------------------------   
# general CCT-wrapper function
def xyz_to_cct(data, cieobs = _cieobs, out = 'cct',mode = 'lut', wl = None,accuracy = 0.1, force_out_of_lut = True, upper_cct_max = 10.0**20,approx_cct_temp = True): 
    """
    Convert data = np.array([[x,y,z]]) tristimlus values to cct (correlated color temperature) 
    and Duv (distance above (>0) or below (<0) the Planckian locus) using Ohno's method or using a brute-force search algorithm.
    """
    if (mode == 'lut') | (mode == 'ohno'):
        return xyz_to_cct_ohno(data = data, cieobs = cieobs, out = out, accuracy = accuracy, force_out_of_lut = force_out_of_lut)
    elif (mode == 'search'):
        return xyz_to_cct_search(data = data, cieobs = cieobs, out = out, wl = wl, accuracy = accuracy, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp)


def xyz_to_duv(data, cieobs = _cieobs, out = 'duv',mode = 'lut', wl = None,accuracy = 0.1, force_out_of_lut = True, upper_cct_max = 10.0**20,approx_cct_temp = True): 
    """
    Convert data = np.array([[x,y,z]]) tristimlus values to Duv (distance above (>0) or below (<0) the Planckian locus) 
    and cct (correlated color temperature) using Ohno's method or using a brute-force search algorithm.
    Mainly a warpper function for colortf()
    """
    if (mode == 'lut') | (mode == 'ohno'):
        return xyz_to_cct_ohno(data = data, cieobs = cieobs, out = out, accuracy = accuracy, force_out_of_lut = force_out_of_lut)
    elif (mode == 'search'):
        return xyz_to_cct_search(data = data, cieobs = cieobs, out = out, wl = wl, accuracy = accuracy, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp)
   
   
#-------------------------------------------------------------------------------------------------   
def cct_to_mired(data):
    """
    Convert data = data = np.array([[cct]) to mired or back.                              
    """
    return np.divide(10**6,data)

