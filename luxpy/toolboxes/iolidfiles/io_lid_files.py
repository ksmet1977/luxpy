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
Module for reading / writing LID data from IES and LDT files.
=============================================================

 :read_lamp_data(): Read in light intensity distribution and other lamp data from LDT or IES files.

 :get_uv_texture(): Create a uv-texture map for use in renderings.
 
 :save_texture(): Save 16 bit grayscale PNG image of uv-texture.
 
 :draw_lid(): Draw 3D light intensity distribution.

    Notes:
        1. Only basic support. Writing is not yet implemented.
        2. Reading IES files is based on Blender's ies2cycles.py
        3. This was implemented to build some uv-texture maps for rendering and only tested for a few files.
        4. Use at own risk. No warranties.
     
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import os
from luxpy.utils import np
from luxpy.utils import plt, Axes3D 

from numpy import matlib
import scipy.interpolate as interp
import matplotlib
from matplotlib import cm

import imageio
imageio.plugins.freeimage.download()
# from skimage import exposure, img_as_uint

__all__ =['read_lamp_data','get_uv_texture','save_texture','draw_lid']


def read_lamp_data(filename, multiplier = 1.0, verbosity = 0, normalize = 'I0', only_common_keys = False):
    """
    Read in light intensity distribution and other lamp data from LDT or IES files.
    
    Args:
        :filename:
            | Filename of IES file.
        :multiplier:
            | 1.0, optional
            | Scaler for candela values.
        :verbosity:
            | 0, optional
            | Display messages while reading file.
        :normalize:
            | 'I0', optional
            | If 'I0': normalize LID to intensity at (theta,phi) = (0,0)
            | If 'max': normalize to max = 1.
        :only_common_keys:
            | False, optional
            | If True, output only common dict keys related to angles, values
            | and such of LID.
            | read_lid_lamp_data(?) for print of common keys and return
            |                       empty dict with common keys.
   
    Returns:
        :lid: dict with IES or LDT file data.
            |
            | If file_ext == 'ies':
            |    dict_keys(
            | ['filename', 'version', 'lamps_num', 'lumens_per_lamp',
            | 'candela_mult', 'v_angles_num', 'h_angles_num', 'photometric_type',
            | 'units_type', 'width', 'length', 'height', 'ballast_factor', 
            | 'future_use', 'input_watts', 'v_angs', 'h_angs', 'lamp_cone_type',
            | 'lamp_h_type', 'candela_values', 'candela_2d', 'v_same', 'h_same',
            | 'intensity', 'theta', 'values', 'phi', 'map','Iv0']
            |
            | If file_ext == 'ldt':
            |    dict_keys(
            | ['filename', 'version', 'manufacturer', 'Ityp','Isym',
            | 'Mc', 'Dc', 'Ng', 'name', Dg', 'cct/cri', 'tflux', 'lumens_per_lamp',
            | 'candela_mult', 'tilt', lamps_num',
            | 'cangles', 'tangles','candela_values', 'candela_2d',
            | 'intensity', 'theta', 'values', 'phi', 'map', 'Iv0']
            | )
            
    """
    common_keys = ['filename', 'version', 'intensity', 'theta', 'phi', \
                   'values', 'map', 'Iv0', 'candela_values', 'candela_2d']
    if filename == '?':
        print(common_keys)
        return dict(zip(common_keys,[np.nan]*len(common_keys))) 
    
    file_ext = filename[-3:].lower()
    if file_ext == 'ies':
        lid = read_IES_lamp_data(filename, multiplier = multiplier, \
                                 verbosity = verbosity, normalize = normalize)
    elif file_ext == 'ldt':
        lid = read_ldt_lamp_data(filename, multiplier = multiplier, normalize = normalize)
    else:
        raise Exception("read_lid_lamp_data(): {:s} --> unsupported file type/extension (only 'ies' or 'ldt': ".format(file_ext))
    
    if only_common_keys == True:
        return {key:value for (key,value) in lid.items() if key in common_keys}
    
    return lid
    
def displaymsg(code, message, verbosity = 1):
    """
    Display messages (used by read_IES_lamp_data).  
    """
    if verbosity > 0:
        print("{}: {}".format(code, message))
    
    

def read_IES_lamp_data(filename, multiplier = 1.0, verbosity = 0, normalize = 'I0'):
    """
    Read in IES files (adapted from Blender's ies2cycles.py).
    
    Args:
        :filename:
            | Filename of IES file.
        :multiplier:
            | 1.0, optional
            | Scaler for candela values.
        :verbosity:
            | 0, optional
            | Display messages while reading file.
        :normalize:
            | 'I0', optional
            | If 'I0': normalize LID to intensity at (theta,phi) = (0,0)
            | If 'max': normalize to max = 1.
            
    Returns:
        :IES: dict with IES file data.
            |
            | dict_keys(
            | ['filename', 'version', 'lamps_num', 'lumens_per_lamp',
            | 'candela_mult', 'v_angles_num', 'h_angles_num', 'photometric_type',
            | 'units_type', 'width', 'length', 'height', 'ballast_factor', 
            | 'future_use', 'input_watts', 'v_angs', 'h_angs', 'lamp_cone_type',
            | 'lamp_h_type', 'candela_values', 'candela_2d', 'v_same', 'h_same',
            | 'intensity', 'theta', 'values', 'phi', 'map','Iv0']
            | )
    """
    version_table = {
        'IESNA:LM-63-1986': 1986,
        'IESNA:LM-63-1991': 1991,
        'IESNA91': 1991,
        'IESNA:LM-63-1995': 1995,
        'IESNA:LM-63-2002': 2002,
    }
    
    name = os.path.splitext(os.path.split(filename)[1])[0]

    file = open(filename, 'rt', encoding='cp1252')
    content = file.read()
    file.close()
    s, content = content.split('\n', 1)


    if s in version_table:
        version = version_table[s]
    else:
        displaymsg('INFO', "IES file does not specify any version", verbosity = verbosity)
        version = None

    keywords = dict()

    while content and not content.startswith('TILT='):
        s, content = content.split('\n', 1)

        if s.startswith('['):
            endbracket = s.find(']')
            if endbracket != -1:
                keywords[s[1:endbracket]] = s[endbracket + 1:].strip()

    s, content = content.split('\n', 1)

    if not s.startswith('TILT'):
        displaymsg('ERROR' "TILT keyword not found, check your IES file", verbosity = verbosity)
        return None

    # fight against ill-formed files
    file_data = content.replace(',', ' ').split()

    lamps_num = int(file_data[0])
    if lamps_num != 1:
        displaymsg('INFO', "Only 1 lamp is supported, %d in IES file" % lamps_num, verbosity = verbosity)
    
    lumens_per_lamp = float(file_data[1])
    candela_mult = float(file_data[2])
    
    v_angles_num = int(file_data[3])
    h_angles_num = int(file_data[4])
    if not v_angles_num or not h_angles_num:
        displaymsg('ERROR', "TILT keyword not found, check your IES file", verbosity = verbosity)
        return None

    photometric_type = int(file_data[5])

    units_type = int(file_data[6])
    if units_type not in [1, 2]:
        displaymsg('INFO', "Units type should be either 1 (feet) or 2 (meters)", verbosity = verbosity)

    width, length, height = map(float, file_data[7:10])
    
    ballast_factor = float(file_data[10])

    future_use = float(file_data[11])
    if future_use != 1.0:
        displaymsg('INFO', "Invalid future use field", verbosity = verbosity)

    input_watts = float(file_data[12])

    v_angs = [float(s) for s in file_data[13:13 + v_angles_num]]
    h_angs = [float(s) for s in file_data[13 + v_angles_num:
                                          13 + v_angles_num + h_angles_num]]

    if v_angs[0] == 0 and v_angs[-1] == 90:
        lamp_cone_type = 'TYPE90'
    elif v_angs[0] == 0 and v_angs[-1] == 180:
        lamp_cone_type = 'TYPE180'
    else:
        displaymsg('INFO', "Lamps with vertical angles (%d-%d) are not supported" %
                       (v_angs[0], v_angs[-1]), verbosity = verbosity)
        lamp_cone_type = 'TYPE180'


    if len(h_angs) == 1 or abs(h_angs[0] - h_angs[-1]) == 360:
        lamp_h_type = 'TYPE360'
    elif abs(h_angs[0] - h_angs[-1]) == 180:
        lamp_h_type = 'TYPE180'
    elif abs(h_angs[0] - h_angs[-1]) == 90:
        lamp_h_type = 'TYPE90'
    else:
        displaymsg('INFO', "Lamps with horizontal angles (%d-%d) are not supported" %
                       (h_angs[0], h_angs[-1]), verbosity = verbosity)
        lamp_h_type = 'TYPE360'
        

    # read candela values
    offset = 13 + len(v_angs) + len(h_angs)
    candela_num = len(v_angs) * len(h_angs)
    candela_values = [float(s) for s in file_data[offset:offset + candela_num]]

    # reshape 1d array to 2d array
    candela_2d = list(zip(*[iter(candela_values)] * len(v_angs)))

    # check if angular offsets are the same
    v_d = [v_angs[i] - v_angs[i - 1] for i in range(1, len(v_angs))]
    h_d = [h_angs[i] - h_angs[i - 1] for i in range(1, len(h_angs))]

    v_same = all(abs(v_d[i] - v_d[i - 1]) < 0.001 for i in range(1, len(v_d)))
    h_same = all(abs(h_d[i] - h_d[i - 1]) < 0.001 for i in range(1, len(h_d)))

    if not h_same:
        displaymsg('INFO', "Different offsets for horizontal angles!", verbosity = verbosity)
        
    # normalize candela values
    maxval = max([max(row) for row in candela_2d])
    candela_2d = [[val / maxval for val in row] for row in candela_2d]
    intensity = maxval * multiplier * candela_mult
    #intensity = max(500, min(intensity, 5000)) #???

    # Summarize in dict():
    IES = {'filename': filename}
    IES['name'] = name
    IES['version'] = version
    IES['lamps_num'] = lamps_num
    IES['lumens_per_lamp'] = lumens_per_lamp
    IES['candela_mult'] = candela_mult
    IES['v_angles_num'] = v_angles_num
    IES['h_angles_num'] = h_angles_num
    IES['photometric_type'] = photometric_type
    IES['units_type'] = units_type
    IES['width'], IES['length'], IES['height'] = width, length, height
    IES['ballast_factor'] = ballast_factor
    IES['future_use'] = future_use
    IES['input_watts'] = input_watts
    IES['v_angs'] = np.asarray(v_angs)
    IES['h_angs'] = np.asarray(h_angs)
    IES['lamp_cone_type'] = lamp_cone_type
    IES['lamp_h_type'] = np.asarray(lamp_h_type)
    IES['candela_values'] = np.asarray(candela_values)
    IES['candela_2d'] = np.asarray(candela_2d)
    IES['v_same'] = v_same
    IES['h_same'] = h_same
    IES['intensity'] = intensity
    IES['map'] = {}
    
    # normalize candela values to max = 1 or I0 = 1:
    IES = _normalize_candela_2d(IES, normalize = normalize, multiplier = multiplier)

    # complete lid to full theta[0-180] and phi [0-360]
    IES = _complete_ies_lid(IES, lamp_h_type = IES['lamp_h_type'])
    
    IES['Iv0'] = IES['intensity']/1000*IES['lumens_per_lamp'] #lid in cd/klm 
    return IES
    
def _complete_ies_lid(IES, lamp_h_type = 'TYPE90'):
    """
    Convert IES LID map with lamp_h_type symmetry to a 'full' map with phi: [0,360] and theta: [0,180].
    """    
    # Create full theta (0-180) and phi (0-360) sets
    IES['theta'] = IES['v_angs']
    if IES['lamp_h_type'] == 'TYPE90':
        IES['values'] = matlib.repmat(IES['candela_2d'],4,1)
        IES['phi'] = np.hstack((IES['h_angs'], IES['h_angs'] + 90, IES['h_angs'] + 180, IES['h_angs']+270))
    elif IES['lamp_h_type'] == 'TYPE180':
        IES['values'] = matlib.repmat(IES['candela_2d'],2,1)
        IES['phi'] = np.hstack((IES['h_angs'], IES['h_angs'] + 180))
    else:
        IES['values'] = IES['candela_2d']
        IES['phi'] = IES['h_angs']
    IES['map']['thetas'] =  IES['theta']
    IES['map']['phis'] = IES['phi']
    IES['map']['values'] = IES['values']
    return IES
  

def read_ldt_lamp_data(filename, multiplier = 1.0, normalize = 'I0'):
    """
    Read in LDT files.
    
    Args:
        :filename:
            | Filename of LDT file.
        :multiplier:
            | 1.0, optional
            | Scaler for candela values.
        :verbosity:
            | 0, optional
            | Display messages while reading file.
        :normalize:
            | 'I0', optional
            | If 'I0': normalize LID to intensity at (theta,phi) = (0,0)
            | If 'max': normalize to max = 1.
            
    Returns:
        :LDT: dict with LDT file data.
            |
            | dict_keys(
            | ['filename', 'version', 'manufacturer', 'Ityp','Isym',
            | 'Mc', 'Dc', 'Ng', 'name', Dg', 'cct/cri', 'tflux', 'lumens_per_lamp',
            | 'candela_mult', 'tilt', lamps_num',
            | 'cangles', 'tangles','candela_values', 'candela_2d',
            | 'intensity', 'theta', 'values', 'phi', 'map', 'Iv0']
            | )
    """
    LDT = {'filename' : filename}
    LDT['version'] = None
    with open(filename) as file:
        c = 0
        cangles = []
        tangles = []
        candela_values = []
        for line in file:
            if c == 0: # manufacturer
                LDT['manufacturer'] = line.rstrip()
            elif c == 1: # type indicator: 1: point with symm. around vert. axis, 2: line luminaire, 3: point with other symm.
                if np.float(line) == 1.0:
                    LDT['Ityp'] = 'point source with symm. around vert. axis'
                elif np.float(line) == 2.0:
                    LDT['Ityp'] = 'line luminaire'
                elif np.float(line) == 3.0:
                    LDT['Ityp'] = 'point source with other symm.'
            elif c == 2: # symm. indicator
                if np.float(line) == 0.0:
                    LDT['Isym'] = (0, 'no symmetry')
                elif np.float(line) == 1.0:
                    LDT['Isym'] = (1, 'symmetry about the vertical axis')
                elif np.float(line) == 2.0:
                    LDT['Isym'] = (2, 'symmetry to plane C0-C180')
                elif np.float(line) == 3.0:
                    LDT['Isym'] = (3, 'symmetry to plane C90-C270')
                elif np.float(line) == 4.0:
                    LDT['Isym'] = (4, 'symmetry to plane C0-C180 and to plane C90-C270')
            elif c == 3: # Number Mc of C-planes between 0 and 360 degrees 
                LDT['Mc'] = np.float(line)
            elif c == 4: # Distance Dc between C-planes (Dc = 0 for non-equidistantly available C-planes)
                LDT['Dc'] = np.float(line)
            elif c == 5: # Number Ng of luminous intensities in each C-plane
                LDT['Ng'] = np.float(line)
            elif c == 6: # Distance Dg between luminous intensities per C-plane (Dg = 0 for non-equidistantly available luminous intensities in C-planes)
                LDT['Dg'] = np.float(line)
            elif c == 8: # luminaire name
                LDT['name'] = line.rstrip()
            elif c == 23: # conversion factor
                LDT['candela_mult'] = np.float(line)
            elif c == 24: # Tilt angle
                LDT['tilt'] = np.float(line)
            elif c == 26: # number of lamps
                LDT['lamps_num'] = np.float(line)
            elif c == 28: # total luminous flux
                LDT['tflux'] = np.float(line)
                LDT['lumens_per_lamp'] = LDT['tflux']
            elif c == 29: # cct/cri
                LDT['cct/cri'] = line.rstrip()
            elif (c >= 42) & (c <= (42 + LDT['Mc'] - 1)): # start of C-angles
                cangles.append(np.float(line))
            elif (c >= 42 + LDT['Mc']) & (c <= (42 + LDT['Mc'] + LDT['Ng'] - 1)): # start of t-angles
                tangles.append(np.float(line))
            elif (c >= (42 + LDT['Mc'] + LDT['Ng'])) & (c <= (42 + LDT['Mc'] + LDT['Ng'] + LDT['Mc']*LDT['Ng'] - 1)):
                candela_values.append(np.float(line))
            c += 1
            
        candela_values = np.array(candela_values)
        LDT['candela_values'] = np.array(candela_values)
        candela_2d = np.array(candela_values).reshape((-1,np.int(LDT['Ng'])))
        LDT['h_angs'] = np.array(cangles)[:candela_2d.shape[0]]
        LDT['v_angs'] = np.array(tangles)
        LDT['candela_2d'] = np.array(candela_2d)
        
        # normalize candela values to max = 1 or I0 = 1:
        LDT = _normalize_candela_2d(LDT, normalize = normalize, multiplier = multiplier)

        # complete lid to full theta[0-180] and phi [0-360]
        LDT = _complete_ldt_lid(LDT, Isym = LDT['Isym'][0])
        
        LDT['Iv0'] = LDT['intensity']/1000*LDT['tflux'] #lid in cd/klm 
        return LDT

    
def _complete_ldt_lid(LDT, Isym = 4):
    """
    Convert LDT LID map with Isym symmetry to a 'full' map with phi: [0,360] and theta: [0,180].
    """
    cangles = LDT['h_angs']
    tangles = LDT['v_angs']
    candela_2d = LDT['candela_2d']
    if Isym == 4:
        # complete cangles:
        a = candela_2d.copy().T
        b = np.hstack((a,a[:,(a.shape[1]-2)::-1]))
        c = np.hstack((b,b[:,(b.shape[1]-2):0:-1]))
        candela_2d_0C360 = np.hstack((c,c[:,:1])) 
        cangles = np.hstack((cangles, cangles[1:] + 90, cangles[1:] + 180, cangles[1:] + 270))
        # complete  tangles:
        a = candela_2d_0C360.copy()
        b = np.vstack((a,np.zeros(a.shape)[1:,:]))
        tangles = np.hstack((tangles, tangles[1:] + 90))
        candela_2d = b
    elif Isym == -4:
        # complete cangles:
        a = candela_2d.copy().T
        b = np.hstack((a,a[:,(a.shape[1]-2)::-1]))
        c = np.hstack((b,b[:,(b.shape[1]-2):0:-1]))
        candela_2d_0C360 = np.hstack((c,c[:,:1])) 
        cangles = np.hstack((cangles, -cangles[(cangles.shape[0]-2)::-1] + 180))
        cangles = np.hstack((cangles, -cangles[(cangles.shape[0]-2):0:-1] + 360))
        cangles = np.hstack((cangles,cangles[:1])) 
        # complete  tangles:
        a = candela_2d_0C360.copy()
        b = np.vstack((a,np.zeros(a.shape)[1:,:]))
        tangles = np.hstack((tangles, -tangles[(tangles.shape[0]-2)::-1] + 180))
        candela_2d = b
    else:
        raise Exception ('complete_ldt_lid(): Other "Isym" than "4", not yet implemented (31/10/2018).')
    
    LDT['map'] = {'thetas': tangles}
    LDT['map']['phis'] = cangles
    LDT['map']['values'] = candela_2d.T
    return LDT         

def _normalize_candela_2d(LID, normalize = 'I0', multiplier = 1):
    
    candela_2d = LID['candela_2d']
    if normalize == 'max': # normalize candela values to max = 1
        maxval = candela_2d.max()
        norm = maxval
    elif normalize == 'I0': # normalize candela values to I0 = 1 
        I0 = candela_2d[LID['h_angs']==0.0, LID['v_angs']==0.0]
        norm = I0[0]
    candela_2d = candela_2d/norm
    candela_mult = LID['candela_mult']
    intensity = norm * multiplier * candela_mult
    LID['candela_2d'] = candela_2d
    LID['intensity'] = intensity
    return LID


#------------------------------------------------------------------------------
# Texture creation and saving
#------------------------------------------------------------------------------

def _spher2cart(theta, phi, r = 1., deg = True):
    """
    Convert spherical to cartesian coordinates.
    
    Args:
        :theta:
            | Float, int or ndarray
            | Angle with positive z-axis.
        :phi:
            | Float, int or ndarray
            | Angle around positive z-axis starting from x-axis.
        :r:
            | 1, optional
            | Float, int or ndarray
            | radius
            
    Returns:
        :x, y, z:
            | tuple of floats, ints or ndarrays
            | Cartesian coordinates
    """
    if deg == True:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    x= r*np.sin(theta)*np.cos(phi)
    y= r*np.sin(theta)*np.sin(phi)
    z= r*np.cos(theta)
    return x,y,z



def get_uv_texture(theta, phi = None, values = None, input_types = ('array','array'),\
                   method = 'linear', theta_min = 0, angle_res = 1, close_phi = False,\
                   deg = True, r = 1, show = True, out = 'values_map'):
    """
    Create a uv-texture map.
    | with specified angular resolution (°) and with positive z-axis as normal.
    |   u corresponds to phi [0° - 360°]
    |   v corresponds to theta [0° - 180°], (or [-90° - 90°])
    
    Args:
        :theta:
            | Float, int or ndarray
            | Angle with positive z-axis.
            | Values corresponding to 0 and 180° must be specified!
        :phi:
            | None, optional
            | Float, int or ndarray
            | Angle around positive z-axis starting from x-axis.
            | If not None: values corresponding to 0 and 360° must be specified!
        :values:
            | None
            | ndarray or mesh of values at (theta, phi) locations. 
        :input_types:
            | ('array','array'), optional
            | Specification of type of input of (angles,values)
        :method:
            | 'linear', optional
            | Interpolation method.
            | (supported scipy.interpolate.griddata methods: 
            |  'nearest', 'linear', 'cubic')
        :theta_min:
            | 0, optional
            | If 0: [0, 180]; If -90: theta range = [-90,90]
        :close_phi:
            | False, optional
            | Make phi angles array closed (full circle).
        :angle_res:
            | 1, optional
            | Resolution in degrees.
        :deg: 
            | True, optional
            | Type of angle input (True: degrees, False: radians).
        :r:
            | 1, optional
            | Float, int or ndarray
            | radius
        :show:
            | True, optional
            | Plot results.
        :out:
            | 'values_map', optional
            | Specifies output: "return eval(out)"
        
    Returns:
        :returns: as specified by :out:.
    """
    
    # Create uv base map:
    #--------------------
    
    # set up uv_map angles:
    theta_map = np.arange(0, 180 + angle_res, angle_res)
    phi_map = np.arange(0, 360 + (close_phi)*angle_res, angle_res)
    
    # create angle base mesh:
    thetam_map, phim_map = np.meshgrid(theta_map, phi_map)
    
    # convert input angles to uv coordinates:
    um_map, vm_map = 0.5*phim_map/180, thetam_map/180
    
    
    # Create uv map from input:
    #-------------------------
    if phi is not None:
        if (phi==0.0).all(): phi = None  # if phi = 0: assume rotational symmetry
        
    # When only (theta,values) data is given--> assume rotational symmetry:
    if phi is None:
        phi = phi_map
        values = np.matlib.repmat(values,np.int(360*(1/angle_res)),1) # assume rotational symmetry, values must be array!
        input_types = (input_types[0],'mesh') 

    # convert radians to degrees:
    if deg == False:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    
    if (input_types[0] == 'array') & (input_types[1] == 'mesh'):
        # create angle input mesh:
        thetam_in, phim_in = np.meshgrid(theta, phi)
    elif (input_types[0] == 'array') & (input_types[1] == 'array'):
        thetam_in, phim_in = theta, phi # work with array data
    
    # convert input angles to uv coordinates:
    um_in, vm_in = 0.5*phim_in/180, thetam_in/180
    
    # Interpolate values for uv_in to values for uv_map:
    values_map = interp.griddata(np.array([um_in.ravel(),vm_in.ravel()]).T, values.ravel(), (um_map,vm_map), method = method)
    
    if show == True:
        xm_map, ym_map, zm_map = _spher2cart(thetam_map,phim_map, r = 1, deg = True)
        xm_in, ym_in, zm_in = _spher2cart(thetam_in,phim_in, r = r, deg = True)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection = '3d')
        ax1.plot(xm_map.ravel(), ym_map.ravel(), zm_map.ravel(),'bo', label = 'Output map')
        ax1.plot(xm_in.ravel(), ym_in.ravel(), zm_in.ravel(),'r.', label = 'Input map')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title('Cartesian map coordinates')
        
  
        ax2 = fig.add_subplot(122,projection = '3d')
        ax2.plot(um_map.ravel(), vm_map.ravel(), values_map.ravel(), 'bo', label = 'Output')
        ax2.plot(um_in.ravel(), vm_in.ravel(), values.ravel(), 'r.', label = 'Input')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        ax2.set_title('uv texture map')
        ax2.legend(loc = 'upper right')
    
    if theta_min == -90:
        values_map = np.roll(values_map, np.int(np.abs(theta_min)/angle_res), axis = 1)
        theta_map = theta_map + theta_min
        
    return eval(out)

def save_texture(filename, tex, bits = 16, transpose = True):
    """
    Save 16 bit grayscale PNG image of uv-texture.
    
    Args:
        :filename:
            | Filename of output image.
        :tex: 
            | ndarray float uv-texture.
        :transpose:
            | True, optional
            | If True: transpose tex (u,v) to set u as columns and v as rows 
            | in texture image.
            
    Returns:
        :None:
            
    Note:
        | Texture is rescaled to max = 1 and saved as uint16.
        | --> Before using uv_map: rescale back to set 'normal' to 1.
    """
    #im = exposure.rescale_intensity(tex, out_range='float')
    im = (((2**bits-1)*tex/tex.max()).astype("uint16")) #
    #im = img_as_uint(im)
    if transpose == True:
        im = im.T
    imageio.imsave(filename, im)
  
#------------------------------------------------------------------------------
# Make plot of LID
#------------------------------------------------------------------------------

def draw_lid(LID, grid_interp_method = 'linear', theta_min = 0, angle_res = 1,
             ax = None, use_scatter_plot = False, plot_colorbar = True, legend_on = False, 
             plot_luminaire_position = True, out = 'ax', **plottingkwargs):
    """
    Draw the light intensity distribution.
    
    Args:
        :LID:
            | dict with IES or LDT file data. 
            | (obtained with iolidfiles.read_lamp_data())
        :grid_interp_method:
            | 'linear', optional
            | Interpolation method for (theta,phi)-grid of normalized luminous intensity values.
            | (supported scipy.interpolate.griddata methods: 
            |  'nearest', 'linear', 'cubic')
        :theta_min:
            | 0, optional
            | If 0: [0, 180]; If -90: theta range = [-90,90]
        :angle_res:
            | 1, optional
            | Resolution in degrees.
        :ax:
            | None, optional
            | If None: create new 3D-axes for plotting.
        :use_scatter_plot:
            | False, optional
            | If True: use plt.scatter for plotting intensity values.
            | If False: use plt.plot_surface for plotting.
        :plot_colorbar:
            | True, optional
            | Plot colorbar representing the normalized luminous intensity values in the LID.
        :legend_on:
            | False, optional
            | plot legend.
        :plot_luminaire_position:
            | True, optional
            | Plot the position of the luminaire (0,0,0) in the graph as a red diamond.
        :out:
            | 'ax', optional
            | string with variable to return
            | default: ax handle to plot.
        
    Returns:
        :returns:
            | Whatever requested as determined by the string in :out:
                
    """
    values_map,phim_map,thetam_map = get_uv_texture(theta = LID['theta'], 
                                                    phi = LID['phi'], 
                                                    values = LID['values'], 
                                                    input_types = ('array','mesh'), 
                                                    method = grid_interp_method, 
                                                    theta_min = theta_min, 
                                                    angle_res = angle_res, 
                                                    deg = True, 
                                                    r = 1, 
                                                    show = False,
                                                    out='values_map,phim_map,thetam_map')
    values_map[np.isnan(values_map)] = 0
    
    # get cartesian coordinates:
    #r = values_map/values_map.max() if normalize_intensity_to_max else values_map
    xm_map,ym_map,zm_map=_spher2cart(thetam_map,phim_map, r = values_map, deg = True)
    
    # make plot:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
    V = values_map
    norm = matplotlib.colors.Normalize(vmin=V.min().min(), vmax=V.max().max())
    if use_scatter_plot:
        ax.scatter(xm_map.ravel(),ym_map.ravel(),-zm_map.ravel(), 
                   c = values_map.ravel(), cmap = 'jet', alpha = 0.5, label = 'Normalized luminous intensity', **plottingkwargs)
    else:
        ax.plot_surface(xm_map,ym_map,-zm_map, facecolors = plt.cm.jet(norm(V)), label = 'Normalized luminous intensity', **plottingkwargs)
    
    if plot_colorbar:
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        m.set_array([])
        cbar = plt.colorbar(m)
        cbar.set_label('Normalized luminous intensity ($I_0 = 1$)')
    
    if plot_luminaire_position:
        ax.plot(0,0,0,color='r',marker='d', markersize = 16, alpha = 0.7, label = 'Luminaire equivalent position')
    
    # calculate aspect ratio:
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    d = np.diff(xyzlim,axis=0)
    r = d/d.max()
    r = r/r.min()
    ax.set_box_aspect(tuple(r[0]))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if legend_on: ax.legend()
    
    
    return eval(out)

if __name__ == '__main__':
    
    # Read lamp data from IES file:
    IES = read_lamp_data('./data/111621PN.ies', verbosity = 1)
    
    # Generate uv-map for rendering / ray-tracing (eg by wrapping this around 
    # a point light source to attenuate the luminous intensity in different directions):
    uv_map = get_uv_texture(theta = IES['theta'], 
                             phi = IES['phi'], 
                             values = IES['values'], 
                             input_types = ('array','mesh'), 
                             method = 'linear', 
                             theta_min = 0, angle_res = 1,
                             deg = True, r = 1, 
                             show = True)
    plt.figure()
    plt.imshow(uv_map)
    
    # draw LID:
    draw_lid(IES)
    
    

        
    
        
  
