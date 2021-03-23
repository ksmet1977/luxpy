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
 
 :render_lid(): Render a light intensity distribution.

    Notes:
        1. Only basic support (reading / visualization). Writing is not yet implemented.
        2. Reading IES files is based on Blender's ies2cycles.py
        3. This was implemented to build some uv-texture maps for rendering and only tested for a few files.
        4. Use at own risk. No warranties. Still under test.
     
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

__all__ =['read_lamp_data','get_uv_texture','save_texture','draw_lid','render_lid']


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
    else:
        raise Exception("Unsupported normalize option (valid string options are 'max', 'I0')")
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

def _cart2spher(x,y,z, deg = True):
    """
    Convert cartesian to spherical coordinates.
    
    Args:
        :x, y, z:
            | tuple of floats, ints or ndarrays
            | Cartesian coordinates
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
        :theta:
            | ndarray of angles with positive z-axis.
        :phi:
            | ndarray of angles around positive z-axis starting from x-axis.
        :r:
            | ndarray of radii
    """
    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    phi[phi<0.0] = phi[phi<0.0] + 2*np.pi
    if deg == True:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return theta,phi,r

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
        values = matlib.repmat(values,np.int(360*(1/angle_res)),1) # assume rotational symmetry, values must be array!
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

def get_cart_lid_map(LID,grid_interp_method = 'linear', theta_min = 0, angle_res = 1,):
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
    return xm_map,ym_map,zm_map,phim_map,thetam_map,values_map

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
    (xm_map,ym_map,zm_map,phim_map,thetam_map,
    values_map) = get_cart_lid_map(LID,
                                   grid_interp_method = grid_interp_method, 
                                   theta_min = theta_min, 
                                   angle_res = angle_res)   
    
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
        plt.sca(ax)
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

#------------------------------------------------------------------------------
# Render LID image
#------------------------------------------------------------------------------

def _norm(x,axis=0):
    return (x**2).sum(axis=axis,keepdims=True)**0.5

def _rotationmatrix_3d(u,theta, length=3):
    theta = theta
    cost = np.cos(theta)
    omcost = 1 - np.cos(theta)
    sint = np.sin(theta)
    R = np.array([[cost + u[0]**2*omcost, u[0]*u[1]*omcost - u[2]*sint, u[0]*u[2]*omcost + u[1]*sint],
                      [u[0]*u[1]*omcost + u[2]*sint, cost + u[1]**2*omcost, u[1]*u[2]*omcost - u[0]*sint],
                      [u[0]*u[2]*omcost - u[1]*sint, u[1]*u[2]*omcost + u[0]*sint, cost + u[2]**2*omcost]])    
    if length == 4:
        R = np.vstack((np.hstack((R,np.zeros((3,1)))),np.zeros((1,4))))
        R[-1,-1] = 1
    return R

# def cross_to_angle(a,b, u = None):
#     crossab = np.cross(a,b) if u is None else u
#     return np.arcsin(_norm(crossab)/(_norm(a)*_norm(b)))

def _dot_to_angle(a,b):
    return np.arctan2(_norm(np.cross(a,b)), np.dot(a,b)) 

def _perpendicular_vector(v):
    
    if v[0] == 0:
        return np.array([1,0,0])
    if v[1] == 0:
        return np.array([0,1,0])
    if v[2] == 0:
        return np.array([0,0,1])
    
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])

def _get_rotation_matrix(n1,n2,length=3):
    
    # calculate angle between normal n1 and normal n2:
    theta = _dot_to_angle(n1,n2)[0] 
    
    # calculate rotation axis to rotate normal n1 to desired normal n2:
    u = np.cross(n1, n2) 
    if _norm(u) == 0:
        u = _perpendicular_vector(n1)
    u = u/_norm(u)

    # get rotation matrix
    R = _rotationmatrix_3d(u,theta, length = length)

    return R, theta, u

def _create_plane(w, h, center = [0,0,0], n = [0,1,0], res = None):
    
    center = np.array(center)
    n = np.array(n)
    
    # create plane with width w and height h and normal along negative y-axis:
    corners = np.array([[-w/2,0,h/2],
                        [-w/2,0,-h/2],
                        [w/2,0,-h/2],
                        [w/2,0,h/2]])
    s1 = -(corners[1] - corners[0])
    s2 = (corners[3] - corners[0])
    n_ = np.cross(s1,s2)
    n_ = n_/_norm(n_)

    # get raotation matrix (and angle theta around axis u)
    # to go from constructed normal n_ to desired normal n:
    R, theta, u = _get_rotation_matrix(n_,n)

    # create grid:
    if res is not None:
        x = np.linspace(corners[0,0],corners[2,0], res)
        y = np.linspace(corners[2,2],corners[0,2], res)
        grid = np.array([[i,0,j] for i in x for j in y])
        grid = np.dot(R,grid.T).T + center
    else:
        grid = None
        
    # rotate plane and recalculate n_ (should match n, if not then n_ & n colinear):
    corners = np.dot(R,corners.T).T 
    s1 = -(corners[1] - corners[0])
    s2 = (corners[3] - corners[0])
    n_ = np.cross(s1,s2)
    n_ = n_/_norm(n_)

    corners = corners + center

    return corners, n_, grid

#corners,n_, grid = _create_plane(w = 20, h = 10, center = np.array([0,1,1]), n = [-1,0,0], res = 100)

                         
def _create_sensor_plane(fov = (90,90), Fd = 1, position = [0,0,1], n = [0,1,0], res = 200):
    fov = np.array(fov)
    n = np.array(n)
    position = np.atleast_2d(position)
    size = np.deg2rad(fov/2)*Fd*2
    
    center = position + np.array([0,Fd,0]) # center of sensor plane
    corners, n, pixels = _create_plane(size[0], size[1], center, n = n, res = res)
    position = -n*Fd + center # new position of sensor focal point

    drays = pixels - position
    drays = drays/_norm(drays,axis = 1)
    
    return position, corners, center, n, pixels, drays

def _get_line_plane_intersection(l0,dl, p0, n, corners):
    # check if not parallel:
    intersects = np.dot(dl,n) != 0

    # get u of intersection point along line segment l(u) = l0 + u*dl:
    u = np.nan*np.ones_like(intersects)
    u[intersects] = np.dot((p0-l0),n) / np.dot(dl[intersects],n)
    
    # intersection points:
    l = l0 + u[...,None]*dl
    
    # check if inside rectangle:
    # a. get length of sides of rectangle:
    s1 = corners[1] - corners[0]
    s2 = corners[3] - corners[0]
    # b. get projection's q1 and q2 of vector(corners[0],l) along s1 and s2:
    q1 = np.dot(l - corners[0],s1)
    q2 = np.dot(l - corners[0],s2)
    # c. check if projections are smaller than sizes and larger than zero:
    in_rectangle = ((q1>=0) & (q1<=np.dot(s1,s1)) & (q2>=0) & (q2<=np.dot(s2,s2)))
    # d. fill rays which miss rectangles with nan's:
    l[np.logical_not(in_rectangle)] = np.nan
    
    return l, u
    
def _plot_plane_edges(corners, ax, color = 'r', marker = 'o'):
    corners = np.vstack((corners,corners[0]))
    for i in range(corners.shape[0]-1):
        ax.plot(np.array((corners[i,0],corners[i+1,0])),  
                np.array((corners[i,1],corners[i+1,1])),
                np.array((corners[i,2],corners[i+1,2])),color = color, marker = marker)
    return ax

def _read_luminous_intensity(thetas, phis, LID, method = 'linear'):
    # Interpolate values for uv_in to values for uv_map:
    thetam_in, phim_in = np.meshgrid(LID['theta'], LID['phi'])
    Iv = LID['values']
    Ivs = interp.griddata(np.array([phim_in.ravel(),thetam_in.ravel()]).T, Iv.ravel(), (phis,thetas), method = method)
    return Ivs

def _get_luminaire_illuminance_at_plane(plum, nlum, pplane, lid, xyzm_maps):
    u0 = (pplane - plum) # vectors from plane points to lum
    # d = _norm(u) # distances between luminaire and points
    
    # # rotate lid of luminaire:
    nz = np.array([0,0,1])
    R, theta, nrot = _get_rotation_matrix(nz,nlum)
    Ri, thetai, nroti = _get_rotation_matrix(nlum,nz)
    
    if xyzm_maps is not None:
        xyzm_maps = np.vstack([xyzm_maps[i].ravel() for i in range(len(xyzm_maps))]).T
        xyzm_maps = np.dot(R,xyzm_maps.T).T
    nlum = np.dot(R,nz.T).T #update LID-normal (pointing away from luminaire)
    u = np.dot(Ri,u0.T).T
        
    if xyzm_maps is not None:
        xyzm_maps = xyzm_maps + plum
    
    # get theta, phi, r relative to source orientation for use with LID:
    theta, phi, r = _cart2spher(u[:,0],u[:,1],u[:,2], deg = True)
    
   
    # luminous intensity Iv along direction (theta,phi)
    Iv = _read_luminous_intensity(theta, phi, lid, method = 'linear')
    
    # illuminance at point pplane:
    Ev = Iv*np.cos(np.deg2rad(theta))/r**2
    return Ev, u0, nlum, r, xyzm_maps

def _get_plane_luminance(plum, nlum, lid, pplane, nplane, psensor, rho, xyzm_maps):
    Ev, u, nlum, dist_source, xyzm_maps = _get_luminaire_illuminance_at_plane(plum, nlum, pplane, lid, xyzm_maps) 
    
    v = -u # check whether rays are pointing in opposite direction as surface normal of plane
    v = v/_norm(v,axis=1)
    reflect = np.dot(v,nplane[...,None])[:,0]
    Ev[reflect<0] = np.nan # when normal is not pointing toward incident rays--> no reflection (black backside of plane)
    
    reflect = np.dot(pplane - psensor,nplane[...,None])[:,0] # plane surface normal must also point towards sensor
    Ev[reflect>0] = np.nan
    
    if xyzm_maps is None:
        return Ev*rho/np.pi,nlum, dist_source, xyzm_maps # assume Lambertian reflector
    else:
        return Ev*rho/np.pi,nlum, dist_source, xyzm_maps.T # assume Lambertian reflector
        
# def map_3D_to_2D(pplane,Lv, fov = (90,90), Fd = 1, res = 100, method = 'linear'):
#     fov = np.array(fov)
#     size = np.deg2rad(fov/2)*Fd*2
#     w, h = size/2
#     corners = np.array([[-w/2,h/2],
#                         [-w/2,-h/2],
#                         [w/2,-h/2],
#                         [w/2,h/2]])
#     x = np.linspace(corners[0,0],corners[2,0], res)
#     y = np.linspace(corners[2,1],corners[0,1], res)
#     xg, yg = np.meshgrid(x,y)
#     Lv2D = interp.griddata(pplane,Lv, (x,y), method = method)
#     return Lv2D


def render_lid(LID = './data/luxpy_test_lid_file.ies', 
               sensor_resolution = 100, sensor_position = [0,-1,0.8], sensor_n = [0,1,-0.2], fov = (90,90), Fd = 2,
               luminaire_position = [0,1.3,2], luminaire_n = [0,0,-1],
               wall_center = [0,2,1], wall_n = [0,-1,0], wall_width = 4, wall_height = 2, wall_rho = 1,
               floor_center = [0,1,0], floor_n = [0,0,1], floor_width = 4, floor_height = 2, floor_rho = 1,
               grid_interp_method='linear', angle_res = 5, theta_min = 0,
               ax3D = None, ax2D = None, join_axes = True, legend_on = True,
               plot_luminaire_position = True, plot_lumiaire_rays = False, plot_luminaire_lid = True,
               plot_sensor_position = True, plot_sensor_pixels = True, plot_sensor_rays = False, 
               plot_wall_edges = True, plot_wall_luminance = True, plot_wall_intersections = False,
               plot_floor_edges = True, plot_floor_luminance = True, plot_floor_intersections = False,
               out = 'Lv2D'):
    """
    Render a light intensity distribution.
    
    Args:
        :LID:
            | dict with IES or LDT file data or string with path/filename. 
            | (dict should be obtained with iolidfiles.read_lamp_data())
        :sensor_resolution:
            | 100, optional
            | Number of sensor 'pixels' along each dimension.
        :sensor_position:
            | [0,-1,0.8], optional
            | x,y,z position of the sensor 'focal' point (is located Fd meters behind actual sensor plane)
        :sensor_n:
            | [0,1,-0.2], optional
            | Sensor plane surface normal
        :fov:
            | (90,90), optional
            | Field of view of sensor image in degrees.
        :Fd:
            | 2, optional
            | 'Focal' distance in meter. Sensor center is located Fd meter away from
            | :sensor_position:
        :luminaire_position:
            | [0,1.3,2], optional
            | x,y,z position of the photometric equivalent point source
        :luminaire_n:
            | [0,0,-1], optional
            | Orientation of lumaire LID (default points downward along z-axis away from source)
        :wall_center:
            | [0,2,1], optiona
            | x,y,z position of the back wall
        :wall_n:
            | [0,-1,0], optional
            | surface normal of wall
        :wall_width:
            | 4, optional
            | width of wall (m)
        :wall_height:
            | 2, optional
            | height of wall (m)
        :wall_rho:
            | 1, optional
            | Diffuse (Lambertian) reflectance of wall.
        :floor_center:
            | [0,1,0], optiona
            | x,y,z position of the floor
        :floor_n:
            | [0,0,1], optional
            | surface normal of floor
        :floor_width:
            | 4, optional
            | width of floor (m)
        :floor_height:
            | 2, optional
            | height of floor (m)
        :floor_rho:
            | 1, optional
            | Diffuse (Lambertian) reflectance of floor.
        :grid_interp_method:
            | 'linear', optional
            | Interpolation method for (theta,phi)-grid of normalized luminous intensity values.
            | (supported scipy.interpolate.griddata methods: 
            |  'nearest', 'linear', 'cubic')
        :theta_min:
            | 0, optional
            | If 0: [0, 180]; If -90: theta range = [-90,90]
            | Only used when generating a plot of the LID in the 3D graphs.
        :angle_res:
            | 1, optional
            | Angle resolution in degrees of LID sampling.
            | Only used when generating a plot of the LID in the 3D graphs.
        :ax3D,ax2D:
            | None, optional
            | If None: create new 3D- or 2D- axes for plotting.
            | If join_axes == True: try and combine two axes on same figure.
        :legend_on:
            | False, optional
            | plot legend.
        :plot_luminaire_position:
            | True, optional
            | Plot the position of the luminaire (0,0,0) in the graph as a red diamond.
        :plot_X_...:
            | VArious options to customize plotting. Mainly allows for plotting of
            | additional info such as plane-ray intersection points, sensor pixels,
            | sensor-to-plane rays, plane-to-luminaire rays, 3D plot of LID, etc.
        :out:
            | 'Lv2D', optional
            | string with variable to return
            | default: variable storing an grayscale image of the rendered LID.
        
    Returns:
        :returns:
            | Whatever requested as determined by the string in :out:
        
    """
    if isinstance(LID,str):
        LID = read_lamp_data(LID, verbosity = 0)
    
    # parse input:
    res = sensor_resolution
    sensor_position = np.atleast_2d(sensor_position)
    sensor_n = np.array(sensor_n)
    luminaire_position = np.atleast_2d(luminaire_position)
    luminaire_n = np.array(luminaire_n)
    wall_center = np.array(wall_center)
    wall_n = np.array(wall_n)
    floor_center = np.array(floor_center)
    floor_n = np.array(floor_n)
    
    
    (xm_map,ym_map,zm_map,phim_map,thetam_map,
        values_map) = get_cart_lid_map(LID,
                                       grid_interp_method = grid_interp_method, 
                                       theta_min = theta_min, 
                                       angle_res = angle_res) 
    
    
    # Setup sensor:
    (sensor_position, sensor_corners, 
     sensor_center, sensor_n, 
     sensor_pixels, drays) = _create_sensor_plane(fov = fov, Fd = Fd, 
                                                 position = sensor_position, 
                                                 n = sensor_n, res = res)
    
    # setup wall:
    wall_corners, wall_n, _ = _create_plane(w = wall_width, h = wall_height, center = wall_center, n = wall_n, res = None)
    
    # setup floor:
    floor_corners, floor_n, _ = _create_plane(w = floor_width, h = floor_height, center = floor_center, n = floor_n, res = None)

    # Trace rays from viewpoint (0,0,0) through camera pixels and calculate
    # the intersection points with the wall(s) and floor:
    intersectionpoints_wall, dist_sensor_wall = _get_line_plane_intersection(sensor_position,drays, wall_center, wall_n, wall_corners)
    intersectionpoints_floor, dist_sensor_floor = _get_line_plane_intersection(sensor_position,drays, floor_center, floor_n, floor_corners)

    # Get luminance values for wall and floor:
    L_wall,luminaire_n_, dist_source_wall, (xm_map_r,ym_map_r,zm_map_r) = _get_plane_luminance(plum = luminaire_position, nlum = luminaire_n, lid = LID, pplane = intersectionpoints_wall, nplane = wall_n, psensor = sensor_position, rho = wall_rho, xyzm_maps=(xm_map,ym_map,zm_map))
    L_floor, _, dist_source_floor, _ = _get_plane_luminance(plum = luminaire_position, nlum = luminaire_n, lid = LID, pplane = intersectionpoints_floor, nplane = floor_n, psensor = sensor_position, rho = floor_rho, xyzm_maps = None)

    # Pool wall and floor intersection points and luminance values:
    intersectionpoints = intersectionpoints_wall.copy()  
    Lvs = L_wall.copy()
    
    cond = np.logical_not(np.isnan(intersectionpoints_floor[:,0])) & (dist_sensor_floor <= dist_sensor_wall)
    intersectionpoints[cond,:] = intersectionpoints_floor[cond,:]
    Lvs[cond] = L_floor[cond]
    
    # check occlusion of floor by wall with respect to source:
    cond = np.isnan(intersectionpoints_wall[:,0]) & (dist_source_floor > dist_source_wall)  # also check distance to source
    intersectionpoints[cond,:] = np.nan
    Lvs[cond] = 0
        
    Lvs[np.isnan(Lvs)] = 0 # were non-intersecting or blocked rays (only 1 bounce !!!)
    maxL = Lvs.max()
    Lv2D = np.flipud(np.reshape(Lvs,(res,res)).T)
    

    # Make plots:
    if ax3D is None:
        fig3D = plt.figure()
        if join_axes:
            if ax2D == False: 
                ax3D = fig3D.add_subplot(111,projection='3d')
            else:
                ax3D = fig3D.add_subplot(121,projection='3d')
                ax2D = fig3D.add_subplot(122)
        else:
            ax3D = fig3D.add_subplot(111,projection='3d')
    
    if ax2D is None:
        fig2D = plt.figure()
        ax2D = fig2D.add_subplot(111)

    if ax3D != False:
        if plot_luminaire_position:
            dv = 1.2
            ax3D.plot(luminaire_position[:,0],luminaire_position[:,1],luminaire_position[:,2],color = 'y',marker='p',markersize=14,label='luminaire')
            ax3D.plot(np.vstack((luminaire_position[:,0],luminaire_position[:,0]+dv*luminaire_n_[0]))[:,0],
                      np.vstack((luminaire_position[:,1],luminaire_position[:,1]+dv*luminaire_n_[1]))[:,0],
                      np.vstack((luminaire_position[:,2],luminaire_position[:,2]+dv*luminaire_n_[2]))[:,0],'y-',linewidth = 3)

        if plot_sensor_pixels:
            ax3D.plot(sensor_pixels[:,0],sensor_pixels[:,1],sensor_pixels[:,2],'g.',label = 'sensor pixels',alpha=0.5)
        if plot_sensor_position:
            dv = 0.1
            ax3D.plot(sensor_position[:,0],sensor_position[:,1],sensor_position[:,2],'ro', label = 'sensor focal point',alpha=0.5)
            ax3D.plot(np.vstack((sensor_position[:,0],sensor_position[:,0]+dv*sensor_n[0]))[:,0],
                      np.vstack((sensor_position[:,1],sensor_position[:,1]+dv*sensor_n[1]))[:,0],
                      np.vstack((sensor_position[:,2],sensor_position[:,2]+dv*sensor_n[2]))[:,0],'r-',linewidth = 3)
        if plot_sensor_rays:
            # for i in range(sensor_pixels.shape[0]):
            #     ax3D.plot(np.vstack((sensor_pixels[i,0],sensor_position[:,0]))[:,0],
            #             np.vstack((sensor_pixels[i,1],sensor_position[:,1]))[:,0],
            #             np.vstack((sensor_pixels[i,2],sensor_position[:,2]))[:,0],'g.-') 
            t = 3
            for i in range(drays.shape[0]):
                if i == 0:
                    ax3D.plot(np.vstack((sensor_position[:,0]+t*drays[i,0],sensor_position[:,0]))[:,0],
                        np.vstack((sensor_position[:,1]+t*drays[i,1],sensor_position[:,1]))[:,0],
                        np.vstack((sensor_position[:,2]+t*drays[i,2],sensor_position[:,2]))[:,0],'c.:',label = 'sensor rays',alpha = 0.4) 
                else:
                    ax3D.plot(np.vstack((sensor_position[:,0]+t*drays[i,0],sensor_position[:,0]))[:,0],
                        np.vstack((sensor_position[:,1]+t*drays[i,1],sensor_position[:,1]))[:,0],
                        np.vstack((sensor_position[:,2]+t*drays[i,2],sensor_position[:,2]))[:,0],'c.:',alpha = 0.4) 
        
        if plot_lumiaire_rays:
            t = 3
            drays_ = intersectionpoints - luminaire_position
            drays_ = drays_/_norm(drays_,axis=1)
            for i in range(drays_.shape[0]):
                if i == 0:
                    ax3D.plot(np.vstack((luminaire_position[:,0]+t*drays_[i,0],luminaire_position[:,0]))[:,0],
                        np.vstack((luminaire_position[:,1]+t*drays_[i,1],luminaire_position[:,1]))[:,0],
                        np.vstack((luminaire_position[:,2]+t*drays_[i,2],luminaire_position[:,2]))[:,0],'y.:',label = 'luminaire rays',alpha = 0.4) 
                else:
                    ax3D.plot(np.vstack((luminaire_position[:,0]+t*drays_[i,0],luminaire_position[:,0]))[:,0],
                        np.vstack((luminaire_position[:,1]+t*drays_[i,1],luminaire_position[:,1]))[:,0],
                        np.vstack((luminaire_position[:,2]+t*drays_[i,2],luminaire_position[:,2]))[:,0],'y.:',alpha = 0.4) 
        if plot_luminaire_lid:
            ax3D.scatter(xm_map_r.ravel(),ym_map_r.ravel(),zm_map_r.ravel(),c=values_map.ravel(),marker='.',alpha = 0.5)
                    
        if plot_wall_intersections:
            ax3D.plot(intersectionpoints_wall[:,0],intersectionpoints_wall[:,1],intersectionpoints_wall[:,2],'bo', label = 'wall-ray intersection')
        if plot_floor_intersections:
            ax3D.plot(intersectionpoints_floor[:,0],intersectionpoints_floor[:,1],intersectionpoints_floor[:,2],'mo', label = 'floor-ray intersections')
        if plot_wall_edges:
            _plot_plane_edges(wall_corners, ax3D, color = 'b', marker = 's')
            dv = 0.1
            ax3D.plot(wall_center[0],wall_center[1],wall_center[2],'b.')
            ax3D.plot(np.vstack((wall_center[0],wall_center[0]+dv*wall_n[0]))[:,0],
                      np.vstack((wall_center[1],wall_center[1]+dv*wall_n[1]))[:,0],
                      np.vstack((wall_center[2],wall_center[2]+dv*wall_n[2]))[:,0],'b-',linewidth = 3)

        if plot_floor_edges:
            _plot_plane_edges(floor_corners, ax3D, color = 'm', marker = 's')
            dv = 0.1
            ax3D.plot(floor_center[0],floor_center[1],floor_center[2],'m.')
            ax3D.plot(np.vstack((floor_center[0],floor_center[0]+dv*floor_n[0]))[:,0],
                      np.vstack((floor_center[1],floor_center[1]+dv*floor_n[1]))[:,0],
                      np.vstack((floor_center[2],floor_center[2]+dv*floor_n[2]))[:,0],'m-',linewidth = 3)
        
        if plot_wall_luminance:
            ax3D.scatter(intersectionpoints_wall[:,0],intersectionpoints_wall[:,1],intersectionpoints_wall[:,2], c = L_wall,cmap='gray',vmin = 0, vmax=maxL)
        if plot_floor_luminance:
            ax3D.scatter(intersectionpoints_floor[:,0],intersectionpoints_floor[:,1],intersectionpoints_floor[:,2], c = L_floor,cmap='gray',vmin = 0, vmax=maxL)
        if legend_on:
            ax3D.legend()
        
    if ax2D != False:
        ax2D.imshow(Lv2D, cmap='gray',vmin = 0, vmax = maxL)
        ax2D.set_xticks([])
        ax2D.set_yticks([])
        
    return eval(out)


if __name__ == '__main__':
    
    # Read lamp data from IES file:
    IES = read_lamp_data('./data/luxpy_test_lid_file2.ies', verbosity = 1)
    
    # # Generate uv-map for rendering / ray-tracing (eg by wrapping this around 
    # # a point light source to attenuate the luminous intensity in different directions):
    # uv_map = get_uv_texture(theta = IES['theta'], 
    #                           phi = IES['phi'], 
    #                           values = IES['values'], 
    #                           input_types = ('array','mesh'), 
    #                           method = 'linear', 
    #                           theta_min = 0, angle_res = 1,
    #                           deg = True, r = 1, 
    #                           show = True)
    # plt.figure()
    # plt.imshow(uv_map)
    
        
    # draw LID:
    draw_lid(IES)
    
    # # Render LID
    # Lv2D = render_lid(IES, sensor_resolution = 40,
    #                     sensor_position = [0,-1,0.8], sensor_n = [0,1,-0.2], fov = (90,90), Fd = 2,
    #                     luminaire_position = [0,1.3,2], luminaire_n = [0,0,-1],
    #                     wall_center = [0,2,1], wall_n = [0,-1,0], wall_width = 4, wall_height = 2, wall_rho = 1,
    #                     floor_center = [0,1,0], floor_n = [0,0,1], floor_width = 4, floor_height = 2, floor_rho = 1,
    #                     ax3D = axs[1], ax2D = axs[2], join_axes = False, 
    #                     plot_luminaire_position = True, plot_lumiaire_rays = False, plot_luminaire_lid = True,
    #                     plot_sensor_position = True, plot_sensor_pixels = False, plot_sensor_rays = False, 
    #                     plot_wall_edges = True, plot_wall_luminance = True, plot_wall_intersections = False,
    #                     plot_floor_edges = True, plot_floor_luminance = True, plot_floor_intersections = False,
    #                     out = 'Lv2D')
    
    # or combine draw and render (but use only 2D image):
    fig = plt.figure(figsize=[14,7])
    axs = [fig.add_subplot(131,projection='3d'), fig.add_subplot(132,projection='3d'),fig.add_subplot(133)]
    draw_lid(IES, ax = axs[0])
    Lv2D = render_lid(IES, sensor_resolution = 100,
                       sensor_position = [0,-1,0.8], sensor_n = [0,1,-0.2], fov = (90,90), Fd = 2,
                       luminaire_position = [0,1.3,2], luminaire_n = [0,0,-1],
                       wall_center = [0,2,1], wall_n = [0,-1,0], wall_width = 4, wall_height = 2, wall_rho = 1,
                       floor_center = [0,1,0], floor_n = [0,0,1], floor_width = 4, floor_height = 2, floor_rho = 1,
                       ax3D = axs[1], ax2D = axs[2], join_axes = False, 
                       plot_luminaire_position = True, plot_lumiaire_rays = False, plot_luminaire_lid = True,
                       plot_sensor_position = True, plot_sensor_pixels = False, plot_sensor_rays = False, 
                       plot_wall_edges = True, plot_wall_luminance = True, plot_wall_intersections = False,
                       plot_floor_edges = True, plot_floor_luminance = True, plot_floor_intersections = False,
                       out = 'Lv2D')

    
    

        
    
        
  
