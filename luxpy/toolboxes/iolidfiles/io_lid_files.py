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

 :read_lamp_data: Read in light intensity distribution and other lamp data from LDT or IES files.

    Notes:
        1.Only basic support. Writing is not yet implemented.
        2.Reading IES files is based on Blender's ies2cycles.py
        3.This was implemented to build some uv-texture maps for rendering and only tested for a few files.
        4. Use at own risk. No warranties.
     
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import np, os 

__all__ =['read_lamp_data']


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
        IES['values'] = np.matlib.repmat(IES['candela_2d'],4,1)
        IES['phi'] = np.hstack((IES['h_angs'], IES['h_angs'] + 90, IES['h_angs'] + 180, IES['h_angs']+270))
    elif IES['lamp_h_type'] == 'TYPE180':
        IES['values'] = np.matlib.repmat(IES['candela_2d'],2,1)
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