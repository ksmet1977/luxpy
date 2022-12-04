# -*- coding: utf-8 -*-
"""
Module for TechnoTeam LMK camera basic control
==============================================

 * lmkActiveX: class for basic LMK control 
 
 * _CAMERAS: dictionary with supported camera-lens combinations 
 
 * define_lens(): to define new lens dictionary
 
 * kill_lmk4_process(): kill the lmk4.exe process in case some error happens 
 
 * read_pcf(): read TechnoTeam pcf file. (output = CIE-RGB)
 
 * plot_pcf(): make a plot of a pcf image. 

 * pcf_to_xyz(): convert pcf image to an XYZ image

 * ciergb_to_xyz(): convert CIE-RGB to XYZ 
 
 * xyz_to_ciergb(): convert XYZ to CIE-RGB

Created on Sat Nov 26 10:52:42 2022

@author: ksmet1977
"""

import os
import time 
import traceback
import subprocess
import copy
import configparser
import numpy as np

import matplotlib.pyplot as plt

from luxpy.utils import is_importable


# import win32 (and if necessary install it):
success = is_importable('win32', pip_string = 'pywin32', try_pip_install = True)
if success:
    import win32
    from win32com.client import Dispatch # pip install pywin32

# import easygui(and if necessary install it):
success = is_importable('easygui', try_pip_install = True)
if success:
    import easygui


__all__ = ['_LABSOFT_PATH','_LABSOFT_CAMERA_PATH','_CAMERAS',
           'get_labsoft_path','define_lens', 'lmkActiveX', 'kill_lmk4_process',
           'read_pcf', 'plot_pcf', 'pcf_to_xyz', 'ciergb_to_xyz','xyz_to_ciergb']

#------------------------------------------------------------------------------
#_LABSOFT_PATH = r'C:/TechnoTeam/LabSoft/'

def get_labsoft_path():
    labsoft_path_file = os.path.join(os.path.dirname(__file__),'labsoft_path.txt')
    with open(labsoft_path_file) as fid:
       labsoft_path = fid.readlines()[0]

    if not os.path.exists(os.path.join(labsoft_path,'bin','lmk4.exe')):
        print('Could not find {:s}.'.format(os.path.join(labsoft_path,'bin','lmk4.exe')))
        print('Make sure the installation path of the labsoft software package in file {:s} is correct.'.format(labsoft_path_file))
        raise FileNotFoundError
    return labsoft_path

_LABSOFT_PATH = get_labsoft_path()
_LABSOFT_CAMERA_PATH = os.path.join(_LABSOFT_PATH,r'Camera') + '/'



#==============================================================================
# Helper functions / classes
#============================================================================== 


#------------------------------------------------------------------------------
def define_lens(lens_type, name, focusFactors = None):
    """ Define a technoteam lens """
    lens = {'name': name}
    if focusFactors is not None:
        ff_base = lens_type[1:lens_type.find('mm')-1]
        ff_base = ff_base[:ff_base.find('_')]
        ff_base = 'TT' + ff_base + 'Scale'
        if isinstance(focusFactors, int):
            focusFactors = (['{:02.0f}'.format(focusFactors)],[focusFactors])
        elif isinstance(focusFactors,dict):
            focusFactors = (list(focusFactors.keys()), list(focusFactors.values()))
        elif isinstance(focusFactors,list):
            focusFactors_str = []
            for ff in focusFactors:
                tmp = ff if isinstance(ff,str) else '{:02.0f}'.format(ff)
                focusFactors_str.append(tmp)
            focusFactors = (focusFactors_str,[i for i in range(len(focusFactors_str))])
        tmp = {('{:s}{:s}'.format(ff_base,i)) : j for i,j in zip(*focusFactors)}
        lens['focusFactors'] = tmp
    return lens

#------------------------------------------------------------------------------
def kill_lmk4_process(verbosity = 1):
    out = subprocess.run(["taskkill", "/F", "/IM", "lmk4.exe"], capture_output = True, # use: C:\Windows\System32\taskkill ?
                          stderr = None, text = True);
    if verbosity > 0: print(out)
    return out


#==============================================================================
# LMK control 
#============================================================================== 
# Define supported cameras and lenses:
    
    
_CAMERAS = {'ttf8847' : {
                            'name'   : 'ttf8847',
                            'lenses' : {'x6_5mm' : define_lens('x6_5mm','o13196f6_5',focusFactors = None),
                                        'x12mm'  : define_lens('x12mm','o95653f12',focusFactors = ['0_3','0_5','1','3','Infinite']),
                                        'x25mm'  : define_lens('x25mm','oB225463f25',focusFactors = [i for i in range(20)]),
                                        'x50mm'  : define_lens('x50mm','oC216813f50',focusFactors = [i for i in range(20)])
                                        }
                        },
    
            'tts20035' : {
                        'name'   : 'tts20035',
                        'lenses' : {'x50mm_M00442'   : define_lens('x50mm_M00442','oM00442f50',focusFactors = [i for i in range(24)]),
                                    'x12mm_TTC_163'  : define_lens('x12mm_TTC_163','oTTC-163_D0224',focusFactors = None),
                                    'x12f50mm_2mm'   : define_lens('x12f50mm_2mm','oTTNED-12_50_2mmEP',focusFactors = None),
                                    'x12f50mm_4mm'   : define_lens('x12f50mm_4mm','oTTNED-12_50_4mmEP',focusFactors = None),
                                    'xvr'            : define_lens('xvr','oTTC-163_D0224',focusFactors = None)    
                                    }

                         }
            }


class lmkActiveX:
    """ 
    Class for TechnoTeam LMK camera basic control
    
    All supported camera/lens combinations are defined in: _CAMERAS
    To add new ones (or new lenses): edit the_CAMERAS dict
    """
    
    # main object containing the lmk ActiveX server object
    lmk = None 
    
    # Verbosity level:
    verbosity_levels = {0 : 'none',
                        1 : 'minimal',
                        2 : 'moderate (default)',
                        3 : 'Detailed',
                        4 : 'All'
                       }
    verbosity = 2
        
    workingImage = None
    errorFlag = None
    
    colorSpace = {
                'CIE-RGB'       : 1,
                'S-RGB'         : 2,
                'EBU-RGB'       : 4,
                'XYZ'           : 16,
                'Lxy'           : 32,
                'Luv'           : 64,
                'Lu_v_'         : 128,
                'L*u*v*'        : 256,
                'C*h*s*_uv'     : 512,
                'L*a*b*'        : 1024,
                'C*h*_ab'       : 2048,
                'HSV'           : 4096,
                'HSI'           : 8192,
                'WST'           : 16384,
                'Lrg'           : 32768,
                'LWS'           : 65536
                }
                
    imageType = {
                'Camera'        : -3,
                'Luminance'     : -2,
                'Color'         : -1,
                'Evaluation[1]' :  0,
                'Evaluation[2]' :  1,
                'Evaluation[3]' :  2,
                'Evaluation[4]' :  3,
                'Evaluation[5]' :  4
                }

    regionType = {
                'Rectangle': {
                    'identifier'    : 0,
                    'points'        : 2
                    },
                'Line': {
                    'identifier'    : 1,
                    'points'        : 2
                    },
                'Circle': {
                    'identifier'    : 2,
                    'points'        : 2
                    },
                'Polygon': {
                    'identifier'    : 3,
                    'points'        : 3
                    },
                'Polyline': {
                    'identifier'    : 4,
                    'points'        : 3
                    },
                'Ellipse': {
                    'identifier'    : 5,
                    'points'        : 3
                    },
                'CircularRing': {
                    'identifier'    : 6,
                    'points'        : 3
                    },
                'OR': {
                    'identifier'    : 7,
                    'points'        : 2
                    },
                'XOR': {
                    'identifier'    : 8,
                    'points'        : 2
                    },
                'AND': {
                    'identifier'    : 9,
                    'points'        : 2
                    }
                }
    
    statisticType = {
                    'standardGrey'          : 0,   # 0 	Standard statistic in grey images
                    'standardColor'         : 1,   # 1 	Standard statistic in color images
                    'sectionalGrey'         : 2,   # 2 	Sectional view in grey images
                    'sectionalColor'        : 3,   # 3 	Sectional view in color images
                    'histogramGrey'         : 4,   # 4 	Histogram in grey images
                    'histogramColor'        : 5,   # 5 	Histogram in color images
                    'bitHistogramGrey'      : 6,   # 6 	Bit histogram in grey images (only images of camera image type)
                    'bitHistorgramColor'    : 7,   # 7 	Bit histogram in color images (only images of color camera image type)
                    'projectionGrey'        : 8,   # 8 	Projection in grey images
                    'projectionColor'       : 9,   # 9 	Projection in color images
                    'luminanceGrey'         :20,   # 20 Luminance objects in grey images
                    'integralGrey'          :22,   # 22 Integral objects in grey images
                    'integralColor'         :23,   # 23 Integral objects in color images
                    'symbolGrey'            :24,   # 24 Symbol objects in grey images
                    'symbolColor'           :25,   # 25 Symbol objects in color images
                    'lightArcGrey'          :26,   # 26 Light arc objects in grey images
                    'spiralWoundGrey'       :28,   # 28 Spiralwoundfilaments in grey images
                    'chromaticityLineColor' :31,   # 31 Chromaticity line diagrams in color images
                    'chromaticityAreaColor' :33,   # 33 Chromaticity area diagrams in color images
                    'threeDviewGrey'        :34,   # 34 3d view in grey images
                    'integralNegativeGrey'  :36,   # 36 Integral objects in grey images (negative contrast)
                    'integralNegativeColor' :38,   # 38 Symbol objects in grey images (negative contrast)
                    'symbolNegativeColor'   :39,   # 39 Symbol objects in color images (negative contrast)
                    'contrastGrey'          :40    # 40 Contrasts objects in grey images
                    }
                    
    camera = copy.deepcopy(_CAMERAS)

            
    # properties capture images
    captureStartRatio       = 10
    captureMaxTries         =  3
    captureFactor           =  3
    captureCountPic         =  1
    captureDefaultMaxTries  =  3
        
    boolStr = ['False', 'True']
    
    # constructor method
    @classmethod
    def __init__(cls,camera, lens,focusfactor = None,
                 autoscan = True, autoexposure = True,
                 modfrequency = 60, maxtime = 10,
                 labsoft_camera_path = _LABSOFT_CAMERA_PATH, 
                 verbosity = None):
                 
        bool2num = lambda x: int(x*1) 

        # process camera and lens input:
        if (camera is None) & (lens is None):
            return cls.camera # so user can have a look at implemented camera's
        cls.camera = cls.camera[camera] if isinstance(camera,str)  else camera
        if isinstance(lens,str): 
            if lens in cls.camera['lenses'].keys():
                cls.lens = cls.camera['lenses'][lens]
            else:
                raise Exception('Lens {:s} not in lensList for camera {:s}'.format(lens, camera['name']))
        elif isinstance(lens,dict):
            cls.lens = lens 
        else:
            raise Exception ('lens must be str or dict object')
                    
        cls.focusfactor = focusfactor 
        cls.autoscan = bool2num(autoscan) # Determination of a good exposure time before the capturing algorithm.
        cls.autoexposure = bool2num(autoexposure) # all exposure times will automatically adjusted if camera exposure time is reduced or enlarged.
        cls.modfrequency = modfrequency 
        cls.maxtime = maxtime
        
        if verbosity is not None: cls.verbosity = verbosity 
        
        cls.labsoft_camera_path = labsoft_camera_path
        cls.objectiveCalibrationPath = cls.labsoft_camera_path + cls.camera['name'].upper() + '/' + cls.lens['name'] + '/' 
        
        if cls.lmk is not None:
            if cls.verbosity > 1: print("Already connected to LMK Labsoft ! Close connection first.")
        else:
            if (cls.objectiveCalibrationPath is None):
                if cls.verbosity > 0: print("WARNING: Cannot connect to lmk with a not-None objectiveCalibrationPath !")
            else:
                # initialize the lmk activeX software and object
                cls.init()

    @classmethod 
    def show_labsoft_gui(cls, show = 3):
        
        cls.show_gui = show
        cls.answer = cls.lmk.iShow(cls.show_gui)
        
        if (cls.answer != 0):
            if cls.verbosity > 0: print('WARNING: show_labsoft_gui(): Problem setting the visibility of the LabSoft GUI.\n')
        
            cls.answer, errorText = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0:
                print('errorText',errorText)
                print('LabSoft error information: {:s}\n'.format(errorText))
            
            return cls.answer
        
    
    @classmethod
    def open_lmk_labsoft_connection(cls, objectiveCalibrationPath = None, show_gui = 3):
        """
        Initializes a connection to LMK LabSoft.

          Input:
            -objectiveCalibrationPath: path to calibration file
          Output:
            -answer: 0=no error, other=error code
        """
        if objectiveCalibrationPath is not None: cls.objectiveCalibrationPath = objectiveCalibrationPath
        if cls.objectiveCalibrationPath is None:
            if cls.verbosity > 0: 
                print('WARNING: open_lmk_labsoft_connection(): no objective path specified. Aborting...\n')
            cls.answer = 171
            for i in range(10):
                #winsound.Beep(400,200)
                print('.')
                time.sleep(0.5)
            return cls.answer
            
        cls.answer = 0
        
        # Establish connection to LMK LabSoft via ActiveX-Server
        cls.lmk = Dispatch('lmk4.LMKAxServer')
        
        # Open LabSoft, necessary command:
        cls.answer = cls.lmk.iOpen()
        
        # temporary fix for license issue:
        if (cls.answer == 137363456):
            easygui.msgbox('When finished continue', 'Close dialog box in LabSoft GUI', 'continue')
            cls.answer = 0
            
        if (cls.answer != 0):
            if cls.verbosity > 0: print('WARNING: open_lmk_labsoft_connection(): Problem opening the LabSoft program. Aborting...\n')
        
            cls.answer, errorText = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0:
                print('errorText',errorText)
                print('LabSoft error information: {:s}\n'.format(errorText))
            
            return cls.answer
        
        return cls.answer

    
    @classmethod
    def close_lmk_labsoft_connection(cls, open_dialog = 0):
        """
        Closes the connection to LMK LabSoft and LabSoft itself.

        Input:
          - open_dialog:
                If 0: No dialog window.
                Else: Opens a dialog window in the Labsoft application.
                      The user can choose whether they wish to save 
                      the current state or not or or cancel 
                      the closing of LabSoft.
        Output:
          -answer: 0=no error, other=error code
        """
        if cls.lmk is None:
            if cls.verbosity > 1: print('close_lmk_labsoft_connection(): lmk is closed.\n')
            return None
        if str(type(cls.lmk)) != "<class 'win32com.client.CDispatch'>": # original matlab: if (isa(obj.lmk, 'COM.lmk4_LMKAxServer') == 0)
            if cls.verbosity > 0: print('WARNING: close_lmk_labsoft_connection(): lmk is not a LabSoft-Handle\n')
            cls.answer = 112
            return cls.answer
        
        cls.answer = 0
        try:
            # Close program
            cls.answer = cls.lmk.iClose(open_dialog)
        except:
            if cls.verbosity > 0: print('WARNING: close_lmk_labsoft_connection(): Problem closing the program\n.')
            cls.answer, errorText = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('LabSoft error information: {:s}\n'.format(errorText))
            
        if (cls.answer != 0):
            if cls.verbosity > 0: print('WARNING: close_lmk_labsoft_connection(): Problem closing the program\n.')
            cls.answer, errorText = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('LabSoft error information: {:s}\n'.format(errorText))
        
        del cls.lmk
        cls.lmk = None
        # kill the process of LabSoft to avoid complications at the next start
        out_subprocess = kill_lmk4_process(verbosity = cls.verbosity)
        
        cls.answer, text = out_subprocess.returncode, out_subprocess.stdout + out_subprocess.stderr
        if (cls.answer != 0):
            if cls.verbosity > 0: print("WARNING: close_lmk_labsoft_connection(): Couldn't correctly close the program\n")
            if cls.verbosity > 0: print('subprocess error information: {:s}\n'.format(text))
            cls.answer = 113
            return cls.answer
        else:
            if cls.verbosity > 1: print('Process killed succesfully!')
            return cls.answer

    # destructor method
    @classmethod
    def delete(cls, open_dialog = 0):
        """ Delete lmk class object (close connection to labsoft) """
        cls.close_lmk_labsoft_connection(open_dialog = open_dialog)
        
    @classmethod
    def __del__(cls, open_dialog = 0):
        """ Delete lmk class object (close connection to labsoft) """
        try: 
            cls.close_lmk_labsoft_connection(open_dialog = open_dialog)
        except:
            cls.close_lmk_labsoft_connection(open_dialog = open_dialog)
            print("Could not delete lmkActiveX class object.")
            print('  Kill lmk4.exe process manually --> run kill_lmk4_process()')
        
    @classmethod
    def setWorkingImage(cls, w):
        """ Set the current working image """
        cls.workingImage = w
        
    
    @classmethod        
    def display_error_info(cls, err_code, process_id = ''):
        """ Get the info for err_code and print """
        if err_code == 0:
            print('{:s} ok!\n'.format(process_id))
        else:
            print('Error {:s}: \n'.format(process_id))
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            print('\tError: {} \t info: {}\n'.format(cls.answer,errorInfo))
    
    @classmethod
    def get_filter_wheel_info(cls):
        """
        Get max. number of filter wheels and their names.
        """
        config = configparser.ConfigParser()
        config.read(cls.labsoft_camera_path + '/' + cls.camera['name'] + '/' + 'camera.ini')
        filter_wheel_max = int(config.get('PropertyList', 'FILTER_WHEEL_MAX'))
        filter_wheel_names = config.get('PropertyList', 'FILTER_WHEEL_NAMES')
        filter_wheel_names = filter_wheel_names.split(' ')
    
        if type(filter_wheel_names) == int:
            filter_wheel_names = np.vstack(filter_wheel_names).astype(int)
        else:
            filter_wheel_names = np.vstack(filter_wheel_names)
    
        return filter_wheel_max, filter_wheel_names
    
    @classmethod
    def measureColorMultipic(cls, countPic = None, defaultMaxTries = None):
        """ Capture a ColorMultiPicture """
        currTries = 0
        if countPic is None: countPic = cls.captureCountPic
        if defaultMaxTries is None: defaultMaxTries = cls.defaultMaxTries
        
        # Capture image
        if cls.verbosity > 1: print('Starting capturing ColorMultiPic image.\n')
        
        # Capture image
        tempFlag = 1
        while (currTries <= defaultMaxTries) and (tempFlag != 0):
            tempFlag, exposure_times = cls.lmk.iColorCaptureMultiPic(['']*cls.filter_wheel_max,countPic)
            if tempFlag == 0:
                if cls.verbosity > 1: 
                    print('ColorMultiPic: image captured.\n')
                    print('Filter wheel exposure times: {}\n'.format(exposure_times))
            else:
                currTries = currTries + 1
                if cls.verbosity > 0: print('Error ColorMultiPic: Try {:1.0f}/{:1.0f}\n'.format(currTries,defaultMaxTries))
                cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
                if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))

        if tempFlag != 0:
            if cls.verbosity > 1: print('Capture image failed!\n')
            

    @classmethod
    def saveImage(cls,folderXYZ,fileNameXYZ):
        """ Save the captured image currently as workingImage to the specified file and folder """
        imgPath = os.path.join(folderXYZ, fileNameXYZ + '.pcf')
        tempFlag = cls.lmk.iSaveImage(cls.workingImage,imgPath)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iSaveImage: image saved: {:s}\n\n'.format(imgPath))
        else:
            if cls.verbosity > 0: print('Error iSaveImage: error saving {:s}\n\n'.format(imgPath))
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
            
    @classmethod
    def loadImage(cls,pathXYZ):
        """ Load a previously captured image from a specific path """
        if pathXYZ[-4:] == '.pcf':
            tempIdxImage = -1
        elif pathXYZ[-3:] == '.pf':
            tempIdxImage = -2
        else:
            if cls.verbosity > 0: print('Could not load image, wrong extension\n')
            return None

        tempFlag = cls.lmk.iLoadImage(tempIdxImage, pathXYZ) # add folder ??
        if tempFlag == 0:
            if cls.verbosity > 1: print('Image {:s} loaded in imageIdx={:0.0f}.\n'.format(pathXYZ,tempIdxImage))
            cls.workingImage = tempIdxImage
        else:
            if cls.verbosity > 0: print('Error iLoadImage: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))


    @classmethod
    def getStatistic(cls,statisticType,regionName,colorSpace):
        """ Get Cmin, Cmax, Cmean, Cvar for specified colorSpace for a specific region """
        tmpA, tmpB = cls.lmk.iCreateStatistic(statisticType, cls.workingImage, cls.getRegionIndexByName(regionName),1,['1'])
        tmpA, tmpB, tmpC = cls.lmk.iGetStatisticselfect(statisticType, 0, cls.workingImage, cls.getRegionIndexByName(regionName))
            
        # Get the statistic (always in RGB values):
        tmp, RirSize, RdrMin, RdrMax, RdrMean, RdrVar = cls.lmk.iGetStandardStatistic2(statisticType,0,2,0,0,0,0,0)
        tmp, GirSize, GdrMin, GdrMax, GdrMean, GdrVar = cls.lmk.iGetStandardStatistic2(statisticType,0,1,0,0,0,0,0)
        tmp, BirSize, BdrMin, BdrMax, BdrMean, BdrVar = cls.lmk.iGetStandardStatistic2(statisticType,0,0,0,0,0,0,0)

        # pause
        
        # Convert these values to the given colorspace:
        tmp, CminX, CminY, CminZ     = cls.lmk.iGetColor(RdrMin,GdrMin, BdrMin, 0, 0, 0, colorSpace, 0, 0, 0)
        tmp, CmaxX, CmaxY, CmaxZ     = cls.lmk.iGetColor(RdrMax, GdrMax, BdrMax, 0, 0, 0, colorSpace, 0, 0, 0)
        tmp, CmeanX, CmeanY, CmeanZ  = cls.lmk.iGetColor(RdrMean, GdrMean, BdrMean, 0, 0, 0, colorSpace, 0, 0, 0)
        tmp, CvarX, CvarY, CvarZ     = cls.lmk.iGetColor(RdrVar, GdrVar, BdrVar, 0, 0, 0, colorSpace, 0, 0, 0)
        Cmin = [CminX, CminY, CminZ]
        Cmax = [CmaxX, CmaxY, CmaxZ]
        Cmean = [CmeanX, CmeanY, CmeanZ]
        Cvar = [CvarX, CvarY, CvarZ]
        
        #delete the statistic object:
        cls.lmk.iDeleteStatistic(statisticType, 0)
        return  Cmin, Cmax, Cmean, Cvar
        
    @classmethod 
    def get_color_autoscan_times(cls):
        tempFlag, autoscantimes = cls.lmk.iColorAutoScanTime([0 for i in cls.filter_wheel_names])
        if tempFlag == 0:
            if cls.verbosity > 1: print('iColorAutoScanTime: ColorAutoScanTimes {}\n'.format(autoscantimes))
        else:
            if cls.verbosity > 0: print('Error iColorAutoScanTime: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return autoscantimes

        
    @classmethod
    def measureCapture_X_map(cls, folder, fileName, X_type = 'XYZ', 
                             startRatio = None, factor = None, 
                             countPic = None, defaultMaxTries = None,
                             autoscan = None, autoexposure = None,
                             modfrequency = None, maxtime = None):
        """ 
        Measure XYZ / Y image and save as .pcf / .pf image. 
        (parameters as set in class attributes) 
        """
        
        if startRatio is None: startRatio = cls.captureStartRatio
        if factor is None: factor = cls.captureFactor
        if countPic is None: countPic = cls.captureCountPic
        if defaultMaxTries is None: defaultMaxTries = cls.captureDefaultMaxTries
        currTries = 0
        
        cls.set_autoscan(autoscan)
        cls.set_autoexposure(autoexposure)
        cls.set_mod_frequency(modfrequency) 
        cls.set_max_exposure_time(maxtime)
        
        # setWeorkingImage:
        if X_type == 'XYZ':
            cls.setWorkingImage(cls.imageType['Color'])
        elif X_type == 'Y':
            cls.setWorkingImage(cls.imageType['Luminance'])
        
        # Capture image:
        tempFlag = 1
        while (currTries <= defaultMaxTries) and (tempFlag != 0):
            if cls.verbosity > 1: print('Starting capturing {}-image: {} \n'.format(X_type,fileName))
    
            if X_type == 'XYZ':
                tempFlag, capture_XImg = cls.lmk.iColorCaptureHighDyn([''], startRatio, factor, countPic)
                fcn_name = 'iColorCaptureHighDyn'
            elif X_type == 'Y':
                tempFlag = cls.lmk.iHighDynPic3(np.nan, startRatio, factor, countPic) # first arg was 0 ??
                capture_XImg = None # iHighDynPic3 doesn't output image, but iColorCaptureHighDyn does ?
                fcn_name = 'iHighDynPic3'
                
            else:
                if cls.verbosity > 0: print("WARNING: Don't know what to capture as X_type = {:s} is unknown (options 'XYZ', 'Y')".format(X_type))

            if tempFlag == 0:
                if cls.verbosity > 1: print('{:s}: image captured\n'.format(fcn_name))
            else:
                currTries = currTries + 1
                if cls.verbosity > 0: print('Error {:s}: Try {:1.0f}/{:1.0f}\n'.format(fcn_name,currTries,defaultMaxTries))
                cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
                if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
                
        if tempFlag != 0:
            if cls.verbosity > 1: print('Capture {}-image failed:{}\n'.format(X_type,fileName))
        
        # Save image:
        if X_type == 'XYZ': 
            imgPath = os.path.join(folder, fileName + '.pcf')
            tempFlag = cls.lmk.iSaveImage(-1,imgPath)
        elif X_type == 'Y':
            imgPath = os.path.join(folder, fileName + '.pf')
            tempFlag = cls.lmk.iSaveImage(-2,imgPath)
        
        if tempFlag == 0:
            if cls.verbosity > 1:  print('iSaveImage: image saved: {:s}\n\n'.format(imgPath))
        else:
            if cls.verbosity > 0: print('Error iSaveImage: error saving {:s}\n\n'.format(imgPath))
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
            
    @classmethod 
    def measureCaptureXYZmap(cls, folderXYZ, fileNameXYZ,
                             startRatio = None, factor = None, 
                             countPic = None, defaultMaxTries = None,
                             autoscan = None, autoexposure = None,
                             modfrequency = None, maxtime = None):
        """ Measure XYZ image and save as .pcf image. (parameters as set in class attributes) """
        
        cls.measureCapture_X_map(folderXYZ, fileNameXYZ, X_type = 'XYZ',
                                 startRatio = startRatio, factor = factor, 
                                 countPic = countPic, defaultMaxTries = defaultMaxTries,
                                 autoscan = autoscan, autoexposure = autoexposure,
                                 modfrequency = modfrequency, maxtime = maxtime)  

    @classmethod
    def measureCaptureYmap(cls, folderY, fileNameY,
                           startRatio = None, factor = None, 
                           countPic = None, defaultMaxTries = None,
                           autoscan = None, autoexposure = None,
                           modfrequency = None, maxtime = None):
        """ Measure Y image and save as .pf image. (parameters as set in class attributes) """
    
        cls.measureCapture_X_map(folderY, fileNameY, X_type = 'Y',
                                 startRatio = startRatio, factor = factor, 
                                 countPic = countPic, defaultMaxTries = defaultMaxTries,
                                 autoscan = autoscan, autoexposure = autoexposure,
                                 modfrequency = modfrequency, maxtime = maxtime)
        

    @classmethod
    def createEllips(cls, centerPt, width, height, regionName):
        """ 
        Create an ellips with a:
            - centerPoint defined by centerPt (contains x and y value)
            - certain width (horizontal axis)
            - certain height (vertical axis)
            - give the ellips region a regionName (string)
        Function returns the regionIndex of the ellips
        """
        listXpts = ['{:1.0f}'.format(x) for x in [centerPt[0], centerPt[0], int(centerPt[0] + width)]]
        listYpts = ['{:1.0f}'.format(x) for x in [centerPt[1], int(centerPt[1] + height), centerPt[1]]]
        
        int32 = lambda x: np.array(x).astype(np.int32) if x is not None else None# check if iCreateRegion really does require int32 values as input (for now: pass python int)
        
        cls.errorFlag,_,_ = cls.lmk.iCreateRegion(int32(cls.workingImage),
                                                  int32(cls.regionType['Ellipse']['identifier']),
                                                  int32(cls.regionType['Ellipse']['points']),
                                                  listXpts,
                                                  listYpts)
            
        # Gets the number of regions defined in the workingImage:
        _,numberOfRegions = cls.lmk.iGetRegionListSize(cls.workingImage, 0)
            
        # Set the last region to the specified regionName:
        cls.lmk.iSetRegionName(cls.workingImage, int(numberOfRegions - 1), regionName)
        regionIndex = cls.getRegionIndexByName(regionName)
        return regionIndex
        
        
    @classmethod
    def createPolygon(cls, pointsXY, regionName):
        """ 
        Create a polygon with:
            - vertices specified in pointsXY (x->width, y->height)
            - give the polygon region a regionName (string)
        Function returns the regionIndex of the polygon
        """
        listXpts = ['{:1.0f}'.format(x) for x in pointsXY[0]]
        listYpts = ['{:1.0f}'.format(x) for x in pointsXY[1]]
        
        int32 = lambda x: np.array(x).astype(np.int32) if x is not None else None# check if iCreateRegion really does require int32 values as input (for now: pass python int)

        cls.errorFlag,_,_ = cls.lmk.iCreateRegion(
                                                 int32(cls.workingImage),
                                                 int32(cls.regionType['Polygon']['identifier']),
                                                 int32((len(listXpts))),
                                                 listXpts,
                                                 listYpts
                                                 )
            
        # Gets the number of regions defined in the workingImage:
        _,numberOfRegions = cls.lmk.iGetRegionListSize(cls.workingImage, 0)
            
        # Set the last region to the specified regionName:
        cls.lmk.iSetRegionName(cls.workingImage, int(numberOfRegions - 1), regionName)
        regionIndex = cls.getRegionIndexByName(regionName)
        return regionIndex
    
    @classmethod
    def createRectangle(cls, topleftXY, bottomrightXY, regionName):
        """ 
        Create a Rectangle spanning:
            - the top-left and bottom-right vertices 
            - give the rectangle region a regionName (string)
        Function returns the regionIndex of the rectangle
        """
        listXpts = ['{:1.0f}'.format(x) for x in [topleftXY[0],bottomrightXY[0]]]
        listYpts = ['{:1.0f}'.format(x) for x in [topleftXY[1],bottomrightXY[1]]]
        
        int32 = lambda x: np.array(x).astype(np.int32) if x is not None else None# check if iCreateRegion really does require int32 values as input (for now: pass python int)

        cls.errorFlag,_,_ = cls.lmk.iCreateRegion(
                                                 int32(cls.workingImage),
                                                 int32(cls.regionType['Polygon']['identifier']),
                                                 int32((len(listXpts))),
                                                 listXpts,
                                                 listYpts
                                                 )
            
        # Gets the number of regions defined in the workingImage:
        _,numberOfRegions = cls.lmk.iGetRegionListSize(cls.workingImage, 0)
            
        # Set the last region to the specified regionName:
        cls.lmk.iSetRegionName(cls.workingImage, int(numberOfRegions - 1), regionName)
        regionIndex = cls.getRegionIndexByName(regionName)
        return regionIndex
        
    @classmethod   
    def deleteRegionByName(cls, regionName):
        """ Delete a Region by regioName """
        cls.errorFlag = cls.lmk.iDeleteRegion(cls.workingImage, cls.getRegionIndexByName(regionName));
        cls.checkForError()
        
        
    @classmethod
    def getRegionIndexByName(cls,regionName):
        """ Return the index of a region with a region name set to regionName. """
        regionIndex = 0
        
        cls.errorFlag, regionIndex = cls.lmk.iGetIndexOfRegion(cls.workingImage, regionName, regionIndex)
        cls.checkForError()
        
        return regionIndex

    @classmethod
    def createStatisticObjectOfRegion(cls, regionName, statisticType):
        """ 
        Create a color statistic object
            call as follows:
            createStatisticObjectOfRegion('regionTestName',statisticType['standardColor'])
        """
        cls.lmk.iCreateStatistic(statisticType,cls.workingImage,cls.getRegionIndexByName(regionName),1,['1'])
        
        
    @classmethod
    def selectRegionByIndex(cls,ind,s):
        """
        Select a region by its index number s defines wether the region is selected or deselected (true or false)        
        """
        if s:
            s = 1
        else:
            s = 0

        cls.lmk.iSelectRegion(cls.workingImage,ind,s)

        
    @classmethod
    def selectRegionByName(cls,regionName,s):
        """ 
        Select a region by its region name, s defines wether the region is selected or deselected (true or false)
        """
        if s:
            s = 1
        else:
            s = 0
        cls.lmk.iSelectRegion(cls.workingImage,cls.getRegionIndexByName(regionName),s)

    @classmethod
    def setIntegrationTime(cls, wishedTime):
        """
        Set integration time.
        
          [int32, double] LMKAxServer::iSetIntegrationTime	(double _dWishedTime, double & _drRealizedTime)
          Parameters
         	_dWishedTime	Wished integration time
         	_drRealizedTime	Realized integration time
        """
        tempFlag, realizedTime = cls.lmk.iSetIntegrationTime(wishedTime,0)
        
        if tempFlag == 0:
            if cls.verbosity > 1: print('iSetIntegrationTime: \t{:1.2f}[s]\n'.format(realizedTime))
        else:
            if cls.verbosity > 0: print('Error iSetIntegrationTime: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
    
    @classmethod
    def getIntegrationTime(cls):
        """
        Get integration time.
        
          [int32, double, double, double, double, double]	LMKAxServer::iGetIntegrationTime
         	(handle, double _drCurrentTime, double & _drPreviousTime,
        		double & _drNextTime, double & _drMinTime, double & _drMaxTime )
        
          Determine current exposure time and other time parameters.
          Parameters
         	_drCurrentTime	Current integration time
         	_drPreviousTime	Next smaller (proposed) time
         	_drNextTime	Next larger (proposed) time
         	_drMinTime	Minimal possible time
         	_drMaxTime	Maximal possible time
        """
        tempFlag, currInt, prevInt, nextInt, minInt, maxInt = cls.lmk.iGetIntegrationTime(0,0,0,0,0)
        
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetIntegrationTime: currInt={:1.3f}[s] prevInt={:1.3f}[s] nextInt={:1.3f}[s] minInt={:1.3f}[s] maxInt={:1.3f}[s] \n'.format(currInt, prevInt, nextInt, minInt, maxInt))
        else:
            if cls.verbosity > 0: print('Error iGetIntegrationTime: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))

        return currInt, prevInt, nextInt, minInt, maxInt
     
    @classmethod
    def set_autoscan(cls, autoscan = None):
        """ 
        Set auto scan. 
        
        If the option Autoscan is on, then the exposure time of the camera is automatically
        determined before each capture by the autoscan algorithm. In the case of a
        color capture the autoscan algorithm is applied to each color filter separately.
        """ 
        if autoscan is not None: cls.autoscan = autoscan
        # if cls.autoscan == 1:
        tempFlag = cls.lmk.iSetAutoscan(int(1*cls.autoscan))
        if tempFlag == 0:
            if cls.verbosity > 1: print('iSetAutoscan: AutoScan {:s}\n'.format(cls.boolStr[cls.autoscan]))
        else:
            if cls.verbosity > 0: print('Error iSetAutoscan: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
            
    @classmethod
    def get_autoscan(cls):
        """ 
        Get auto scan. 
        """ 
        tempFlag, autoscan = cls.lmk.iGetAutoscan(0)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetAutoscan: AutoScan {:s}\n'.format(cls.boolStr[autoscan]))
        else:
            if cls.verbosity > 0: print('Error iGetAutoscan: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return autoscan
    
    @classmethod
    def set_autoexposure(cls, autoexposure = None):
        """ 
        Set Automatic-Flag for all exposure times. 
        
        If this flag is set, all exposure times will automatically 
        adjusted if camera exposure time is reduced or enlarged.
        """
        if autoexposure is not None: cls.autoexposure = autoexposure
        # if cls.autoexposure == 1:
        tempFlag = cls.lmk.iSetAutomatic(int(1*cls.autoexposure))
        if tempFlag == 0:
            if cls.verbosity > 1: print('iSetAutomatic: iSetAutomatic {:s}\n'.format(cls.boolStr[cls.autoexposure]))
        else:
            if cls.verbosity > 0: print('Error iSetAutomatic: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
    
    @classmethod
    def get_autoexposure(cls):
        """ 
        Get Automatic-Flag for all exposure times.
        """ 
        tempFlag, autoexposure = cls.lmk.iGetAutomatic(0)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetAutomatic: AutoExposure {:s}\n'.format(cls.boolStr[autoexposure]))
            
        else:
            if cls.verbosity > 0: print('Error iGetAutomatic: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return autoexposure
    
    @classmethod
    def set_focusfactor(cls, focusfactor = None):
        """ Set focus factor of lens """
        # List of available focus factors.
        # [int32, SafeArray Pointer(string), int32]
        #   iGetFocusFactorList	(handle, SafeArray Pointer(string), int32)
        #
        #   listFocusFactors	List of factors
        #   selectedIdxFocus	Index of currently used factor, indexes begin with "0"
        _, listFocusFactors, selectedIdxFocus = cls.lmk.iGetFocusFactorList([''],0)
        
        if focusfactor is not None: cls.focusfactor = focusfactor
        if cls.focusfactor is not None:
            # Set a focus factor
            tempFlag = cls.lmk.iSetFocusFactor(cls.focusfactor)
            if tempFlag == 0:
                _, tempFocusFactorList, tempCurrIdx = cls.lmk.iGetFocusFactorList([''],0)
                if cls.verbosity > 1: print('iSetFocusFactor: \t{:s} (Idx={:.0f})\n'.format(tempFocusFactorList[tempCurrIdx],tempCurrIdx))
            else:
                if cls.verbosity > 0: print('Error iSetFocusFactor: \n')
                cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
                if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))

    @classmethod
    def get_focusfactor(cls):
        """ Get focus factor of lens """
        # Get a focus factor
        _, FocusFactorList, CurrIdx = cls.lmk.iGetFocusFactorList([''],0)
        if cls.verbosity > 1: print('iGetFocusFactor: \t{:s} (Idx={:.0f})\n'.format(FocusFactorList[CurrIdx],CurrIdx))
        return FocusFactorList[CurrIdx], CurrIdx

    @classmethod 
    def set_max_exposure_time(cls, maxtime = None):
        """ 
        Set the maximum possible exposure time.
            int LMKAxServer::iSetMaxCameraTime	(	double 	_dMaxCameraTime	)
         
          The maximum values is of course restricted by camera properties.
          But you can use an even smaller time to avoid to long meausrement times.
         
         Parameters
           _dMaxCameraTime     Wished value
            maxCameraTime
        """
        if maxtime is not None: cls.maxtime = maxtime
        if cls.maxtime is not None:
            tempFlag = cls.lmk.iSetMaxCameraTime(cls.maxtime)
            if tempFlag == 0:
                tempFlag, maxIntT = cls.lmk.iGetMaxCameraTime(0)
                if cls.verbosity > 1: print('iSetMaxCameraTime: \t{:.0f}[s]\n'.format(maxIntT))
            else:
                if cls.verbosity > 0: print('Error iSetMaxCameraTime: \n');
                cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
                if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))

    @classmethod 
    def get_max_exposure_time(cls):
        """ 
        Get the maximum possible exposure time.
        """
        tempFlag, maxIntT = cls.lmk.iGetMaxCameraTime(0)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetMaxCameraTime: \t{:.0f}[s]\n'.format(maxIntT))
        else:
            if cls.verbosity > 0: print('Error iGetMaxCameraTime: \n');
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return maxIntT

    @classmethod 
    def set_mod_frequency(cls, modfrequency = None):
        """ 
         Set the frequency of modulated light.
            int LMKAxServer::iSetModulationFrequency	(	double 	_dModFrequency	)
        
          If the light source is driven by alternating current,
          there are some restriction for the exposure times.
          Please inform the program about the modulation frequency.
        
           Parameters
             _dModFrequency	Frequency of light source. 0 if no modulation is to be concerend
        """
        if modfrequency is not None: cls.modfrequency = modfrequency
        if (cls.modfrequency is not None) and not (np.isnan(cls.modfrequency).any()):
            tempFlag = cls.lmk.iSetModulationFrequency(cls.modfrequency)
            if tempFlag == 0:
                tempFlag, modFrequencyMeas = cls.lmk.iGetModulationFrequency(0)
                if cls.verbosity > 1: print('iGetModulationFrequency: \t{:.0f}[Hz]\n'.format(modFrequencyMeas))
            else:
                if cls.verbosity > 0: print('Error iSetModulationFrequency: \n')
                cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
                if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
                
    @classmethod 
    def get_mod_frequency(cls):
        """ 
        Get the frequency setting of modulated light.
        """
        tempFlag, modFrequencyMeas = cls.lmk.iGetModulationFrequency(0)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetModulationFrequency: \t{:.0f}[Hz]\n'.format(modFrequencyMeas))
        else:
            if cls.verbosity > 0: print('Error iGetModulationFrequency: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return modFrequencyMeas
                
    @classmethod
    def init(cls):
        """    init lmk ActiveX """
         
 
        # open a connection, if not already open:
        if cls.lmk is not None: 
            return cls.lmk # already open connection
        else:
            cls.open_lmk_labsoft_connection(objectiveCalibrationPath = cls.objectiveCalibrationPath)
        
        # Set new camera calibration data & lens
        if (cls.camera is None) | (cls.lens is None):
            # use no camera
            tempFlag = cls.lmk.iSetNewCamera2('','');
            return None
        else:
            tempFlag = cls.lmk.iSetNewCamera2(cls.camera['name'],cls.lens['name']);

        if tempFlag == 0:
            if cls.verbosity > 1: print('iSetNewCamera2 ok!\n')
        else:
            if cls.verbosity > 0: print('Error iSetNewCamera2: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        
        # Returns path to current camera lens combination.
        _, cls.cameraLensPath = cls.lmk.iGetCameraLensPath('')
        if cls.verbosity > 1: print('iGetCameraLensPath: {:s}\n'.format(cls.cameraLensPath))
        
        # Set max. number of filter wheels and their names:
        cls.filter_wheel_max, cls.filter_wheel_names = cls.get_filter_wheel_info()
        
        # Set parameters capturing method
        # Set auto scan
        cls.set_autoscan(autoscan = cls.autoscan)
        
        # Set Automatic-Flag for all exposure times. 
        cls.set_autoexposure(autoexposure = cls.autoexposure)
        
        # Set lens focus factor.
        cls.set_focusfactor(focusfactor = cls.focusfactor)        
        
        # Set the maximum possible exposure time.
        cls.set_max_exposure_time(maxtime = cls.maxtime)        
        
        # Set the frequency of modulated light.
        cls.set_mod_frequency(modfrequency = cls.modfrequency) 
        
        # Set units to cd/m² :
        cls.set_converting_units()
       
    @classmethod
    def set_converting_units(cls, units_name = "L", units = "cd/m²", units_factor = 1):
        """ Set the converting units (units_name, units, units_factor)"""
        if units_name is not None: cls.units_name = units_name 
        if units is not None: cls.units = units 
        if units_factor is not None: cls.units_factor = units_factor
        tempFlag = cls.lmk.iSetConvertingUnits(units_name, units, units_factor)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iSetConvertingUnits: \tunits_name: {}; units: {}; units_factor: {}\n'.format(cls.units_name, cls.units, cls.units_factor))
        else:
            if cls.verbosity > 0: print('Error iSetConvertingUnits: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))

    @classmethod
    def get_converting_units(cls):
        """ Get the converting units (units_name, units, units_factor)"""
        tempFlag, units_name, units, units_factor = cls.lmk.iGetConvertingUnits(cls.units_name, cls.units, cls.units_factor)
        if tempFlag == 0:
            if cls.verbosity > 1: print('iGetConvertingUnits: \tunits_name: {}; units: {}; units_factor: {}\n'.format(units_name, units, units_factor))
        else:
            if cls.verbosity > 0: print('Error iSetConvertingUnits: \n')
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            if cls.verbosity > 0: print('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
        return units_name, units, units_factor
     
       
    @classmethod
    def set_verbosity(cls, value):
        if value in cls.verbosity_levels:
            cls.verbosity = value
    
    @classmethod
    def checkForError(cls):
        global ST
        ST = traceback.extract_stack(limit = 1) # matlab: ST = dbstack('-completenames',1)
        if (cls.errorFlag != 0):
            cls.answer, errorInfo = cls.lmk.iGetErrorInformation('')
            raise Exception('\tError: {} \t info: {:s}\n'.format(cls.answer,errorInfo))
   
#==============================================================================
# Working with PCF images
#==============================================================================   
def read_pcf(fname):
    """ Read a TechnoTeam PCF image. (!!! output = float32 CIE-RGB !!!)"""
    encoding = 'latin_1'
    with open(fname,'rb') as f:
        
        i = 0
        for line in f: 
            line = line.rstrip(b'\n').decode(encoding)
            if line == '\r': 
                break
            if i == 0: 
                img_type = (line[line.find('=')+1:]) 
            elif i == 1:
                rows = int(line[line.find('=')+1:])
            elif i == 2:
                cols = int(line[line.find('=')+1:])
            i+=1
            
        totalNumberOfBytes = rows*cols*12 # Twelve bytes per pixel. 
        # 3 "Float" values each for the three colors. The chromaticity 
        # values in the file are in the order of blue - green - read.
        
        pixeldata = f.read()
    
        f.close()
    
    img_raw = np.frombuffer(pixeldata, dtype=np.float32, offset = 1).reshape((rows,cols,3))
    return img_raw[...,::-1].copy() # BGR to RGB


def ciergb_to_xyz(rgb):
    """ Convert CIE-RGB to XYZ """
    # matrix from TechnoTeam's LMK4.pdf (p311, February 27, 2017 version)
    M = np.array([[2.7689,1.7518,1.1302],
                  [1.0000,4.5907,0.0601],
                  [0.0000,0.0565,5.5943]])
    
    return (rgb @ M.T)

def xyz_to_ciergb(xyz):
    """ Convert XYZ to CIE-RGB """
    # matrix from TechnoTeam's LMK4.pdf (p311, February 27, 2017 version)
    M = np.array([[0.4185,-0.1587,-0.0828],
                  [-0.0912,0.2524,0.0157],
                  [0.0009,-0.0026,0.1786]])  
    return (xyz @ M.T)


def pcf_to_xyz(pcf_image):
    """ Convert a TechnoTeam PCF image to XYZ """
    return ciergb_to_xyz(pcf_image)

def img_to_01(img):
    """ Normalize image to 0 - 1 range """
    img = (img - img.min())/(img.max() - img.min())
    return img

def plot_pcf(img, to_01_range = True, ax = None):
    """ Plot a TechnoTeam PCF image. """
    if isinstance(img, str):
        img = read_pcf(img)
    if to_01_range:
        img = img_to_01(img) # min -> 0, max -> 1 for imshow (avoid clipping)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.imshow(img)
    norm_str = '(normalized to 0-1 range)' if to_01_range  else '(not normalized)'
    ax.set_title('PCF image {:s}'.format(norm_str))
    return img
        
   
    
if __name__ == '__main__':
    #========================================================
    # TEST CODE
    #========================================================
    folder_name = r'./'
    
    print('\n--0-- Test lmkActiveX init')      
    # Create an lmkX object:
    lmkX = lmkActiveX('tts20035', 'xvr', focusfactor = None,
                      autoscan = True, autoexposure = True,
                      modfrequency = 60, maxtime = 10,
                      verbosity = None)
    
    print('\n--1-- Test measure multiple XYZ images')
    # Make multiple measurement and save:
    for i in range(2):
        lmkX.measureCaptureXYZmap(folder_name, 'testXYZ_{:1.0f}'.format(i))
    
    print('\n--2-- Test measure Y image')
    lmkX.measureCaptureYmap(folder_name, 'testY_{:1.0f}'.format(0))

    print('\n--3-- Test measure of ColorMultipic')
    # Capture ColorMultipic image
    lmkX.measureColorMultipic(1,1)
    
    print('\n--4a-- Test creation of elliptical region')
    # Create an elliptical region
    lmkX.createEllips([3018//2, 4103//2],300,100,'ellipse_test') 
    
    print('\n--4b-- Test creation of rectangular region and selecting it')
    # Create an rectangle region and select it:
    lmkX.createRectangle([3018//2 - 200, 4103//2 - 200],[3018//2 + 200, 4103//2 + 200],'rectangle_test') 
    lmkX.selectRegionByName('rectangle_test',1)
    
    print('\n--5-- Test get_color_autoscan_times')
    # Get color_autoscan_exposuretimes:
    color_autoscan_times = lmkX.get_color_autoscan_times() 
    print(color_autoscan_times)
    
    print('\n--6a-- Test measurement times for autoscan off')
    tb = time.time()
    lmkX.measureCaptureXYZmap(folder_name, 'testXYZ2_{:1.0f}'.format(0), autoscan = False)
    te = time.time()
    print('Time autoscan == False', te - tb)
    
    print('\n--6b-- Test measurement times for autoscan off')
    tb = time.time()
    lmkX.measureCaptureXYZmap(folder_name, 'testXYZ2_{:1.0f}'.format(1), autoscan = True)
    te = time.time()
    print('Time autoscan == True', te - tb)

    print('\n--7-- TESTS END')
    del lmkX 
    
    # if False: 
    #     kill_lmk4_process(1) 
    

