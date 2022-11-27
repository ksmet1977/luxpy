# -*- coding: utf-8 -*-
"""
Test stereoscopic viewer toolbox 
--------------------------------


Created on Sun Nov 27 13:57:50 2022

@author: u0032318
"""
import os
import time
import numpy as np
from luxpy.toolboxes import stereoscopicviewer as sv

if __name__ == '__main__':       
    
    ###############################################################################
    # Example code on how to:
    #    - update the action function in the pipeFcns
    #    - prepare / generate stimulus textures
    #    - use the Viewer with(out) VR HMD
    ###############################################################################
    
    #===========================================================================
    # Tests of different options:
    #===========================================================================
    script_path = os.path.dirname(os.path.realpath(__file__)) # LMK save requires absolute path !!!
    
    #=== Action definitions ===================================================
    use_LMK = False
    
    if use_LMK:
        ## Setup instance of LMK for measurements:
        from luxpy.toolboxes.technoteam_lmk import lmkActiveX, kill_lmk4_process
        
        modfrequency = 72 # check what freq. the headset is running on: Quest2: 72 Hz, Rift CV2: 90 Hz
        lmkX = lmkActiveX('tts20035', 'xvr', focusfactor = None,
                          autoscan = False, autoexposure = True,
                          modfrequency = modfrequency, maxtime = 10,
                          verbosity = None)
                           
        def a_action(self, frameNumber, out, 
                     lmkX = None, folder_name = None, file_name_base = None):
            if not os.path.exists(folder_name): 
                os.makedirs(folder_name, exist_ok = True)
            if lmkX is not None: 
                file = os.path.join(folder_name, '{:s}_{:1.0f}'.format(file_name_base, self.texIdx))
                lmkX.measureCaptureXYZmap(folder_name, '{:s}_{:1.0f}'.format(file_name_base, self.texIdx))
                
            return file + '.pcf'
    
        def a_check(self, frameNumber, out):
            if isinstance(out[1][1],(int, float)):
                return False
            else:
                return bool(os.path.exists(out[1][1]))
        
        actionWrapperDict = {'action' : (a_action, {'lmkX':lmkX, 'folder_name' : script_path+'./temp_color_stimulus_folder/','file_name_base':'lmk-23-11-2022'}),
                             'check'  : (a_check,  {})}
    else:
        #--------------------------------------------------------------------------
        def a_action(self, frameNumber, out, **kwargs):
            t_start = time.time() # e.g. get time or start measurement
            return t_start # stored in out[3] so it is available for a later check
    
        def a_check(self, frameNumber, out, delay = 3):
            t_now = time.time() 
            dt = t_now - out[1][1] 
            return dt >= delay
        
        actionWrapperDict =  {'action' : (a_action, {}),
                              'check'  : (a_check,  {'delay' : 0.1})}
    
    #==== Prepare stimuli =====================================================
        
    stimulus_list = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]])*255
    texFiles, _ = sv.generate_stimulus_tex_list(stimulus_list, rgba_save_folder = 'rgba_texs')
    
    # # Generate stimulus textures:
    # os.makedirs('./temp_color_stimulus_folder/', exist_ok = True)
    # texFiles = []
    # for i,color in enumerate(stimulus_list): 
    #     texFiles.append('./temp_color_stimulus_folder/tex_{:1.0f}_{:1.0f}_{:1.0f}.jpg'.format(*color[:3]))
    #     sv.makeColorTex(color, texHeight = 100, texWidth = 100, save = texFiles[-1])
    # texFiles = [(texFile,texFile) for texFile in texFiles]
    
    # or load files from folder or list.iml file:
    texFiles, _ = sv.generate_stimulus_tex_list('./luxpy/toolboxes/stereoscopicviewer/harfang/spheremaps/list.iml')
    
    #==== Prepare hmdviewer ===================================================
    
    hmdviewer = sv.HmdStereoViewer(screen_uSelfMapTexture = texFiles, 
                                screen_geometry = 'sphere',vrFlag = False)
    
    # pipeFcns = copy.deepcopy(hmdviewer.pipeFcnsDef) 
    # pipeFcns[-1] = (pipeFcns[-1][0], actionWrapperDict) # replace default action with custom action
    
    # === Display =============================================================
    
    # hmdviewer.display()
    
    #==== Run tests of display + action ========================================
    # hmdviewer.run()
    # hmdviewer.run(only_once = True)
    t_begin = time.time()
    
    hmdviewer.run(pipeFcnsUpdate = [None,None, (None, actionWrapperDict)],  only_once = False,
                  u_delay = 1, a_delay = 1)
       
    t_end = time.time()
    print('Total run time: ', t_end - t_begin)
