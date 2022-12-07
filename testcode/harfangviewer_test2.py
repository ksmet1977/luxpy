# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 08:34:44 2022

@author: u0032318
"""

from luxpy.toolboxes.stereoscopicviewer import harfangviewer as hv
import matplotlib.pyplot as plt 
import imageio
import numpy as np

file = './grid_image.jpg'
# im = imageio.imread(file)
# im = np.dstack((im,im,im))
# imageio.imsave(file,im)
# plt.imshow(im)

vrFlag = False

hmdviewer = hv.HmdStereoViewer(screen_uSelfColor = [0,0,1,1], screen_geometry = 'sphere',vrFlag = vrFlag) 

# hmdviewer.set_texture(screen_uSelfMapTexture = file)

hmdviewer.display()
