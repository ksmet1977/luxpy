#---------------------------------------------------------------------
The assets in the 'assets' and 'assets_compiled' folders 
have been trimmed to reduce the size of the toolbox:
 * core/noise has been deleted
 * only pbr & forwardpipeline related shader files have been kept in assets/core/shader
 * of all the variants generated in assets_compiled/core/shader 
	only the vertex and fragment shaders for 'pbr_var-8_pipe-forward-cfg-1'
	have been kept, as those seem to be the ones needed for the viewer to work

#---------------------------------------------------------------------
Should you run into problems, try the following:
 * step 1: delete the assets and assets_compiled folders
 * step 2: download the noise, pbr and shader folders from: 
	   https://github.com/harfang3d/harfang-core-package
 * step 3: put these 3 folders in a folder ./assets/core/
 * step 4: download assetC-*.zip for your system from: 
	   https://dev.harfang3d.com/releases/
 * step 5: unzip the assetC-*.zip file and open the folder
 * step 6: drag and drop the assets folder created in step 3 
	   onto the assetc.exe file for compilation of the assets.
You now have a full assets_compiled folder in the same folder, 
i.e.: '/luxpy/toolboxes/stereoscopicviewer/harfang/' 
as your assets folder, and harfang should now work.
#---------------------------------------------------------------------




