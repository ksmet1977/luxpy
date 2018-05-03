LuxPy package structure
=======================
|	/utils
|		/ helpers
|         helpers.py
|		/.math
|			math.py
|			optimizers.py
|	
|	/spectrum
|		cmf.py
|		spectral.py
|		spectral_databases.py
|	
|	/color
|		colortransformations.py
|		cct.py
|		/.cat
|			chromaticadaptation.py	
|		/.cam
|			colorappearancemodels.py
|				cam_02_X.py
|				cam15u.py
|				sww16.py
|		colortf.py
|		/.deltaE
|			colordifferences.py
|		/.cri
|			colorrendition.py
|			/utils
|				DE_scalers.py
|				helpers.py
|				init_cri_defaults_database.py
|				graphics.py
|			/indices
|				indices.py
|					cie_wrappers.py
|					ies_wrappers.py
|					cri2012.py
|					mcri.py
|					cqs.py
|			/ies_tm30
|				ies_tm30_metrics.py
|				ies_tm30_graphics.py
|			/.VFPX
|				VF_PX_models.py (imported in .cri as .VFPX)
|					vectorshiftmodel.py
|					pixelshiftmodel.py
|		/utils
|			plotters.py
|		
|		
|	/classes
|		SPD.py
|		CDATA.py
|		
|	/data
|		/cmfs
|		/spds
|		/rfls
|		/cctluts
|
|		
|	/toolboxes
|		
|		/.photbiochem
|			cie_tn003_2015.py
|			/data
|			
|		/.indvcmf
|			individual_observer_cmf_model.py
|			/data
|		
|		/.spdbuild
|			spd_builder.py
|			
|		/.hypspcsim
|			hyperspectral_img_simulator.py

Global constants
================
The package and sub-packages use several global constants that set the default state/behaviour
of the calculations. LuxPy 'global constants' start with '_' and are in an 
all _CAPITAL format. 

E.g.:
 * _PKG_PATH (absolute path to luxpy package)
 * _SEP (operating system operator)
 * _EPS = 7./3 - 4./3 -1 (machine epsilon)
 * _CIEOBS = '1931_2' (default CIE observer color matching function)
 * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
DO NOT CHANGE THESE CONSTANTS!