# Python toolbox for lighting and color science
![alt text][logo]

[logo]: https://github.com/ksmet1977/luxpy/blob/master/images/LUXPY_logo.jpg

* Author: K. A.G. Smet (ksmet1977 at gmail.com)
* Version: 1.2.06
* Date: April 12, 2018
* License: [GPLv3](https://github.com/ksmet1977/luxpy/blob/master/LICENSE.md)

**Luxpy** is an open source package under a GPLv3 license that supports several common
**lighting**, **colorimetric**, **color appearance** and other **color science**
 related calculations and models, such as:
* spectral data interpolation (conform CIE15-2004) and normalization
* calculation of daylight phase, blackbody radiator and other reference illuminant spectra
* calculation of tristimulus values
* correlated color temperature and Duv
* color space transformations
* chromatic adaptation transforms 
* color appearance models 
* color rendition indices 
* ...

-------------------------------------------------------------------------------
## How to use (basics)?
An overview of the basic usage is given in the [**luxpy basic usage.ipynb**](https://github.com/ksmet1977/luxpy/blob/master/luxpy_basic_usage.ipynb) jupyter notebook (for instruction on how to install the luxpy package and open up a jupyter notebook, see below).

-------------------------------------------------------------------------------
## Installation
1. Install miniconda (Download installer from: https://conda.io/miniconda.html or https://repo.continuum.io/miniconda/, e.g. https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe. Make sure **conda.exe** can be found on the windows system path (if necessary, do a manual add)
		
2. Create a virtual environment with a full anaconda distribution:

		
        $ conda create --name py36 python=3.6 anaconda
    
		
3. Activate virtual environment:


        $ activate py36
    
		
4. Install pip to the **py36** virtual (conda) environment (just to ensure any packages to be installed with pip to this virtual (conda) environment will be installed here and not globally):


        (py36) $ conda install -n py36 pip
    
		
5. Install **luxpy** from the [Python Package Index](https://pypi.python.org/pypi/luxpy/):


        (py36) $ pip install luxpy
    
		
If any errors show up, try and do a manual install of the dependencies: scipy, numpy, pandas, matplotlib and setuptools,
either using e.g. "conda install scipy" or "pip install scipy", and try and reinstall luxpy using pip) 

## Use / import of luxpy package in "spyder" / "jupyter notebook":

### Spyder: a matlab-like IDE:
		
1. Install spyder in the **py36** environment:


        $ conda install -n py36 spyder 
			
			
2. Run spyder in the **py36** activate environment


        (py36) $ spyder
        
		
3. Import the luxpy package by typing in the commandline or in a script file:


        import luxpy as lx 
        
		
### Jupyter notebook:
	
1. Start jupyter notebook after activating the **py36** environment:


    (py36) $ jupyter notebook
        
			
2. Open an existing or new notebook (e.g. open "luxpy basic usage.ipynb" for an overview of how to use the luxpy package, or see README.md on www.github.com/ksmet1977/luxpy)
			
3. To import luxpy package type the following a a jupyter notebook cell and run:
			

        import luxpy as lx


-------------------------------------------------------------------------------
## Module overview
    
    0.1.  helpers/ helpers.py (imported directly into luxpy namespace, details see end of this file)
    0.2.  math/ math.py (imported as math into the luxpy namespace, details see end of this file)
    
    1a.  spectral/ cmf.py
    1b.  spectral/ spectral.py
    1c.  spectral/ spectral_databases
    2a.  ctf/ colortransforms.py (imported directly into luxpy namespace)
    3a.  cct/ cct.py (imported directly into luxpy namespace)
    4a.  cat/ chromaticadaptation.py (imported in luxpy namespace as .cat)
    5a.  cam/ colorappearancemodels.py (imported in luxpy namespace as .cam)
    2b.  ctf/ colortf.py (imported directly into luxpy namespace)
    6a.  cri/ colorrendition_indices.py (imported in luxpy namespace as .cri)
    7a. graphics/ plotters.py (imported directly into luxpy namespace)
    8a. classes/ SPD (imported directly into luxpy namespace)
    8b. classes/ CDATA, XYZ, LAB (imported directly into luxpy namespace)
    
For more details see [luxpy_module_overview.md](https://github.com/ksmet1977/luxpy/blob/master/luxpy_module_overview.md) 
or \__doc__string of each function. To get help on for example the **spd_to_xyz()** function type:


    import luxpy as lx
    ?lx.spd_to_xyz
    
To get a list of functions/modules, type:


    dir(lx)

