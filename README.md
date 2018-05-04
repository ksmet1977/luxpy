# Python toolbox for lighting and color science
![alt text][logo]

[logo]: https://github.com/ksmet1977/luxpy/blob/master/images/LUXPY__logo.jpg

* Author: K. A.G. Smet (ksmet1977 at gmail.com)
* Version: 1.3.00
* Date: May 3, 2018
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
* calculation of photobiological quantities
* Multi-component spectrum creation and optimization
* Hyper-spectral image simulation and rendering
* ...

-------------------------------------------------------------------------------
## How to use (basics)?
An overview of the basic usage is given in the [**luxpy basic usage.ipynb**](https://github.com/ksmet1977/luxpy/blob/master/luxpy_basic_usage.ipynb) jupyter notebook 

For more details on installation, structure, functionality, etc.: 
 1. see [**luxpy.readthedocs.io**](http://luxpy.readthedocs.io/en/latest/) 
 2. or see the \__doc__string of each function. 
        
    To get help on, for example the **spd_to_xyz()** function, type:


            import luxpy as lx
            ?lx.spd_to_xyz
    
    To get a list of functions/modules, type:


            dir(lx)
    


