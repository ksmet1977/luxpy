# Python toolbox for lighting and color science
![alt text][logo]

[logo]: https://github.com/ksmet1977/luxpy/blob/master/docs/images/LUXPY__logo.jpg

* Author: K. A.G. Smet (ksmet1977 at gmail.com)
* Version: 1.3.07
* Date: June 1, 2018
* License: [GPLv3](https://github.com/ksmet1977/luxpy/blob/master/LICENSE.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1298963.svg)](https://doi.org/10.5281/zenodo.1298963)

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

For more details on installation, structure, functionality, etc., see: 
 1. the github pages on: [**ksmet1977.github.io/luxpy/**](http://ksmet1977.github.io/luxpy/) 
 2. the [**LuxPy_Documentation**](https://github.com/ksmet1977/luxpy/blob/master/LuxPy_Documentation.pdf) pdf
 2. or, the **\__doc__string** of each function. 
        
    To get help on, for example the **spd_to_xyz()** function, type:


            import luxpy as lx
            ?lx.spd_to_xyz
    
    To get a list of functions/modules, type:


            dir(lx)
    


## Python tutorials
### Some basic tutorials can be found at:
 * [cs231n.github.io/python-numpy-tutorial/](http://cs231n.github.io/python-numpy-tutorial/) 
 * [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/) 
### A list of basic and more advanced is given at:
 * [wiki.python.org/moin/BeginnersGuide/Programmers](https://wiki.python.org/moin/BeginnersGuide/Programmers)
### Matlab versus Python:
 * [scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html)
### Udemy.com:
 * [udemy.com](http://www.udemy.com)Udemy.com offers some great courses. Although some of these are payed, they often come at huge discounted prices. 
### Youtube.com:
 * [www.youtube.com](https://www.youtube.com/results?search_query=python+numpy+tutorial) also has lots of free online tutorials.