# Python toolbox for lighting and color science
![alt text][logo]

[logo]: https://github.com/ksmet1977/luxpy/blob/master/docs/images/LUXPY__logo.jpg

* Author: Kevin A.G. Smet (ksmet1977 at gmail.com)
* Version: 1.5.3
* Date: Sep 12, 2020
* License: [GPLv3](https://github.com/ksmet1977/luxpy/blob/master/LICENSE.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1298963.svg)](https://doi.org/10.5281/zenodo.1298963)

### Cite LuxPy:
If you use the package, please cite the following **tutorial paper** published in **LEUKOS**:
[**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23. DOI: 10.1080/15502724.2018.1518717**](https://www.tandfonline.com/doi/full/10.1080/15502724.2018.1518717)

-------------------------------------------------------------------------------
## What is LuxPy?
**Luxpy** is an open source package under a GPLv3 license that supports several common
**lighting**, **colorimetric**, **color appearance** and other **color science**
 related calculations and models, such as:
* spectral data interpolation (conform CIE15-2018) and normalization
* calculation of daylight phase, blackbody radiator and other reference illuminant spectra
* calculation of tristimulus values
* correlated color temperature and Duv
* color space transformations
* chromatic adaptation transforms 
* color appearance models 
* color rendition indices 
* calculation of photobiological quantities
* multi-component spectrum creation and optimization
* hyper-spectral image simulation and rendering
* MacAdam ellipses
* color differences (cam02ucs, DE2000, ...)
* modelling of individual observer color matching functions (Asano, 2016)
* calculation of CIEOP06 (cfr. CIE TC1-97) color matching functions and cone-fundamentals
* display characterization
* ...

As of May 2019, LuxPy now also has a toolbox **spectro** for **spectral measurements** with **JETI** and **OceanOptics** spectrometers:
* spectro.jeti: easy installation (dll's are part of sub-package).
* spectro.oceanoptics (under development): more tricky installation (requires manual install of **seabreeze** SDK and **python-seabreeze**, **pyusb**, ...; see [here](https://github.com/ksmet1977/luxpy/blob/master/luxpy/toolboxes/spectro/oceanoptics/oceanoptics.md) or subpackage help for more info)

** NEW **: ANSI/IES-TM30-2018 graphical output (**Color Rendition Report**, **Color Vector Graphic**, ...)
-------------------------------------------------------------------------------
## How to use LuxPy (basics)?
Luxpy can be easily installed from pypi `pip install luxpy` or anaconda `conda install -c ksmet1977 luxpy`.

An overview of the *basic usage* is given in the [**luxpy basic usage.ipynb**](https://github.com/ksmet1977/luxpy/blob/master/luxpy_basic_usage.ipynb) jupyter notebook,
 as well as the tutorial paper published in LEUKOS: [**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23. DOI: 10.1080/15502724.2018.1518717**](https://www.tandfonline.com/doi/full/10.1080/15502724.2018.1518717)

 * !!! **If the jupyter notebook fails to download or gives an error** (*Github seems to experience some type of problem sometimes with its jupyter notebook backend, see [issue](https://github.com/iurisegtovich/PyTherm-applied-thermodynamics/issues/11)*), try opening up the file using the [nbviewer.jupyter.org](https://nbviewer.jupyter.org) online viewer, (or just click this direct link: [nbviewer.jupyter.org/github/ksmet1977/luxpy/blob/master/luxpy_basic_usage.ipynb](https://nbviewer.jupyter.org/github/ksmet1977/luxpy/blob/master/luxpy_basic_usage.ipynb)) and then download it from there (use download notebook button at the top right of the page). 
 
For more details on structure, functionality, etc., see: 
 1. the github pages on: [**ksmet1977.github.io/luxpy/**](http://ksmet1977.github.io/luxpy/) 
 2. the [**LuxPy_Documentation**](https://github.com/ksmet1977/luxpy/blob/master/LuxPy_Documentation.pdf) pdf
 3. or, the **\__doc__string** of each function. 
        
    To get help on, for example the **spd_to_xyz()** function, type:


            import luxpy as lx
            ?lx.spd_to_xyz
    
    To get a list of functions/modules, type:


            dir(lx)
    

-------------------------------------------------------------------------------
## Python tutorials
#### Some basic tutorials can be found at:
 * [cs231n.github.io/python-numpy-tutorial/](http://cs231n.github.io/python-numpy-tutorial/) 
 * [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/) 
#### A list of basic and more advanced is given at:
 * [wiki.python.org/moin/BeginnersGuide/Programmers](https://wiki.python.org/moin/BeginnersGuide/Programmers)
#### Matlab versus Python:
 * [scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html)
#### Udemy.com:
 * [Udemy.com](https://www.udemy.com/courses/search/?ref=home&src=ukw&q=python%20numpy) offers some great courses. Although some of these are payed, they often come at huge discounted prices. 
#### Youtube.com:
 * [www.youtube.com](https://www.youtube.com/results?search_query=python+numpy+tutorial) also has lots of free online tutorials.