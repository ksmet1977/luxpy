Installation
============

Install luxpy
-------------

	1. Install miniconda 
		* download the installer from: https://conda.io/miniconda.html 
		  or https://repo.continuum.io/miniconda/)
		* e.g. https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
		* Make sure 'conda.exe' can be found on the windows system path, if necessary do a manual add.
		
	2. Create a virtual environment with full anaconda distribution by typing the following at the commandline: 
		
		``>> conda create --name py36 python=3.6 anaconda``
		
	3. Activate the virtual environment:
		
		``>> activate py36``
		
	4. Install pip to virtual environment (just to ensure any packages to be 
		installed with pip to this virt. env. will be installed here and not globally):
		
		``>> conda install -n py36 pip``
		
	5. Install luxpy package from pypi:
		
		``>> pip install luxpy``
		

:Note:
	If any errors show up, try and do a manual install of the dependencies: 
	scipy, numpy, pandas, matplotlib and setuptools,
	either using e.g. 
	``>> conda install scipy`` 
	or 
	``>> pip install scipy``,
	and try and reinstall luxpy using pip. 
	

Use of LuxPy package in Spyder IDE
---------------------------------- 
		
	6. Install spyder in py36 environment:
		
		``>> conda install -n py36 spyder``
		
	7. Run spyder 
		
		``>> spyder``
	
	8. To import the luxpy package, on Spyder's commandline for the IPython kernel (or in script) type:
		
		``import luxpy as lx``
		
		
Use of LuxPy package in Jupyter notebook
----------------------------------------
	
	6. Start jupyter notebook:
		
		``>> jupyter notebook``
			
			
	7. Open an existing or new notebook: 
		e.g. open "luxpy_basic_usage.ipynb" for an overview of how to use the LuxPy package.
		
			
	8. To import LuxPy package type:
	
		``import luxpy as lx``
