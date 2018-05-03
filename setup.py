# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
setup(
  name = 'luxpy',
  packages = find_packages(), 
  version = '1.3.00',
  license = 'GPLv3',
  description = 'Python package for lighting and color science',
  author = 'Kevin A.G. Smet',
  author_email = 'ksmet1977@gmail.com',
  url = 'https://github.com/ksmet1977/luxpy',
  download_url = 'https://github.com/ksmet1977/luxpy/archive/1.3.00.tar.gz',
  keywords = ['color', 'color appearance', 'colorimetry','photometry','CIE','color perception','lighting','color rendering','IES'], 
  install_requires=[
          'numpy',
		  'scipy',
		  'matplotlib',
		  'pandas',
          'itertools',
          'opencv-python',
		  'colorsys',
		  'mpl_toolkits',
		  'warnings',
		  'collections',
		  'os'
      ],
  package_data={'luxpy': ['luxpy/data/cmfs/*.dat',
                          'luxpy/data/cmfs/*.txt',
                          'luxpy/data/cctluts/*.dat',
                          'luxpy/data/cctluts/*.txt', 
                          'luxpy/data/spds/*.dat',
                          'luxpy/data/spds/*.txt',
                          'luxpy/data/rfls/*.dat',
                          'luxpy/data/rfls/*.txt',
                          'luxpy/toolboxes/photbiochem/data/*.dat',
                          'luxpy/toolboxes/photbiochem/data/*.txt',
                          'luxpy/toolboxes/indvcmf/data/*.dat',
                          'luxpy/toolboxes/indvcmf/data/*.txt',
                          'luxpy/toolboxes/hypspcim/data/*.*'
                          ]},
  include_package_data = True,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3.6',
    ],  
  python_requires='>=3.6',
)
