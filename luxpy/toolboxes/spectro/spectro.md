# luxpy.spectro: package for spectral measurements

## Supported devices:

 * **JETI**: specbos 1211, ...
 * **OceanOptics**: QEPro, QE65Pro, QE65000, USB2000, USB650,...
 
## Functions:

 * get_spd(): wrapper function to measure a spectral power distribution using a spectrometer of one of the supported manufacturers. 
 
## Notes:

 1. For info on the input arguments of **get_spd()**, see help for each identically named function in each of the subpackages. 
 2. The use of [**jeti**](https://github.com/ksmet1977/luxpy/blob/master/luxpy/toolboxes/spectro/jeti/jeti.md). 
  spectrometers requires access to some dll files (delivered with this package).
 3. The use of [**oceanoptics**](https://github.com/ksmet1977/luxpy/blob/master/luxpy/toolboxes/spectro/oceanoptics/oceanoptics.md). 
  spectrometers requires the **manual installation** of **python-seabreeze**, 
 as well as some other 'manual' settings. See installation help for oceanoptics sub-package. 
 