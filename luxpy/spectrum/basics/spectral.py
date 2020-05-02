# -*- coding: utf-8 -*-

########################################################################
# <LUXPY: a Python package for lighting and color science.>
# Copyright (C) <2017>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################

"""
Module supporting basic spectral calculations.
==============================================

 :_WL3: Default wavelength specification in vector-3 format: 
        ndarray([start, end, spacing])

 :_BB: Dict with constants for blackbody radiator calculation 
       constant are (c1, c2, n, na, c, h, k). 

 :_S012_DAYLIGHTPHASE: ndarray with CIE S0,S1, S2 curves for daylight 
        phase calculation.

 :_INTERP_TYPES: Dict with interpolation types associated with various types of
                 spectral data according to CIE recommendation:  

 :_S_INTERP_TYPE: Interpolation type for light source spectral data

 :_R_INTERP_TYPE: Interpolation type for reflective/transmissive spectral data

 :_CRI_REF_TYPE: Dict with blackbody to daylight transition (mixing) ranges for
                 various types of reference illuminants used in color rendering
                 index calculations.

 :getwlr(): Get/construct a wavelength range from a (start, stop, spacing) 
            3-vector.

 :getwld(): Get wavelength spacing of ndarray with wavelengths.

 :spd_normalize(): Spectrum normalization (supports: area, max, lambda, 
                   radiometric, photometric and quantal energy units).

 :cie_interp(): Interpolate / extrapolate spectral data following standard 
                [`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_]

 :spd(): | All-in-one function that can:
         |  1. Read spectral data from data file or take input directly as 
            pandas.dataframe or ndarray.
         |  2. Convert spd-like data from ndarray to pandas.dataframe and back.
         |  3. Interpolate spectral data.
         |  4. Normalize spectral data.

 :xyzbar(): Get color matching functions.
        
 :vlbar(): Get Vlambda function.

 :spd_to_xyz(): Calculates xyz tristimulus values from spectral data. 
            
 :spd_to_ler():  Calculates Luminous efficacy of radiation (LER) 
                 from spectral data.

 :spd_to_power(): Calculate power of spectral data in radiometric, photometric
                  or quantal energy units.
         
 :blackbody(): Calculate blackbody radiator spectrum.
             
 :daylightlocus(): Calculates daylight chromaticity from cct. 

 :daylightphase(): Calculate daylight phase spectrum.
         
 :cri_ref(): Calculates a reference illuminant spectrum based on cct for color 
             rendering index calculations.
            (`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_, 
             `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9. <http://www.cie.co.at/index.php?i_ca_id=1027>`_,
             `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. New York, NY: The Illuminating Engineering Society of North America. <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
 
 :detect_peakwl(): Detect peak wavelengths and fwhm of peaks in spectrum spd.   

 :spd_to_indoor(): Convert spd to indoor variant by multiplying it with the CIE spectral transmission for glass. 

References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    
    2. `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017),
    ISBN 978-3-902842-61-9. 
    <http://www.cie.co.at/index.php?i_ca_id=1027>`_
    
    3. `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
    New York, NY: The Illuminating Engineering Society of North America. 
    <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

#--------------------------------------------------------------------------------------------------
from luxpy import  _CIEOBS, math
from luxpy.utils import np, pd, sp, plt, _PKG_PATH, _SEP, _EPS, np2d, getdata
from .cmf import _CMF
from .spectral_databases import _CIE_GLASS_ID
from scipy import signal
__all__ = ['_WL3','_BB','_S012_DAYLIGHTPHASE','_INTERP_TYPES','_S_INTERP_TYPE', '_R_INTERP_TYPE','_CRI_REF_TYPE',
           '_CRI_REF_TYPES', 'getwlr','getwld','spd_normalize','cie_interp','spd','xyzbar', 'vlbar', 
           'spd_to_xyz', 'spd_to_ler', 'spd_to_power',
           'blackbody','daylightlocus','daylightphase','cri_ref','detect_peakwl', 'spd_to_indoor']


#--------------------------------------------------------------------------------------------------
# set standard SPD wavelength interval interval and spacing
_WL3 = [360.0,830.0,1.0]

#--------------------------------------------------------------------------------------------------
# set some colorimetric constants related to illuminants
_BB = {'c1' : 3.74183e-16, 'c2' : 1.4388*0.01,'n': 1.000, 'na': 1.00028, 'c' : 299792458, 'h' : 6.626070040e-34, 'k' : 1.38064852e-23} # blackbody c1,c2 & n standard values

_S012_DAYLIGHTPHASE=np.array([[360.000,361.000,362.000,363.000,364.000,365.000,366.000,367.000,368.000,369.000,370.000,371.000,372.000,373.000,374.000,375.000,376.000,377.000,378.000,379.000,380.000,381.000,382.000,383.000,384.000,385.000,386.000,387.000,388.000,389.000,390.000,391.000,392.000,393.000,394.000,395.000,396.000,397.000,398.000,399.000,400.000,401.000,402.000,403.000,404.000,405.000,406.000,407.000,408.000,409.000,410.000,411.000,412.000,413.000,414.000,415.000,416.000,417.000,418.000,419.000,420.000,421.000,422.000,423.000,424.000,425.000,426.000,427.000,428.000,429.000,430.000,431.000,432.000,433.000,434.000,435.000,436.000,437.000,438.000,439.000,440.000,441.000,442.000,443.000,444.000,445.000,446.000,447.000,448.000,449.000,450.000,451.000,452.000,453.000,454.000,455.000,456.000,457.000,458.000,459.000,460.000,461.000,462.000,463.000,464.000,465.000,466.000,467.000,468.000,469.000,470.000,471.000,472.000,473.000,474.000,475.000,476.000,477.000,478.000,479.000,480.000,481.000,482.000,483.000,484.000,485.000,486.000,487.000,488.000,489.000,490.000,491.000,492.000,493.000,494.000,495.000,496.000,497.000,498.000,499.000,500.000,501.000,502.000,503.000,504.000,505.000,506.000,507.000,508.000,509.000,510.000,511.000,512.000,513.000,514.000,515.000,516.000,517.000,518.000,519.000,520.000,521.000,522.000,523.000,524.000,525.000,526.000,527.000,528.000,529.000,530.000,531.000,532.000,533.000,534.000,535.000,536.000,537.000,538.000,539.000,540.000,541.000,542.000,543.000,544.000,545.000,546.000,547.000,548.000,549.000,550.000,551.000,552.000,553.000,554.000,555.000,556.000,557.000,558.000,559.000,560.000,561.000,562.000,563.000,564.000,565.000,566.000,567.000,568.000,569.000,570.000,571.000,572.000,573.000,574.000,575.000,576.000,577.000,578.000,579.000,580.000,581.000,582.000,583.000,584.000,585.000,586.000,587.000,588.000,589.000,590.000,591.000,592.000,593.000,594.000,595.000,596.000,597.000,598.000,599.000,600.000,601.000,602.000,603.000,604.000,605.000,606.000,607.000,608.000,609.000,610.000,611.000,612.000,613.000,614.000,615.000,616.000,617.000,618.000,619.000,620.000,621.000,622.000,623.000,624.000,625.000,626.000,627.000,628.000,629.000,630.000,631.000,632.000,633.000,634.000,635.000,636.000,637.000,638.000,639.000,640.000,641.000,642.000,643.000,644.000,645.000,646.000,647.000,648.000,649.000,650.000,651.000,652.000,653.000,654.000,655.000,656.000,657.000,658.000,659.000,660.000,661.000,662.000,663.000,664.000,665.000,666.000,667.000,668.000,669.000,670.000,671.000,672.000,673.000,674.000,675.000,676.000,677.000,678.000,679.000,680.000,681.000,682.000,683.000,684.000,685.000,686.000,687.000,688.000,689.000,690.000,691.000,692.000,693.000,694.000,695.000,696.000,697.000,698.000,699.000,700.000,701.000,702.000,703.000,704.000,705.000,706.000,707.000,708.000,709.000,710.000,711.000,712.000,713.000,714.000,715.000,716.000,717.000,718.000,719.000,720.000,721.000,722.000,723.000,724.000,725.000,726.000,727.000,728.000,729.000,730.000,731.000,732.000,733.000,734.000,735.000,736.000,737.000,738.000,739.000,740.000,741.000,742.000,743.000,744.000,745.000,746.000,747.000,748.000,749.000,750.000,751.000,752.000,753.000,754.000,755.000,756.000,757.000,758.000,759.000,760.000,761.000,762.000,763.000,764.000,765.000,766.000,767.000,768.000,769.000,770.000,771.000,772.000,773.000,774.000,775.000,776.000,777.000,778.000,779.000,780.000,781.000,782.000,783.000,784.000,785.000,786.000,787.000,788.000,789.000,790.000,791.000,792.000,793.000,794.000,795.000,796.000,797.000,798.000,799.000,800.000,801.000,802.000,803.000,804.000,805.000,806.000,807.000,808.000,809.000,810.000,811.000,812.000,813.000,814.000,815.000,816.000,817.000,818.000,819.000,820.000,821.000,822.000,823.000,824.000,825.000,826.000,827.000,828.000,829.000,830.000],
[61.500,62.230,62.960,63.690,64.420,65.150,65.880,66.610,67.340,68.070,68.800,68.260,67.720,67.180,66.640,66.100,65.560,65.020,64.480,63.940,63.400,63.640,63.880,64.120,64.360,64.600,64.840,65.080,65.320,65.560,65.800,68.700,71.600,74.500,77.400,80.300,83.200,86.100,89.000,91.900,94.800,95.800,96.800,97.800,98.800,99.800,100.800,101.800,102.800,103.800,104.800,104.910,105.020,105.130,105.240,105.350,105.460,105.570,105.680,105.790,105.900,104.990,104.080,103.170,102.260,101.350,100.440,99.530,98.620,97.710,96.800,98.510,100.220,101.930,103.640,105.350,107.060,108.770,110.480,112.190,113.900,115.070,116.240,117.410,118.580,119.750,120.920,122.090,123.260,124.430,125.600,125.590,125.580,125.570,125.560,125.550,125.540,125.530,125.520,125.510,125.500,125.080,124.660,124.240,123.820,123.400,122.980,122.560,122.140,121.720,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,120.520,119.740,118.960,118.180,117.400,116.620,115.840,115.060,114.280,113.500,113.460,113.420,113.380,113.340,113.300,113.260,113.220,113.180,113.140,113.100,112.870,112.640,112.410,112.180,111.950,111.720,111.490,111.260,111.030,110.800,110.370,109.940,109.510,109.080,108.650,108.220,107.790,107.360,106.930,106.500,106.730,106.960,107.190,107.420,107.650,107.880,108.110,108.340,108.570,108.800,108.450,108.100,107.750,107.400,107.050,106.700,106.350,106.000,105.650,105.300,105.210,105.120,105.030,104.940,104.850,104.760,104.670,104.580,104.490,104.400,103.960,103.520,103.080,102.640,102.200,101.760,101.320,100.880,100.440,100.000,99.600,99.200,98.800,98.400,98.000,97.600,97.200,96.800,96.400,96.000,95.910,95.820,95.730,95.640,95.550,95.460,95.370,95.280,95.190,95.100,94.500,93.900,93.300,92.700,92.100,91.500,90.900,90.300,89.700,89.100,89.240,89.380,89.520,89.660,89.800,89.940,90.080,90.220,90.360,90.500,90.480,90.460,90.440,90.420,90.400,90.380,90.360,90.340,90.320,90.300,90.110,89.920,89.730,89.540,89.350,89.160,88.970,88.780,88.590,88.400,87.960,87.520,87.080,86.640,86.200,85.760,85.320,84.880,84.440,84.000,84.110,84.220,84.330,84.440,84.550,84.660,84.770,84.880,84.990,85.100,84.780,84.460,84.140,83.820,83.500,83.180,82.860,82.540,82.220,81.900,81.970,82.040,82.110,82.180,82.250,82.320,82.390,82.460,82.530,82.600,82.830,83.060,83.290,83.520,83.750,83.980,84.210,84.440,84.670,84.900,84.540,84.180,83.820,83.460,83.100,82.740,82.380,82.020,81.660,81.300,80.360,79.420,78.480,77.540,76.600,75.660,74.720,73.780,72.840,71.900,72.140,72.380,72.620,72.860,73.100,73.340,73.580,73.820,74.060,74.300,74.510,74.720,74.930,75.140,75.350,75.560,75.770,75.980,76.190,76.400,75.090,73.780,72.470,71.160,69.850,68.540,67.230,65.920,64.610,63.300,64.140,64.980,65.820,66.660,67.500,68.340,69.180,70.020,70.860,71.700,72.230,72.760,73.290,73.820,74.350,74.880,75.410,75.940,76.470,77.000,75.820,74.640,73.460,72.280,71.100,69.920,68.740,67.560,66.380,65.200,63.450,61.700,59.950,58.200,56.450,54.700,52.950,51.200,49.450,47.700,49.790,51.880,53.970,56.060,58.150,60.240,62.330,64.420,66.510,68.600,68.240,67.880,67.520,67.160,66.800,66.440,66.080,65.720,65.360,65.000,65.100,65.200,65.300,65.400,65.500,65.600,65.700,65.800,65.900,66.000,65.500,65.000,64.500,64.000,63.500,63.000,62.500,62.000,61.500,61.000,60.230,59.460,58.690,57.920,57.150,56.380,55.610,54.840,54.070,53.300,53.860,54.420,54.980,55.540,56.100,56.660,57.220,57.780,58.340,58.900,59.200,59.500,59.800,60.100,60.400,60.700,61.000,61.300,61.600,61.900],
[38.000,38.440,38.880,39.320,39.760,40.200,40.640,41.080,41.520,41.960,42.400,42.010,41.620,41.230,40.840,40.450,40.060,39.670,39.280,38.890,38.500,38.150,37.800,37.450,37.100,36.750,36.400,36.050,35.700,35.350,35.000,35.840,36.680,37.520,38.360,39.200,40.040,40.880,41.720,42.560,43.400,43.690,43.980,44.270,44.560,44.850,45.140,45.430,45.720,46.010,46.300,46.060,45.820,45.580,45.340,45.100,44.860,44.620,44.380,44.140,43.900,43.220,42.540,41.860,41.180,40.500,39.820,39.140,38.460,37.780,37.100,37.060,37.020,36.980,36.940,36.900,36.860,36.820,36.780,36.740,36.700,36.620,36.540,36.460,36.380,36.300,36.220,36.140,36.060,35.980,35.900,35.570,35.240,34.910,34.580,34.250,33.920,33.590,33.260,32.930,32.600,32.130,31.660,31.190,30.720,30.250,29.780,29.310,28.840,28.370,27.900,27.540,27.180,26.820,26.460,26.100,25.740,25.380,25.020,24.660,24.300,23.880,23.460,23.040,22.620,22.200,21.780,21.360,20.940,20.520,20.100,19.710,19.320,18.930,18.540,18.150,17.760,17.370,16.980,16.590,16.200,15.900,15.600,15.300,15.000,14.700,14.400,14.100,13.800,13.500,13.200,12.740,12.280,11.820,11.360,10.900,10.440,9.980,9.520,9.060,8.600,8.350,8.100,7.850,7.600,7.350,7.100,6.850,6.600,6.350,6.100,5.910,5.720,5.530,5.340,5.150,4.960,4.770,4.580,4.390,4.200,3.970,3.740,3.510,3.280,3.050,2.820,2.590,2.360,2.130,1.900,1.710,1.520,1.330,1.140,0.950,0.760,0.570,0.380,0.190,0.000,-0.160,-0.320,-0.480,-0.640,-0.800,-0.960,-1.120,-1.280,-1.440,-1.600,-1.790,-1.980,-2.170,-2.360,-2.550,-2.740,-2.930,-3.120,-3.310,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.730,-3.960,-4.190,-4.420,-4.650,-4.880,-5.110,-5.340,-5.570,-5.800,-5.940,-6.080,-6.220,-6.360,-6.500,-6.640,-6.780,-6.920,-7.060,-7.200,-7.340,-7.480,-7.620,-7.760,-7.900,-8.040,-8.180,-8.320,-8.460,-8.600,-8.690,-8.780,-8.870,-8.960,-9.050,-9.140,-9.230,-9.320,-9.410,-9.500,-9.640,-9.780,-9.920,-10.060,-10.200,-10.340,-10.480,-10.620,-10.760,-10.900,-10.880,-10.860,-10.840,-10.820,-10.800,-10.780,-10.760,-10.740,-10.720,-10.700,-10.830,-10.960,-11.090,-11.220,-11.350,-11.480,-11.610,-11.740,-11.870,-12.000,-12.200,-12.400,-12.600,-12.800,-13.000,-13.200,-13.400,-13.600,-13.800,-14.000,-13.960,-13.920,-13.880,-13.840,-13.800,-13.760,-13.720,-13.680,-13.640,-13.600,-13.440,-13.280,-13.120,-12.960,-12.800,-12.640,-12.480,-12.320,-12.160,-12.000,-12.130,-12.260,-12.390,-12.520,-12.650,-12.780,-12.910,-13.040,-13.170,-13.300,-13.260,-13.220,-13.180,-13.140,-13.100,-13.060,-13.020,-12.980,-12.940,-12.900,-12.670,-12.440,-12.210,-11.980,-11.750,-11.520,-11.290,-11.060,-10.830,-10.600,-10.700,-10.800,-10.900,-11.000,-11.100,-11.200,-11.300,-11.400,-11.500,-11.600,-11.660,-11.720,-11.780,-11.840,-11.900,-11.960,-12.020,-12.080,-12.140,-12.200,-12.000,-11.800,-11.600,-11.400,-11.200,-11.000,-10.800,-10.600,-10.400,-10.200,-9.960,-9.720,-9.480,-9.240,-9.000,-8.760,-8.520,-8.280,-8.040,-7.800,-8.140,-8.480,-8.820,-9.160,-9.500,-9.840,-10.180,-10.520,-10.860,-11.200,-11.120,-11.040,-10.960,-10.880,-10.800,-10.720,-10.640,-10.560,-10.480,-10.400,-10.420,-10.440,-10.460,-10.480,-10.500,-10.520,-10.540,-10.560,-10.580,-10.600,-10.510,-10.420,-10.330,-10.240,-10.150,-10.060,-9.970,-9.880,-9.790,-9.700,-9.560,-9.420,-9.280,-9.140,-9.000,-8.860,-8.720,-8.580,-8.440,-8.300,-8.400,-8.500,-8.600,-8.700,-8.800,-8.900,-9.000,-9.100,-9.200,-9.300,-9.350,-9.400,-9.450,-9.500,-9.550,-9.600,-9.650,-9.700,-9.750,-9.800],
[5.300,5.380,5.460,5.540,5.620,5.700,5.780,5.860,5.940,6.020,6.100,5.790,5.480,5.170,4.860,4.550,4.240,3.930,3.620,3.310,3.000,2.820,2.640,2.460,2.280,2.100,1.920,1.740,1.560,1.380,1.200,0.970,0.740,0.510,0.280,0.050,-0.180,-0.410,-0.640,-0.870,-1.100,-1.040,-0.980,-0.920,-0.860,-0.800,-0.740,-0.680,-0.620,-0.560,-0.500,-0.520,-0.540,-0.560,-0.580,-0.600,-0.620,-0.640,-0.660,-0.680,-0.700,-0.750,-0.800,-0.850,-0.900,-0.950,-1.000,-1.050,-1.100,-1.150,-1.200,-1.340,-1.480,-1.620,-1.760,-1.900,-2.040,-2.180,-2.320,-2.460,-2.600,-2.630,-2.660,-2.690,-2.720,-2.750,-2.780,-2.810,-2.840,-2.870,-2.900,-2.890,-2.880,-2.870,-2.860,-2.850,-2.840,-2.830,-2.820,-2.810,-2.800,-2.780,-2.760,-2.740,-2.720,-2.700,-2.680,-2.660,-2.640,-2.620,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.520,-2.440,-2.360,-2.280,-2.200,-2.120,-2.040,-1.960,-1.880,-1.800,-1.770,-1.740,-1.710,-1.680,-1.650,-1.620,-1.590,-1.560,-1.530,-1.500,-1.480,-1.460,-1.440,-1.420,-1.400,-1.380,-1.360,-1.340,-1.320,-1.300,-1.290,-1.280,-1.270,-1.260,-1.250,-1.240,-1.230,-1.220,-1.210,-1.200,-1.180,-1.160,-1.140,-1.120,-1.100,-1.080,-1.060,-1.040,-1.020,-1.000,-0.950,-0.900,-0.850,-0.800,-0.750,-0.700,-0.650,-0.600,-0.550,-0.500,-0.480,-0.460,-0.440,-0.420,-0.400,-0.380,-0.360,-0.340,-0.320,-0.300,-0.270,-0.240,-0.210,-0.180,-0.150,-0.120,-0.090,-0.060,-0.030,0.000,0.020,0.040,0.060,0.080,0.100,0.120,0.140,0.160,0.180,0.200,0.230,0.260,0.290,0.320,0.350,0.380,0.410,0.440,0.470,0.500,0.660,0.820,0.980,1.140,1.300,1.460,1.620,1.780,1.940,2.100,2.210,2.320,2.430,2.540,2.650,2.760,2.870,2.980,3.090,3.200,3.290,3.380,3.470,3.560,3.650,3.740,3.830,3.920,4.010,4.100,4.160,4.220,4.280,4.340,4.400,4.460,4.520,4.580,4.640,4.700,4.740,4.780,4.820,4.860,4.900,4.940,4.980,5.020,5.060,5.100,5.260,5.420,5.580,5.740,5.900,6.060,6.220,6.380,6.540,6.700,6.760,6.820,6.880,6.940,7.000,7.060,7.120,7.180,7.240,7.300,7.430,7.560,7.690,7.820,7.950,8.080,8.210,8.340,8.470,8.600,8.720,8.840,8.960,9.080,9.200,9.320,9.440,9.560,9.680,9.800,9.840,9.880,9.920,9.960,10.000,10.040,10.080,10.120,10.160,10.200,10.010,9.820,9.630,9.440,9.250,9.060,8.870,8.680,8.490,8.300,8.430,8.560,8.690,8.820,8.950,9.080,9.210,9.340,9.470,9.600,9.490,9.380,9.270,9.160,9.050,8.940,8.830,8.720,8.610,8.500,8.350,8.200,8.050,7.900,7.750,7.600,7.450,7.300,7.150,7.000,7.060,7.120,7.180,7.240,7.300,7.360,7.420,7.480,7.540,7.600,7.640,7.680,7.720,7.760,7.800,7.840,7.880,7.920,7.960,8.000,7.870,7.740,7.610,7.480,7.350,7.220,7.090,6.960,6.830,6.700,6.550,6.400,6.250,6.100,5.950,5.800,5.650,5.500,5.350,5.200,5.420,5.640,5.860,6.080,6.300,6.520,6.740,6.960,7.180,7.400,7.340,7.280,7.220,7.160,7.100,7.040,6.980,6.920,6.860,6.800,6.820,6.840,6.860,6.880,6.900,6.920,6.940,6.960,6.980,7.000,6.940,6.880,6.820,6.760,6.700,6.640,6.580,6.520,6.460,6.400,6.310,6.220,6.130,6.040,5.950,5.860,5.770,5.680,5.590,5.500,5.560,5.620,5.680,5.740,5.800,5.860,5.920,5.980,6.040,6.100,6.140,6.180,6.220,6.260,6.300,6.340,6.380,6.420,6.460,6.500]
])  
    

#--------------------------------------------------------------------------------------------------
# Define interpolation types (conform CIE2004:15): 
_INTERP_TYPES = {'linear' : ['rfl','RFL','r','R','xyzbar','cmf','lms','undefined'],'cubic': ['S', 'spd','SPD','Le'],'none':None}
_S_INTERP_TYPE = 'cubic'
_R_INTERP_TYPE = 'linear'


#--------------------------------------------------------------------------------------------------
# reference illuminant default and mixing range settings:
_CRI_REF_TYPE = 'ciera'
_CRI_REF_TYPES = {'ciera': [5000.0 , 5000.0], 'cierf': [4000.0, 5000.0], 'cierf-224-2017': [4000.0, 5000.0],\
                  'iesrf':[4000.0, 5000.0],'iesrf-tm30-15':[4500.0, 5500.0],'iesrf-tm30-18':[4000.0, 5000.0],\
                  'BB':[5000.0,5000.0],'DL':[5000.0,5000.0]} #mixing ranges for various cri_reference_illuminant types


#--------------------------------------------------------------------------------------------------
def getwlr(wl3 = None):
    """
    Get/construct a wavelength range from a 3-vector (start, stop, spacing).
    
    Args:
        :wl3: 
            | list[start, stop, spacing], optional 
            | (defaults to luxpy._WL3)

    Returns:
        :returns: 
            | ndarray (.shape = (n,)) with n wavelengths ranging from
            | start to stop, with wavelength interval equal to spacing.
    """
    if wl3 is None: wl3 = _WL3
    
    # Wavelength definition:
    wl = wl3 if (len(wl3) != 3) else np.linspace(wl3[0],wl3[1],int(np.floor((wl3[1]-wl3[0]+wl3[2])/wl3[2]))) # define wavelengths from [start = l0, stop = ln, spacing = dl]
    
    return wl

#------------------------------------------------------------------------------
def getwld(wl):
    """
    Get wavelength spacing. 
    
    Args:
        :wl: 
            | ndarray with wavelengths
        
    Returns:
        :returns: 
            | - float:  for equal wavelength spacings
            | - ndarray (.shape = (n,)): for unequal wavelength spacings
    """
    d = np.diff(wl)
    dl = (np.hstack((d[0],d[0:-1]/2.0,d[-1]))+np.hstack((0.0,d[1:]/2.0,0.0)))
    if np.array_equal(dl,dl.mean()*np.ones(dl.shape)): dl = dl[0]
    return dl


#------------------------------------------------------------------------------
def spd_normalize(data, norm_type = None, norm_f = 1, wl = True, cieobs = _CIEOBS):
    """
    Normalize a spectral power distribution (SPD).
    
    Args:
        :data: 
            | ndarray
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            | for 'max' and 'area' 
            | or which wavelength is normalized to 1 for 'lambda' option.
        :wl: 
            | True or False, optional 
            | If True, the first column of data contains wavelengths.
        :cieobs:
            | _CIEOBS or str, optional
            | Type of cmf set to use for normalization using photometric units 
            | (norm_type == 'pu')
    
    Returns:
        :returns: 
            | ndarray with normalized data.
    """
    if norm_type is not None:
        if not isinstance(norm_type,list): norm_type = [norm_type]
        
        if norm_f is not None:
            if not isinstance(norm_f,list): norm_f = [norm_f]
                
        if ('lambda' in norm_type) | ('qu' in norm_type):
            wl = True # for lambda & 'qu' normalization wl MUST be first column
            wlr = data[0]
            
        if (('area' in norm_type) | ('ru' in norm_type) | ('pu' in norm_type) | ('pusa' in norm_type)) & (wl == True):
            dl = getwld(data[0])
        else:
            dl = 1 #no wavelengths provided
            
        offset = int(wl)
        for i in range(data.shape[0]-offset):  
            norm_type_ = norm_type[i] if (len(norm_type)>1) else norm_type[0]

            if norm_f is not None:
                norm_f_ = norm_f[i] if (len(norm_f)>1) else norm_f[0]
            else:
                norm_f_ = 560.0 if (norm_type_ == 'lambda') else 1.0
      
            if norm_type_=='max':
                data[i+offset]=norm_f_*data[i+offset]/np.max(data[i+offset])
            elif norm_type_=='area':
                data[i+offset]=norm_f_*data[i+offset]/(np.sum(data[i+offset])*dl)
            elif norm_type_=='lambda':
                wl_index = np.abs(wlr-norm_f_).argmin()
                data[i+offset]=data[i+offset]/data[i+offset][wl_index]
            elif (norm_type_ == 'ru') | (norm_type_ == 'pu') | (norm_type == 'pusa') | (norm_type_ == 'qu'):
                rpq_power = spd_to_power(data[[0,i+offset],:], cieobs = cieobs, ptype = norm_type_)
                data[i+offset] = (norm_f/rpq_power)*data[i+offset]
            else:
                data[i+offset]=data[i+offset]/norm_f_
    return data

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def cie_interp(data,wl_new, kind = None, negative_values_allowed = False, extrap_values = None):
    """
    Interpolate / extrapolate spectral data following standard CIE15-2018.
    
    | The kind of interpolation depends on the spectrum type defined in :kind:. 
    | Extrapolation is always done by replicate the closest known values.
    
    Args:
        :data: 
            | ndarray with spectral data 
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :wl_new: 
            | ndarray with new wavelengths
        :kind: 
            | None, optional
            |   - If :kind: is None, return original data.
            |   - If :kind: is a spectrum type (see _INTERP_TYPES), the correct 
            |     interpolation type if automatically chosen.
            |   - Or :kind: can be any interpolation type supported by 
            |     scipy.interpolate.interp1d (math.interp1d if nan's are present!!)
        :negative_values_allowed: 
            | False, optional
            | If False: negative values are clipped to zero.
        :extrap_values:
            | None, optional
            | If None: use CIE recommended 'closest value' approach when extrapolating.
            | If float or list or ndarray, use those values to fill extrapolated value(s).
            | If 'ext': use normal extrapolated values by scipy.interpolate.interp1d
    
    Returns:
        :returns: 
            | ndarray of interpolated spectral data.
            | (.shape = (number of spectra + 1, number of wavelength in wl_new))
    """
    if (kind is not None):
        # Wavelength definition:
        wl_new = getwlr(wl_new)
        
        if (not np.array_equal(data[0],wl_new)) | np.isnan(data).any():
       
            extrap_values = np.atleast_1d(extrap_values)

            # Set interpolation type based on data type:
            if kind in _INTERP_TYPES['linear']:
                kind = 'linear'
            elif kind in _INTERP_TYPES['cubic']:
                kind = 'cubic'

            # define wl, S, wl_new:
            wl = np.array(data[0])
            S = data[1:]
            wl_new = np.array(wl_new)
        
            # Interpolate each spectrum in S: 
            N = S.shape[0]
            nan_indices = np.isnan(S)
            
            # Interpolate all (if not all rows have nan):
            rows_with_nans = np.where(nan_indices.sum(axis=1))[0]
            if not (rows_with_nans.size == N):
                #allrows_nans = False
                if extrap_values[0] is None:
                    fill_value = (0,0)
                elif extrap_values[0][:3] == 'ext':#(((type(extrap_values[0])==np.str_)|(type(extrap_values[0])==str)) and (extrap_values[0][:3]=='ext')):
                    fill_value = 'extrapolate'
                else:
                    fill_value = (extrap_values[0],extrap_values[-1])
                Si = sp.interpolate.interp1d(wl, S, kind = kind, bounds_error = False, fill_value = fill_value)(wl_new)
                
                #extrapolate by replicating closest known (in source data!) value (conform CIE2004 recommendation) 
                if extrap_values[0] is None:
                    Si[:,wl_new<wl[0]] = S[:,:1]
                    Si[:,wl_new>wl[-1]] = S[:,-1:]  
                    
            else:
                #allrows_nans = True
                Si = np.zeros([N,wl_new.shape[0]]);Si.fill(np.nan)
            
            # Re-interpolate those which have none:
            if nan_indices.any():
                #looping required as some values are NaN's
                for i in rows_with_nans:
                    
                    nonan_indices = np.logical_not(nan_indices[i])
                    wl_nonan = wl[nonan_indices]
                    S_i_nonan = S[i][nonan_indices]
                    Si_nonan = math.interp1(wl_nonan,S_i_nonan, wl_new, kind = kind, ext = 'extrapolate')
#                    Si_nonan = sp.interpolate.interp1d(wl_nonan, S_i_nonan, kind = kind, bounds_error = False, fill_value = 'extrapolate')(wl_new)
                  
                    #extrapolate by replicating closest known (in source data!) value (conform CIE2004 recommendation) 
                    if extrap_values[0] is None:
                        Si_nonan[wl_new<wl_nonan[0]] = S_i_nonan[0]
                        Si_nonan[wl_new>wl_nonan[-1]] = S_i_nonan[-1]
                    elif extrap_values[0][:3] == 'ext':#(((type(extrap_values[0])==np.str_)|(type(extrap_values[0])==str)) and (extrap_values[0][:3]=='ext')):
                        pass
                    else:
                        Si_nonan[wl_new<wl_nonan[0]] = extrap_values[0]
                        Si_nonan[wl_new>wl_nonan[-1]] = extrap_values[-1]  
                    Si[i] = Si_nonan              
                
            # No negative values allowed for spectra:    
            if negative_values_allowed == False:
                if np.any(Si): Si[Si<0.0] = 0.0
            
            # Add wavelengths to data array: 
            return np.vstack((wl_new,Si))  
    
    return data


#--------------------------------------------------------------------------------------------------
def spd(data = None, interpolation = None, kind = 'np', wl = None,\
        columns = None, sep = ',',header = None, datatype = 'S', \
        norm_type = None, norm_f = None):
    """
    | All-in-one function that can:
    |    1. Read spectral data from data file or take input directly 
         as pandas.dataframe or ndarray.
    |    2. Convert spd-like data from ndarray to pandas.dataframe and back.
    |    3. Interpolate spectral data.
    |    4. Normalize spectral data.
            
    Args:
        :data: 
            | - str with path to file containing spectral data
            | - ndarray with spectral data
            | - pandas.dataframe with spectral data
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :interpolation:
            | None, optional
            | - None: don't interpolate
            | - str with interpolation type or spectrum type
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :columns: 
            | -  None or list[str] of column names for dataframe, optional
        :header: 
            | None or 'infer', optional
            | - None: no header in file
            | - 'infer': infer headers from file
        :sep: 
            | ',' or '\t' or other char, optional
            | Column separator in case :data: specifies a data file. 
        :datatype': 
            | 'S' (light source) or 'R' (reflectance) or other, optional
            | Specifies a type of spectral data. 
            | Is used when creating column headers when :column: is None.
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
              for 'max' and 'area' 
              or which wavelength is normalized to 1 for 'lambda' option.
    
    Returns:
        :returns: 
            | ndarray or pandas.dataframe 
            | with interpolated and/or normalized spectral data.
    """
    transpose = True if isinstance(data,str) else False #when spd comes from file -> transpose (columns in files should be different spectra)
         
    # Wavelength definition:
    wl = getwlr(wl)
    
    # Data input:
    if data is not None:
        if (interpolation is None) & (norm_type is None):
            data = getdata(data = data, kind = 'np', columns = columns, sep = sep, header = header, datatype = datatype, copy = True)
            if (transpose == True): data = data.T
        else:
            data = getdata(data = data, kind = 'np', columns = columns, sep = sep, header = header, datatype = datatype, copy = True)#interpolation requires np-array as input
            if (transpose == True): data = data.T
            data = cie_interp(data = data, wl_new = wl,kind = interpolation)
            data = spd_normalize(data,norm_type = norm_type, norm_f = norm_f, wl = True)
        
        if isinstance(data,pd.DataFrame): columns = data.columns #get possibly updated column names

    else:
        data = np2d(wl)
  
     
    if ((data.shape[0] - 1) == 0): columns = None #only wavelengths
       
    if kind == 'df':  data = data.T
        
    # convert to desired kind:
    data = getdata(data = data,kind = kind, columns = columns, datatype = datatype, copy = False) # already copy when data is not None, else new anyway
        
    return data


#--------------------------------------------------------------------------------------------------
def xyzbar(cieobs = _CIEOBS, scr = 'dict', wl_new = None, norm_type = None, norm_f = None, kind = 'np'):
    """
    Get color matching functions.  
    
    Args:
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Sets the type of color matching functions to load.
        :scr: 
            | 'dict' or 'file', optional
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            | for 'max' and 'area' 
            | or which wavelength is normalized to 1 for 'lambda' option.
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe

    Returns:
        :returns: 
            | ndarray or pandas.dataframe with CMFs 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if scr is 'file':
        dict_or_file = _PKG_PATH + _SEP + 'data' + _SEP + 'cmfs' + _SEP + 'ciexyz_' + cieobs + '.dat'
    elif scr is 'dict':
        dict_or_file = _CMF[cieobs]['bar']
    elif scr == 'cieobs':
        dict_or_file = cieobs #can be file or data itself
    return spd(data = dict_or_file, wl = wl_new, interpolation = 'linear', kind = kind, columns = ['wl','xb','yb','zb'])

#--------------------------------------------------------------------------------------------------
def vlbar(cieobs = _CIEOBS, scr = 'dict', wl_new = None, norm_type = None, norm_f = None, kind = 'np', out = 1):
    """
    Get Vlambda functions.  
    
    Args:
        :cieobs: 
            | str, optional
            | Sets the type of Vlambda function to obtain.
        :scr: 
            | 'dict' or array, optional
            | - 'dict': get from ybar from _CMF
            | - 'array': ndarray in :cieobs:
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
            | Vlambda is obtained by collecting Ybar.
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            | for 'max' and 'area' 
            | or which wavelength is normalized to 1 for 'lambda' option.
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe
        :out: 
            | 1 or 2, optional
            |     1: returns Vlambda
            |     2: returns (Vlambda, Km)
    
    Returns:
        :returns: 
            | dataframe or ndarray with Vlambda of type :cieobs: 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if scr is 'dict':
        dict_or_file = _CMF[cieobs]['bar'][[0,2],:] 
        K = _CMF[cieobs]['K']
    elif scr is 'vltype':
        dict_or_file = cieobs #can be file or data itself
        K = 1
    Vl = spd(data = dict_or_file, wl = wl_new, interpolation = 'linear', kind = kind, columns = ['wl','Vl'])

    if out == 2:
        return Vl, K
    else:
        return Vl


#--------------------------------------------------------------------------------------------------
def spd_to_xyz(data,  relative = True, rfl = None, cieobs = _CIEOBS, K = None, out = None, cie_std_dev_obs = None):
    """
    Calculates xyz tristimulus values from spectral data.
       
    Args: 
        :data: 
            | ndarray or pandas.dataframe with spectral data
            | (.shape = (number of spectra + 1, number of wavelengths))
            | Note that :data: is never interpolated, only CMFs and RFLs. 
            | This way interpolation errors due to peaky spectra are avoided. 
              Conform CIE15-2018.
        :relative: 
            | True or False, optional
            | Calculate relative XYZ (Yw = 100) or absolute XYZ (Y = Luminance)
        :rfl: 
            | ndarray with spectral reflectance functions.
            | Will be interpolated if wavelengths do not match those of :data:
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines the color matching functions to be used in the 
              calculation of XYZ.
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2' (relative == False) 
            |   or K = 100/sum(spd*dl)        (relative == True)
        :out:
            | None or 1 or 2, optional
            | Determines number and shape of output. (see :returns:)
        :cie_std_dev_obs: 
            | None or str, optional
            | - None: don't use CIE Standard Deviate Observer function.
            | - 'f1': use F1 function.
    
    Returns:
        :returns:
            | If rfl is None:
            |    If out is None: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 1: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        Note that xyz == xyzw, with (.shape = (data.shape[0],3))
            | If rfl is not None:
            |   If out is None: ndarray of xyz values 
            |         (.shape = (rfl.shape[0],data.shape[0],3))
            |   If out == 1: ndarray of xyz values 
            |       (.shape = (rfl.shape[0]+1,data.shape[0],3))
            |        The xyzw values of the light source spd are the first set 
            |        of values of the first dimension. The following values 
            |       along this dimension are the sample (rfl) xyz values.
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        with xyz.shape = (rfl.shape[0],data.shape[0],3)
            |        and with xyzw.shape = (data.shape[0],3)
             
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    
    data = getdata(data,kind = 'np') if isinstance(data,pd.DataFrame) else np2d(data) # convert to np format and ensure 2D-array

    # get wl spacing:
    dl = getwld(data[0])
    
    # get cmf,k for cieobs:
    if isinstance(cieobs,str):
        if K is None: K = _CMF[cieobs]['K']
        scr = 'dict'
    else:
        scr = 'cieobs'
        if (K is None) & (relative == False): K = 1
    
    # Interpolate to wl of data:
    cmf = xyzbar(cieobs = cieobs, scr = scr, wl_new = data[0], kind = 'np') 
    
    # Add CIE standard deviate observer function to cmf if requested:
    if cie_std_dev_obs is not None:
        cmf_cie_std_dev_obs = xyzbar(cieobs = 'cie_std_dev_obs_' + cie_std_dev_obs.lower(), scr = scr, wl_new = data[0], kind = 'np')
        cmf[1:] = cmf[1:] + cmf_cie_std_dev_obs[1:] 
    
    # Rescale xyz using k or 100/Yw:
    if relative == True: K = 100.0/np.dot(data[1:],cmf[2,:]*dl)

    # Interpolate rfls to lambda range of spd and calculate xyz:
    if rfl is not None: 
        rfl = cie_interp(data=np2d(rfl),wl_new = data[0],kind = 'rfl')
        rfl = np.concatenate((np.ones((1,data.shape[1])),rfl[1:])) #add rfl = 1 for light source spectrum
        xyz = K*np.array([np.dot(rfl,(data[1:]*cmf[i+1,:]*dl).T) for i in range(3)])#calculate tristimulus values
        rflwasnotnone = 1
    else:
        rfl = np.ones((1,data.shape[1]))
        xyz = (K*(np.dot((cmf[1:]*dl),data[1:].T))[:,None,:])
        rflwasnotnone = 0
    xyz = np.transpose(xyz,[1,2,0]) #order [rfl,spd,xyz]
    
    # Setup output:
    if out == 2:
        xyzw = xyz[0,...]
        xyz = xyz[rflwasnotnone:,...]
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz,xyzw
    elif out == 1:
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz
    else: 
        xyz = xyz[rflwasnotnone:,...]
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz

def spd_to_ler(data, cieobs = _CIEOBS, K = None):
    """
    Calculates Luminous efficacy of radiation (LER) from spectral data.
       
    Args: 
        :data: 
            | ndarray or pandas.dataframe with spectral data
            | (.shape = (number of spectra + 1, number of wavelengths))
            | Note that :data: is never interpolated, only CMFs and RFLs. 
            | This way interpolation errors due to peaky spectra are avoided. 
            | Conform CIE15-2018.
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Determines the color matching function set used in the 
            | calculation of LER. For cieobs = '1931_2' the ybar CMF curve equals
            | the CIE 1924 Vlambda curve.
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2'
      
    Returns:
        :ler: 
            | ndarray of LER values. 
             
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    
    if isinstance(cieobs,str):    
        if K == None: K = _CMF[cieobs]['K']
        Vl = vlbar(cieobs = cieobs, scr = 'dict',wl_new = data[0], kind = 'np')[1:2] #also interpolate to wl of data
    else:
        Vl = spd(wl = data[0], data = cieobs, interpolation = 'cmf', kind = 'np')[1:2]
        if K is None: raise Exception("spd_to_ler: User defined Vlambda, but no K scaling factor has been supplied.")
    dl = getwld(data[0])
    return ((K * np.dot((Vl*dl),data[1:].T))/np.sum(data[1:]*dl, axis = data.ndim-1)).T


def spd_to_power(data, ptype = 'ru', cieobs = _CIEOBS):
    """
    Calculate power of spectral data in radiometric, photometric 
    or quantal energy units.
    
    Args:
        :data: 
            | ndarray with spectral data
        :ptype: 
            | 'ru' or str, optional
            | str: - 'ru': in radiometric units 
            |      - 'pu': in photometric units 
            |      - 'pusa': in photometric units with Km corrected 
            |                to standard air (cfr. CIE TN003-2015)
            |      - 'qu': in quantal energy units
        :cieobs: 
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
    
    Returns:
        returns: 
            | ndarray with normalized spectral data (SI units)
    """
    # get wavelength spacing:
    dl = getwld(data[0])
    
    if ptype == 'ru': #normalize to radiometric units
        p = np2d(np.dot(data[1:],dl*np.ones(data.shape[1]))).T

    elif ptype == 'pusa': # normalize in photometric units with correction of Km to standard air
    
        # Calculate correction factor for Km in standard air:
        na = _BB['na'] # n for standard air
        c = _BB['c'] # m/s light speed
        lambdad = c/(na*54*1e13)/(1e-9) # 555 nm lambda in standard air
        Km_correction_factor = 1/(1 - (1 - 0.9998567)*(lambdad - 555)) # correction factor for Km in standard air

        # Get Vlambda and Km (for E):
        Vl, Km = vlbar(cieobs = cieobs, wl_new = data[0], out = 2)
        Km *= Km_correction_factor
        p = Km*np2d(np.dot(data[1:],dl*Vl[1])).T
        
    elif ptype == 'pu': # normalize in photometric units
    
        # Get Vlambda and Km (for E):
        Vl, Km = vlbar(cieobs = cieobs, wl_new = data[0], out = 2)
        p = Km*np2d(np.dot(data[1:],dl*Vl[1])).T

    
    elif ptype == 'qu': # normalize to quantual units

        # Get Quantal conversion factor:
        fQ = ((1e-9)/(_BB['h']*_BB['c']))
        p = np2d(fQ*np.dot(data[1:],dl*data[0])).T

    return p


    
#------------------------------------------------------------------------------
#---CIE illuminants------------------------------------------------------------
#------------------------------------------------------------------------------

def blackbody(cct, wl3 = None):
    """
    Calculate blackbody radiator spectrum for correlated color temperature (cct).
    
    Args:
        :cct: 
            | int or float 
            | (for list of cct values, use cri_ref() with ref_type = 'BB')
        :wl3: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.

    Returns:
        :returns:
            | ndarray with blackbody radiator spectrum
            | (:returns:[0] contains wavelengths)
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    cct = float(cct)
    if wl3 is None: wl3 = _WL3 
    wl=getwlr(wl3)
    def fSr(x):
        return (1/np.pi)*_BB['c1']*((x*1.0e-9)**(-5))*(_BB['n']**(-2.0))*(np.exp(_BB['c2']*((_BB['n']*x*1.0e-9*(cct+_EPS))**(-1.0)))-1.0)**(-1.0)
    return np.vstack((wl,(fSr(wl)/fSr(560.0))))

#------------------------------------------------------------------------------
def daylightlocus(cct, force_daylight_below4000K = False):
    """ 
    Calculates daylight chromaticity from correlated color temperature (cct).
    
    Args:
        :cct: 
            | int or float or list of int/floats or ndarray
        :force_daylight_below4000K: 
            | False or True, optional
            | Daylight locus approximation is not defined below 4000 K, 
            | but by setting this to True, the calculation can be forced to 
            | calculate it anyway.
    
    Returns:
        :returns: 
            | (ndarray of x-coordinates, ndarray of y-coordinates)
        
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    cct = np2d(cct)
    if np.any((cct < 4000.0) & (force_daylight_below4000K == False)):
        raise Exception('spectral.daylightlocus(): Daylight locus approximation not defined below 4000 K')
    
    xD = -4.607*((1e3/cct)**3.0)+2.9678*((1e3/cct)**2.0)+0.09911*(1000.0/cct)+0.244063
    p = cct>=7000.0
    xD[p] = -2.0064*((1.0e3/cct[p])**3.0)+1.9018*((1.0e3/cct[p])**2.0)+0.24748*(1.0e3/cct[p])+0.23704
    yD = -3.0*xD**2.0+2.87*xD-0.275
    return xD,yD
    
   
   
#------------------------------------------------------------------------------
def daylightphase(cct, wl3 = None, force_daylight_below4000K = False, verbosity = None):
    """
    Calculate daylight phase spectrum for correlated color temperature (cct).
        
    Args:
        :cct: 
            | int or float 
            | (for list of cct values, use cri_ref() with ref_type = 'DL')
        :wl3: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :force_daylight_below4000K: 
            | False or True, optional
            | Daylight locus approximation is not defined below 4000 K, 
            | but by setting this to True, the calculation can be forced to 
            | calculate it anyway.
        :verbosity: 
            | None, optional
            |   If None: do not print warning when CCT < 4000 K.

    Returns:
        :returns: 
            | ndarray with daylight phase spectrum
            | (:returns:[0] contains wavelengths)

    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
     """
    cct = float(cct)
    if wl3 is None: wl3 = _WL3 
    if (cct < (4000.0)) & (force_daylight_below4000K == False):
        if verbosity is not None:
            print('Warning daylightphase spd not defined below 4000 K. Using blackbody radiator instead.')
        return blackbody(cct,wl3)
    else:
        
        wl = getwlr(wl3) 
        
        #interpolate _S012_DAYLIGHTPHASE first to wl range:
        if  not np.array_equal(_S012_DAYLIGHTPHASE[0],wl):
            S012_daylightphase = cie_interp(data = _S012_DAYLIGHTPHASE, wl_new = wl, kind = 'linear',negative_values_allowed = True)
        else:
            S012_daylightphase = _S012_DAYLIGHTPHASE

        xD, yD = daylightlocus(cct, force_daylight_below4000K = force_daylight_below4000K)
        
        M1 = (-1.3515-1.7703*xD+5.9114*yD)/(0.0241+0.2562*xD-0.7341*yD)
        M2 = (0.03-31.4424*xD+30.0717*yD)/(0.0241+0.2562*xD-0.7341*yD)
        Sr = S012_daylightphase[1,:]+M1*S012_daylightphase[2,:]+M2*S012_daylightphase[3,:]
        Sr560 = Sr[:,np.where(np.abs(S012_daylightphase[0,:] - 560.0) == np.min(np.abs(S012_daylightphase[0,:] - 560)))[0]]
        Sr /= Sr560
        Sr[Sr==float('NaN')] = 0
        return np.vstack((wl,Sr))
    
#------------------------------------------------------------------------------
def cri_ref(ccts, wl3 = None, ref_type = _CRI_REF_TYPE, mix_range = None, cieobs=_CIEOBS, norm_type = None, norm_f = None, force_daylight_below4000K = False):
    """
    Calculates a reference illuminant spectrum based on cct 
    for color rendering index calculations .
    
    Args:
        :ccts: 
            | list of int/floats or ndarray with ccts.
        :wl3: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :ref_type:
            | str or list[str], optional
            | Specifies the type of reference spectrum to be calculated.
            | Defaults to luxpy._CRI_REF_TYPE. 
            | If :ref_type: is list of strings, then for each cct in :ccts: 
            | a different reference illuminant can be specified. 
            | If :ref_type: == 'spd', then :ccts: is assumed to be an ndarray
            | of reference illuminant spectra.
        :mix_range: 
            | None or ndarray, optional
            | Determines the cct range between which the reference illuminant is
            | a weigthed mean of a Planckian and Daylight Phase spectrum. 
            | Weighthing is done as described in IES TM30:
            |    SPDreference = (Te-T)/(Te-Tb)*Planckian+(T-Tb)/(Te-Tb)*daylight
            |    with Tb and Te are resp. the starting and end CCTs of the 
            |    mixing range and whereby the Planckian and Daylight SPDs 
            |    have been normalized for equal luminous flux.
            | If None: use the default specified for :ref_type:.
            | Can be a ndarray with shape[0] > 1, in which different mixing
            | ranges will be used for cct in :ccts:.
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Required for the normalization of the Planckian and Daylight SPDs 
            | when calculating a 'mixed' reference illuminant.
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            | for 'max' and 'area' 
            | or which wavelength is normalized to 1 for 'lambda' option.
        :force_daylight_below4000K: 
            | False or True, optional
            | Daylight locus approximation is not defined below 4000 K, 
            | but by setting this to True, the calculation can be forced to 
            | calculate it anyway.
    
    Returns:
        :returns: 
            | ndarray with reference illuminant spectra.
            | (:returns:[0] contains wavelengths)

    Note: 
        Future versions will have the ability to take a dict as input 
        for ref_type. This way other reference illuminants can be specified 
        than the ones in _CRI_REF_TYPES. 
    """
    if ref_type == 'spd':
        
        # ccts already contains spectrum of reference:
        return spd(ccts, wl = wl3, norm_type = norm_type, norm_f = norm_f)

    else:
        if mix_range is not None: mix_range = np2d(mix_range)

        if not (isinstance(ref_type,list) | isinstance(ref_type,dict)): ref_type = [ref_type]
   
        for i in range(len(ccts)):
            cct = ccts[i]

            # get ref_type and mix_range:
            if isinstance(ref_type,dict):
                raise Exception("cri_ref(): dictionary ref_type: Not yet implemented")
            else:

                ref_type_ = ref_type[i] if (len(ref_type)>1) else ref_type[0]

                if mix_range is None:
                    mix_range_ =  _CRI_REF_TYPES[ref_type_]

                else:
                    mix_range_ = mix_range[i] if (mix_range.shape[0]>1) else mix_range[0]  #must be np2d !!!            
      
            if (mix_range_[0] == mix_range_[1]) | (ref_type_[0:2] == 'BB') | (ref_type_[0:2] == 'DL'):
                if ((cct < mix_range_[0]) & (not (ref_type_[0:2] == 'DL'))) | (ref_type_[0:2] == 'BB'):
                    Sr = blackbody(cct,wl3)
                elif ((cct >= mix_range_[0]) & (not (ref_type_[0:2] == 'BB'))) | (ref_type_[0:2] == 'DL') :
                    Sr = daylightphase(cct,wl3,force_daylight_below4000K = force_daylight_below4000K)
            else:
                SrBB = blackbody(cct,wl3)
                SrDL = daylightphase(cct,wl3,verbosity = None,force_daylight_below4000K = force_daylight_below4000K)
                cmf = xyzbar(cieobs = cieobs, scr = 'dict', wl_new = wl3)
                wl = SrBB[0]
                ld = getwld(wl)

                SrBB = 100.0*SrBB[1]/np.array(np.sum(SrBB[1]*cmf[2]*ld))
                SrDL = 100.0*SrDL[1]/np.array(np.sum(SrDL[1]*cmf[2]*ld))
                Tb, Te = float(mix_range_[0]), float(mix_range_[1])
                cBB, cDL = (Te-cct)/(Te-Tb), (cct-Tb)/(Te-Tb)
                if cBB < 0.0:
                    cBB = 0.0
                elif cBB > 1:
                    cBB = 1.0
                if cDL < 0.0:
                    cDL = 0.0
                elif cDL > 1:
                    cDL = 1.0

                Sr = SrBB*cBB + SrDL*cDL
                Sr[Sr==float('NaN')] = 0.0
                Sr560 = Sr[np.where(np.abs(wl - 560.0) == np.min(np.abs(wl - 560.0)))[0]]
                Sr = np.vstack((wl,(Sr/Sr560)))
                     
            if i == 0:
                Srs = Sr[1]
            else:
                Srs = np.vstack((Srs,Sr[1]))
                    
        Srs = np.vstack((Sr[0],Srs))

        return  spd(Srs, wl = None, norm_type = norm_type, norm_f = norm_f)
    
    
def detect_peakwl(spd, n = 1,verbosity = 1, **kwargs):
    """
    Detect primary peak wavelengths and fwhm in spectrum spd.
    
    Args:
        :spd:
            | ndarray with spectral data (2xN). 
            | First row should be wavelengths.
        :n:
            | 1, optional
            | The number of peaks to try to detect in spd. 
        :verbosity:
            | Make a plot of the detected peaks, their fwhm, etc.
        :kwargs:
            | Additional input arguments for scipy.signal.find_peaks.
    Returns:
        :prop:
            | list of dictionaries with keys: 
            | - 'peaks_idx' : index of detected peaks
            | - 'peaks' : peak wavelength values (nm)
            | - 'heights' : height of peaks
            | - 'fwhms' : full-width-half-maxima of peaks
            | - 'fwhms_mid' : wavelength at the middle of the fwhm-range of the peaks (if this is different from the values in 'peaks', then their is some non-symmetry in the peaks)
            | - 'fwhms_mid_heights' : height at the middle of the peak
    """
    props = []
    for i in range(spd.shape[0]-1):
        peaks_, prop_ = signal.find_peaks(spd[i+1,:], **kwargs)
        prominences = signal.peak_prominences(spd[i+1,:], peaks_)[0]
        peaks = [peaks_[prominences.argmax()]]
        prominences[prominences.argmax()] = 0
        for j in range(n-1):
            peaks.append(peaks_[prominences.argmax()])
            prominences[prominences.argmax()] = 0
        peaks = np.sort(np.array(peaks))
        peak_heights = spd[i+1,peaks]
        widths, width_heights, left_ips, right_ips = signal.peak_widths(spd[i+1,:], peaks, rel_height=0.5)
        left_ips, right_ips = left_ips + spd[0,0], right_ips + spd[0,0]
    
        # get middle of fwhm and calculate peak position and height:
        mpeaks = left_ips + widths/2
        hmpeaks = sp.interpolate.interp1d(spd[0,:],spd[i+1,:])(mpeaks)
    
        prop = {'peaks_idx' : peaks,'peaks' : spd[0,peaks], 'heights' : peak_heights,
                'fwhms' : widths, 'fwhms_mid' : mpeaks, 'fwhms_mid_heights' : hmpeaks}
        props.append(prop)
        if verbosity == 1:
            print('Peak properties:', prop)
            results_half = (widths, width_heights, left_ips, right_ips)
            plt.plot(spd[0,:],spd[i+1,:],'b-',label = 'spectrum')
            plt.plot(spd[0,peaks],spd[i+1,peaks],'ro', label = 'peaks')
            plt.hlines(*results_half[1:], color="C2", label = 'FWHM range of peaks')
            plt.plot(mpeaks,hmpeaks,'gd', label = 'middle of FWHM range')
    return props


def spd_to_indoor(spd):
    """
    Convert spd to indoor variant by multiplying it with the CIE spectral transmission for glass.
    """
    Tglass = cie_interp(_CIE_GLASS_ID['T'].copy(), spd[0,:], kind = 'rfl')[1:,:]
    spd_ = spd.copy()
    spd_[1:,:] *= Tglass
    return spd_