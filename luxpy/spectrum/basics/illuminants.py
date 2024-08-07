# -*- coding: utf-8 -*-
"""
Module for (CIE) illuminants
============================

 :_BB: Dict with constants for blackbody radiator calculation 
       constant are (c1, c2, n, na, c, h, k). 

 :_S012_DAYLIGHTPHASE: ndarray with CIE S0,S1, S2 curves for daylight 
        phase calculation (linearly interpolated to 1 nm).
        
 :_CRI_REF_TYPES: Dict with blackbody to daylight transition (mixing) ranges for
                 various types of reference illuminants used in color rendering
                 index calculations.
        
 :blackbody(): Calculate blackbody radiator spectrum.
 
 :_DAYLIGHT_LOCI_PARAMETERS: dict with parameters for daylight loci for various CMF sets; used by daylightlocus().
 
 :_DAYLIGHT_M12_COEFFS: dict with coefficients in weights M1 & M2 for daylight phases for various CMF sets.
 
 :get_daylightloci_parameters(): Get parameters for the daylight loci functions xD(1000/CCT) and yD(xD); used by daylightlocus().

 :get_daylightphase_Mi_coeffs(): Get coefficients of Mi weights of daylight phase for specific cieobs following Judd et al. (1964).

 :_get_daylightphase_Mi_values(): Get daylight phase coefficients M1, M2 following Judd et al. (1964).         
             
 :daylightlocus(): Calculates daylight chromaticity from cct. 

 :daylightphase(): Calculate daylight phase spectrum.
         
 :cri_ref(): Calculates a reference illuminant spectrum based on cct for color 
             rendering index calculations.
            (`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_, 
             `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9. <http://www.cie.co.at/index.php?i_ca_id=1027>`_,
             `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. New York, NY: The Illuminating Engineering Society of North America. <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
 
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

    4. `CIE 191:2010 Recommended System for Mesopic Photometry Based on Visual Performance.
    (ISBN 978-3-901906-88-6), http://cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance>`_

    5. Judd, D. B., MacAdam, D. L., Wyszecki, G., Budde, H. W., Condit, H. R., Henderson, S. T., & Simonds, J. L. (1964). Spectral Distribution of Typical Daylight as a Function of Correlated Color Temperature. J. Opt. Soc. Am., 54(8), 1031–1040. https://doi.org/10.1364/JOSA.54.001031


Created on Wed Jul  8 22:05:17 2020

@author: ksmet1977@gmail.com
"""
import numpy as np 

from luxpy import _CIEOBS
from luxpy.utils import np2d, _EPS
from . import spd, cie_interp, spd_to_xyz, xyzbar, getwlr, getwld, _WL3, _BB
from . import _CMF, _CIE_GLASS_ID

__all__ = ['_BB','_S012_DAYLIGHTPHASE',
           '_CRI_REF_TYPE', '_CRI_REF_TYPES','cri_ref',
           'blackbody','spd_to_indoor',
           'daylightlocus','daylightphase',
           'get_daylightloci_parameters','get_daylightphase_Mi_coeffs',
           '_DAYLIGHT_LOCI_PARAMETERS','_DAYLIGHT_M12_COEFFS']

#--------------------------------------------------------------------------------------------------
# set some colorimetric constants related to illuminants
#_BB = {'c1' : 3.74183e-16, 'c2' : 1.4388*0.01,'n': 1.000, 'na': 1.00028, 'c' : 299792458, 'h' : 6.626070040e-34, 'k' : 1.38064852e-23} # blackbody c1,c2 & n standard values
# IMPORTED FROM spectral.py AS THIS IS ALSO NEEDED in spd_to_power() !!! 

# Daylight component spectra S0,S1,S2 (linearly interpolated to 1 nm)
_S012_DAYLIGHTPHASE=np.array([[360.000,361.000,362.000,363.000,364.000,365.000,366.000,367.000,368.000,369.000,370.000,371.000,372.000,373.000,374.000,375.000,376.000,377.000,378.000,379.000,380.000,381.000,382.000,383.000,384.000,385.000,386.000,387.000,388.000,389.000,390.000,391.000,392.000,393.000,394.000,395.000,396.000,397.000,398.000,399.000,400.000,401.000,402.000,403.000,404.000,405.000,406.000,407.000,408.000,409.000,410.000,411.000,412.000,413.000,414.000,415.000,416.000,417.000,418.000,419.000,420.000,421.000,422.000,423.000,424.000,425.000,426.000,427.000,428.000,429.000,430.000,431.000,432.000,433.000,434.000,435.000,436.000,437.000,438.000,439.000,440.000,441.000,442.000,443.000,444.000,445.000,446.000,447.000,448.000,449.000,450.000,451.000,452.000,453.000,454.000,455.000,456.000,457.000,458.000,459.000,460.000,461.000,462.000,463.000,464.000,465.000,466.000,467.000,468.000,469.000,470.000,471.000,472.000,473.000,474.000,475.000,476.000,477.000,478.000,479.000,480.000,481.000,482.000,483.000,484.000,485.000,486.000,487.000,488.000,489.000,490.000,491.000,492.000,493.000,494.000,495.000,496.000,497.000,498.000,499.000,500.000,501.000,502.000,503.000,504.000,505.000,506.000,507.000,508.000,509.000,510.000,511.000,512.000,513.000,514.000,515.000,516.000,517.000,518.000,519.000,520.000,521.000,522.000,523.000,524.000,525.000,526.000,527.000,528.000,529.000,530.000,531.000,532.000,533.000,534.000,535.000,536.000,537.000,538.000,539.000,540.000,541.000,542.000,543.000,544.000,545.000,546.000,547.000,548.000,549.000,550.000,551.000,552.000,553.000,554.000,555.000,556.000,557.000,558.000,559.000,560.000,561.000,562.000,563.000,564.000,565.000,566.000,567.000,568.000,569.000,570.000,571.000,572.000,573.000,574.000,575.000,576.000,577.000,578.000,579.000,580.000,581.000,582.000,583.000,584.000,585.000,586.000,587.000,588.000,589.000,590.000,591.000,592.000,593.000,594.000,595.000,596.000,597.000,598.000,599.000,600.000,601.000,602.000,603.000,604.000,605.000,606.000,607.000,608.000,609.000,610.000,611.000,612.000,613.000,614.000,615.000,616.000,617.000,618.000,619.000,620.000,621.000,622.000,623.000,624.000,625.000,626.000,627.000,628.000,629.000,630.000,631.000,632.000,633.000,634.000,635.000,636.000,637.000,638.000,639.000,640.000,641.000,642.000,643.000,644.000,645.000,646.000,647.000,648.000,649.000,650.000,651.000,652.000,653.000,654.000,655.000,656.000,657.000,658.000,659.000,660.000,661.000,662.000,663.000,664.000,665.000,666.000,667.000,668.000,669.000,670.000,671.000,672.000,673.000,674.000,675.000,676.000,677.000,678.000,679.000,680.000,681.000,682.000,683.000,684.000,685.000,686.000,687.000,688.000,689.000,690.000,691.000,692.000,693.000,694.000,695.000,696.000,697.000,698.000,699.000,700.000,701.000,702.000,703.000,704.000,705.000,706.000,707.000,708.000,709.000,710.000,711.000,712.000,713.000,714.000,715.000,716.000,717.000,718.000,719.000,720.000,721.000,722.000,723.000,724.000,725.000,726.000,727.000,728.000,729.000,730.000,731.000,732.000,733.000,734.000,735.000,736.000,737.000,738.000,739.000,740.000,741.000,742.000,743.000,744.000,745.000,746.000,747.000,748.000,749.000,750.000,751.000,752.000,753.000,754.000,755.000,756.000,757.000,758.000,759.000,760.000,761.000,762.000,763.000,764.000,765.000,766.000,767.000,768.000,769.000,770.000,771.000,772.000,773.000,774.000,775.000,776.000,777.000,778.000,779.000,780.000,781.000,782.000,783.000,784.000,785.000,786.000,787.000,788.000,789.000,790.000,791.000,792.000,793.000,794.000,795.000,796.000,797.000,798.000,799.000,800.000,801.000,802.000,803.000,804.000,805.000,806.000,807.000,808.000,809.000,810.000,811.000,812.000,813.000,814.000,815.000,816.000,817.000,818.000,819.000,820.000,821.000,822.000,823.000,824.000,825.000,826.000,827.000,828.000,829.000,830.000],
[61.500,62.230,62.960,63.690,64.420,65.150,65.880,66.610,67.340,68.070,68.800,68.260,67.720,67.180,66.640,66.100,65.560,65.020,64.480,63.940,63.400,63.640,63.880,64.120,64.360,64.600,64.840,65.080,65.320,65.560,65.800,68.700,71.600,74.500,77.400,80.300,83.200,86.100,89.000,91.900,94.800,95.800,96.800,97.800,98.800,99.800,100.800,101.800,102.800,103.800,104.800,104.910,105.020,105.130,105.240,105.350,105.460,105.570,105.680,105.790,105.900,104.990,104.080,103.170,102.260,101.350,100.440,99.530,98.620,97.710,96.800,98.510,100.220,101.930,103.640,105.350,107.060,108.770,110.480,112.190,113.900,115.070,116.240,117.410,118.580,119.750,120.920,122.090,123.260,124.430,125.600,125.590,125.580,125.570,125.560,125.550,125.540,125.530,125.520,125.510,125.500,125.080,124.660,124.240,123.820,123.400,122.980,122.560,122.140,121.720,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,121.300,120.520,119.740,118.960,118.180,117.400,116.620,115.840,115.060,114.280,113.500,113.460,113.420,113.380,113.340,113.300,113.260,113.220,113.180,113.140,113.100,112.870,112.640,112.410,112.180,111.950,111.720,111.490,111.260,111.030,110.800,110.370,109.940,109.510,109.080,108.650,108.220,107.790,107.360,106.930,106.500,106.730,106.960,107.190,107.420,107.650,107.880,108.110,108.340,108.570,108.800,108.450,108.100,107.750,107.400,107.050,106.700,106.350,106.000,105.650,105.300,105.210,105.120,105.030,104.940,104.850,104.760,104.670,104.580,104.490,104.400,103.960,103.520,103.080,102.640,102.200,101.760,101.320,100.880,100.440,100.000,99.600,99.200,98.800,98.400,98.000,97.600,97.200,96.800,96.400,96.000,95.910,95.820,95.730,95.640,95.550,95.460,95.370,95.280,95.190,95.100,94.500,93.900,93.300,92.700,92.100,91.500,90.900,90.300,89.700,89.100,89.240,89.380,89.520,89.660,89.800,89.940,90.080,90.220,90.360,90.500,90.480,90.460,90.440,90.420,90.400,90.380,90.360,90.340,90.320,90.300,90.110,89.920,89.730,89.540,89.350,89.160,88.970,88.780,88.590,88.400,87.960,87.520,87.080,86.640,86.200,85.760,85.320,84.880,84.440,84.000,84.110,84.220,84.330,84.440,84.550,84.660,84.770,84.880,84.990,85.100,84.780,84.460,84.140,83.820,83.500,83.180,82.860,82.540,82.220,81.900,81.970,82.040,82.110,82.180,82.250,82.320,82.390,82.460,82.530,82.600,82.830,83.060,83.290,83.520,83.750,83.980,84.210,84.440,84.670,84.900,84.540,84.180,83.820,83.460,83.100,82.740,82.380,82.020,81.660,81.300,80.360,79.420,78.480,77.540,76.600,75.660,74.720,73.780,72.840,71.900,72.140,72.380,72.620,72.860,73.100,73.340,73.580,73.820,74.060,74.300,74.510,74.720,74.930,75.140,75.350,75.560,75.770,75.980,76.190,76.400,75.090,73.780,72.470,71.160,69.850,68.540,67.230,65.920,64.610,63.300,64.140,64.980,65.820,66.660,67.500,68.340,69.180,70.020,70.860,71.700,72.230,72.760,73.290,73.820,74.350,74.880,75.410,75.940,76.470,77.000,75.820,74.640,73.460,72.280,71.100,69.920,68.740,67.560,66.380,65.200,63.450,61.700,59.950,58.200,56.450,54.700,52.950,51.200,49.450,47.700,49.790,51.880,53.970,56.060,58.150,60.240,62.330,64.420,66.510,68.600,68.240,67.880,67.520,67.160,66.800,66.440,66.080,65.720,65.360,65.000,65.100,65.200,65.300,65.400,65.500,65.600,65.700,65.800,65.900,66.000,65.500,65.000,64.500,64.000,63.500,63.000,62.500,62.000,61.500,61.000,60.230,59.460,58.690,57.920,57.150,56.380,55.610,54.840,54.070,53.300,53.860,54.420,54.980,55.540,56.100,56.660,57.220,57.780,58.340,58.900,59.200,59.500,59.800,60.100,60.400,60.700,61.000,61.300,61.600,61.900],
[38.000,38.440,38.880,39.320,39.760,40.200,40.640,41.080,41.520,41.960,42.400,42.010,41.620,41.230,40.840,40.450,40.060,39.670,39.280,38.890,38.500,38.150,37.800,37.450,37.100,36.750,36.400,36.050,35.700,35.350,35.000,35.840,36.680,37.520,38.360,39.200,40.040,40.880,41.720,42.560,43.400,43.690,43.980,44.270,44.560,44.850,45.140,45.430,45.720,46.010,46.300,46.060,45.820,45.580,45.340,45.100,44.860,44.620,44.380,44.140,43.900,43.220,42.540,41.860,41.180,40.500,39.820,39.140,38.460,37.780,37.100,37.060,37.020,36.980,36.940,36.900,36.860,36.820,36.780,36.740,36.700,36.620,36.540,36.460,36.380,36.300,36.220,36.140,36.060,35.980,35.900,35.570,35.240,34.910,34.580,34.250,33.920,33.590,33.260,32.930,32.600,32.130,31.660,31.190,30.720,30.250,29.780,29.310,28.840,28.370,27.900,27.540,27.180,26.820,26.460,26.100,25.740,25.380,25.020,24.660,24.300,23.880,23.460,23.040,22.620,22.200,21.780,21.360,20.940,20.520,20.100,19.710,19.320,18.930,18.540,18.150,17.760,17.370,16.980,16.590,16.200,15.900,15.600,15.300,15.000,14.700,14.400,14.100,13.800,13.500,13.200,12.740,12.280,11.820,11.360,10.900,10.440,9.980,9.520,9.060,8.600,8.350,8.100,7.850,7.600,7.350,7.100,6.850,6.600,6.350,6.100,5.910,5.720,5.530,5.340,5.150,4.960,4.770,4.580,4.390,4.200,3.970,3.740,3.510,3.280,3.050,2.820,2.590,2.360,2.130,1.900,1.710,1.520,1.330,1.140,0.950,0.760,0.570,0.380,0.190,0.000,-0.160,-0.320,-0.480,-0.640,-0.800,-0.960,-1.120,-1.280,-1.440,-1.600,-1.790,-1.980,-2.170,-2.360,-2.550,-2.740,-2.930,-3.120,-3.310,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.500,-3.730,-3.960,-4.190,-4.420,-4.650,-4.880,-5.110,-5.340,-5.570,-5.800,-5.940,-6.080,-6.220,-6.360,-6.500,-6.640,-6.780,-6.920,-7.060,-7.200,-7.340,-7.480,-7.620,-7.760,-7.900,-8.040,-8.180,-8.320,-8.460,-8.600,-8.690,-8.780,-8.870,-8.960,-9.050,-9.140,-9.230,-9.320,-9.410,-9.500,-9.640,-9.780,-9.920,-10.060,-10.200,-10.340,-10.480,-10.620,-10.760,-10.900,-10.880,-10.860,-10.840,-10.820,-10.800,-10.780,-10.760,-10.740,-10.720,-10.700,-10.830,-10.960,-11.090,-11.220,-11.350,-11.480,-11.610,-11.740,-11.870,-12.000,-12.200,-12.400,-12.600,-12.800,-13.000,-13.200,-13.400,-13.600,-13.800,-14.000,-13.960,-13.920,-13.880,-13.840,-13.800,-13.760,-13.720,-13.680,-13.640,-13.600,-13.440,-13.280,-13.120,-12.960,-12.800,-12.640,-12.480,-12.320,-12.160,-12.000,-12.130,-12.260,-12.390,-12.520,-12.650,-12.780,-12.910,-13.040,-13.170,-13.300,-13.260,-13.220,-13.180,-13.140,-13.100,-13.060,-13.020,-12.980,-12.940,-12.900,-12.670,-12.440,-12.210,-11.980,-11.750,-11.520,-11.290,-11.060,-10.830,-10.600,-10.700,-10.800,-10.900,-11.000,-11.100,-11.200,-11.300,-11.400,-11.500,-11.600,-11.660,-11.720,-11.780,-11.840,-11.900,-11.960,-12.020,-12.080,-12.140,-12.200,-12.000,-11.800,-11.600,-11.400,-11.200,-11.000,-10.800,-10.600,-10.400,-10.200,-9.960,-9.720,-9.480,-9.240,-9.000,-8.760,-8.520,-8.280,-8.040,-7.800,-8.140,-8.480,-8.820,-9.160,-9.500,-9.840,-10.180,-10.520,-10.860,-11.200,-11.120,-11.040,-10.960,-10.880,-10.800,-10.720,-10.640,-10.560,-10.480,-10.400,-10.420,-10.440,-10.460,-10.480,-10.500,-10.520,-10.540,-10.560,-10.580,-10.600,-10.510,-10.420,-10.330,-10.240,-10.150,-10.060,-9.970,-9.880,-9.790,-9.700,-9.560,-9.420,-9.280,-9.140,-9.000,-8.860,-8.720,-8.580,-8.440,-8.300,-8.400,-8.500,-8.600,-8.700,-8.800,-8.900,-9.000,-9.100,-9.200,-9.300,-9.350,-9.400,-9.450,-9.500,-9.550,-9.600,-9.650,-9.700,-9.750,-9.800],
[5.300,5.380,5.460,5.540,5.620,5.700,5.780,5.860,5.940,6.020,6.100,5.790,5.480,5.170,4.860,4.550,4.240,3.930,3.620,3.310,3.000,2.820,2.640,2.460,2.280,2.100,1.920,1.740,1.560,1.380,1.200,0.970,0.740,0.510,0.280,0.050,-0.180,-0.410,-0.640,-0.870,-1.100,-1.040,-0.980,-0.920,-0.860,-0.800,-0.740,-0.680,-0.620,-0.560,-0.500,-0.520,-0.540,-0.560,-0.580,-0.600,-0.620,-0.640,-0.660,-0.680,-0.700,-0.750,-0.800,-0.850,-0.900,-0.950,-1.000,-1.050,-1.100,-1.150,-1.200,-1.340,-1.480,-1.620,-1.760,-1.900,-2.040,-2.180,-2.320,-2.460,-2.600,-2.630,-2.660,-2.690,-2.720,-2.750,-2.780,-2.810,-2.840,-2.870,-2.900,-2.890,-2.880,-2.870,-2.860,-2.850,-2.840,-2.830,-2.820,-2.810,-2.800,-2.780,-2.760,-2.740,-2.720,-2.700,-2.680,-2.660,-2.640,-2.620,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.600,-2.520,-2.440,-2.360,-2.280,-2.200,-2.120,-2.040,-1.960,-1.880,-1.800,-1.770,-1.740,-1.710,-1.680,-1.650,-1.620,-1.590,-1.560,-1.530,-1.500,-1.480,-1.460,-1.440,-1.420,-1.400,-1.380,-1.360,-1.340,-1.320,-1.300,-1.290,-1.280,-1.270,-1.260,-1.250,-1.240,-1.230,-1.220,-1.210,-1.200,-1.180,-1.160,-1.140,-1.120,-1.100,-1.080,-1.060,-1.040,-1.020,-1.000,-0.950,-0.900,-0.850,-0.800,-0.750,-0.700,-0.650,-0.600,-0.550,-0.500,-0.480,-0.460,-0.440,-0.420,-0.400,-0.380,-0.360,-0.340,-0.320,-0.300,-0.270,-0.240,-0.210,-0.180,-0.150,-0.120,-0.090,-0.060,-0.030,0.000,0.020,0.040,0.060,0.080,0.100,0.120,0.140,0.160,0.180,0.200,0.230,0.260,0.290,0.320,0.350,0.380,0.410,0.440,0.470,0.500,0.660,0.820,0.980,1.140,1.300,1.460,1.620,1.780,1.940,2.100,2.210,2.320,2.430,2.540,2.650,2.760,2.870,2.980,3.090,3.200,3.290,3.380,3.470,3.560,3.650,3.740,3.830,3.920,4.010,4.100,4.160,4.220,4.280,4.340,4.400,4.460,4.520,4.580,4.640,4.700,4.740,4.780,4.820,4.860,4.900,4.940,4.980,5.020,5.060,5.100,5.260,5.420,5.580,5.740,5.900,6.060,6.220,6.380,6.540,6.700,6.760,6.820,6.880,6.940,7.000,7.060,7.120,7.180,7.240,7.300,7.430,7.560,7.690,7.820,7.950,8.080,8.210,8.340,8.470,8.600,8.720,8.840,8.960,9.080,9.200,9.320,9.440,9.560,9.680,9.800,9.840,9.880,9.920,9.960,10.000,10.040,10.080,10.120,10.160,10.200,10.010,9.820,9.630,9.440,9.250,9.060,8.870,8.680,8.490,8.300,8.430,8.560,8.690,8.820,8.950,9.080,9.210,9.340,9.470,9.600,9.490,9.380,9.270,9.160,9.050,8.940,8.830,8.720,8.610,8.500,8.350,8.200,8.050,7.900,7.750,7.600,7.450,7.300,7.150,7.000,7.060,7.120,7.180,7.240,7.300,7.360,7.420,7.480,7.540,7.600,7.640,7.680,7.720,7.760,7.800,7.840,7.880,7.920,7.960,8.000,7.870,7.740,7.610,7.480,7.350,7.220,7.090,6.960,6.830,6.700,6.550,6.400,6.250,6.100,5.950,5.800,5.650,5.500,5.350,5.200,5.420,5.640,5.860,6.080,6.300,6.520,6.740,6.960,7.180,7.400,7.340,7.280,7.220,7.160,7.100,7.040,6.980,6.920,6.860,6.800,6.820,6.840,6.860,6.880,6.900,6.920,6.940,6.960,6.980,7.000,6.940,6.880,6.820,6.760,6.700,6.640,6.580,6.520,6.460,6.400,6.310,6.220,6.130,6.040,5.950,5.860,5.770,5.680,5.590,5.500,5.560,5.620,5.680,5.740,5.800,5.860,5.920,5.980,6.040,6.100,6.140,6.180,6.220,6.260,6.300,6.340,6.380,6.420,6.460,6.500]
])  

#--------------------------------------------------------------------------------------------------
# reference illuminant default and mixing range settings, and cieobs, cieobs_Y_normalization settings:
_CRI_REF_TYPE = 'ciera'
_CRI_REF_TYPES = {'ciera': {'mix_range' : [5000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : None}, 
                  'cierf': {'mix_range' : [4000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : None}, 
                  'cierf-224-2017': {'mix_range' : [4000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : None},
                  'iesrf': {'mix_range' : [4000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : '1964_10'},
                  'iesrf-tm30-15': {'mix_range' : [4500.0 , 5500.0], 'cieobs' : None, 'cieobs_Y_normalization' : None},
                  'iesrf-tm30-18': {'mix_range' : [4000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : '1964_10'},
                  'iesrf-tm30-20': {'mix_range' : [4000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : '1964_10'},
                  'BB':{'mix_range' : [5000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : None},
                  'DL':{'mix_range' : [5000.0 , 5000.0], 'cieobs' : None, 'cieobs_Y_normalization' : None}
                  } #mixing ranges, cieobs (for DL), cieobs_Y_normalization (for normalization of mixed illuminants) for various cri_reference_illuminant types

_DAYLIGHT_LOCI_PARAMETERS = None # temporary initialization

#------------------------------------------------------------------------------
#---CIE illuminant functions---------------------------------------------------
#------------------------------------------------------------------------------

def blackbody(cct, wl3 = None, n = None, relative = True):
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
        :n:
            | None, optional
            | Refractive index.
            | If None: use the one stored in _BB['n']
        :relative:
            | False, optional
            | True: return relative spectrum normalized to 560 nm
            | False: return absolute spectral radiance (Planck's law; W/(sr.m².nm)) 
            

    Returns:
        :returns:
            | ndarray with blackbody radiator spectrum
            | (:returns:[0] contains wavelengths)
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    cct = float(cct)
    if wl3 is None: wl3 = _WL3 
    if n is None: n = _BB['n']
    wl = getwlr(wl3)
    def fSr(x):
        return (1/np.pi)*_BB['c1']*((x*1.0e-9)**(-5))*(n**(-2.0))*(np.exp(_BB['c2']*((n*x*1.0e-9*(cct+_EPS))**(-1.0)))-1.0)**(-1.0)
    if relative:
        return np.vstack((wl,(fSr(wl)/fSr(560.0))))
    else:
        return np.vstack((wl,fSr(wl)))

#------------------------------------------------------------------------------
def _get_daylightlocus_parameters(ccts, spds, cieobs):
    """
    Get daylight locus parameters for a single cieobs from daylight phase spectra
    determined based on parameters for '1931_2' as reported in CIE15-20xx.
    """
    
    #------------------------------------------------
    # Get locus parameters:
    #======================
    # get xy coordinates for new cieobs:
    xyz_ = spd_to_xyz(spds, cieobs = cieobs)
    xy = xyz_[...,:2]/xyz_.sum(axis=-1,keepdims=True)
    
    # Fit 3e order polynomal xD(1/T) [4000 K < T <= 7000 K]:
    l7 = ccts<7000
    pxT_l7 = np.polyfit((1000/ccts[l7]), xy[l7,0],3)  
    
    # Fit 3e order polynomal xD(1/T) [T > 7000 K]:
    L7 = ccts>=7000
    pxT_L7 = np.polyfit((1000/ccts[L7]), xy[L7,0],3)  
    
    # Fit 2nd order polynomal yD(xD):
    pxy = np.round(np.polyfit(xy[:,0],xy[:,1],2),3)
    #pxy = np.hstack((0,pxy)) # make also 3e order for easy stacking
        
    return (xy, pxy, pxT_l7, pxT_L7, l7, L7)


#------------------------------------------------------------------------------
def daylightlocus(cct, force_daylight_below4000K = False, cieobs = None, daylight_locus = None):
    """ 
    Calculates daylight chromaticity (xD,yD) from correlated color temperature (cct).
    
    Args:
        :cct: 
            | int or float or list of int/floats or ndarray
        :force_daylight_below4000K: 
            | False or True, optional
            | Daylight locus approximation is not defined below 4000 K, 
            | but by setting this to True, the calculation can be forced to 
            | calculate it anyway.
        :cieobs:
            | CMF set corresponding to xD, yD output.
            | If None: use default CIE15-20xx locus for '1931_2'
            | Else: use the locus specified in :daylight_locus:
        :daylight_locus:
            | None, optional
            | dict with xD(T) and yD(xD) parameters to calculate daylight locus 
            | for specified cieobs.
            | If None: use pre-calculated values.
            | If 'calc': calculate them on the fly.
    
    Returns:
        :(xD, yD): 
            | (ndarray of x-coordinates, ndarray of y-coordinates)
        
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    cct = np2d(cct)
    if np.any((cct < 4000.0) & (force_daylight_below4000K == False)):
        raise Exception('spectral.daylightlocus(): Daylight locus approximation not defined below 4000 K')
    
    if (cieobs is None): # use default values for '1931_2' reported in CIE15-20xx
        xD = -4.607*((1e3/cct)**3.0)+2.9678*((1e3/cct)**2.0)+0.09911*(1000.0/cct)+0.244063
        p = cct>=7000.0
        xD[p] = -2.0064*((1.0e3/cct[p])**3.0)+1.9018*((1.0e3/cct[p])**2.0)+0.24748*(1.0e3/cct[p])+0.23704
        yD = -3.0*xD**2.0+2.87*xD-0.275
    else:
        if isinstance(cieobs, str):
            if daylight_locus is None:
                daylight_locus = _DAYLIGHT_LOCI_PARAMETERS[cieobs]
            else:
                if isinstance(daylight_locus,str):
                    if daylight_locus == 'calc':
                        daylight_locus = get_daylightloci_parameters(cieobs = [cieobs])[cieobs]
        else:
            daylight_locus = get_daylightloci_parameters(cieobs = cieobs)['cmf_0']
        pxy, pxT_l7, pxT_L7 = daylight_locus['pxy'], daylight_locus['pxT_l7k'], daylight_locus['pxT_L7k']
        xD = np.polyval(pxT_l7, 1000/cct)
        p = cct>=7000.0
        xD[p] = np.polyval(pxT_L7, 1000/cct[p])
        yD = np.polyval(pxy, xD)        
        
    return xD,yD

def get_daylightphase_Mi_coeffs(cieobs = None, wl3 = None, S012_daylightphase = None):
    """
    Get coefficients of Mi weights of daylight phase for specific cieobs
    
    Args:
        :cieobs:
            | None or str or ndarray or list of str or list of ndarrays, optional
            | CMF set to get coefficients for.
            | If None: get coeffs for all CMFs in _CMF
        :wl3:
            | None, optional
            | Wavelength range to interpolate S012_daylightphase to.
        :S012_daylightphase:
            | None, optional
            | Daylight phase component functions.
            | If None: use _S012_DAYLIGHTPHASE
    
    Returns:
        :Mcoeffs:
            | Dictionary with i,j,k,i1,j1,k1,i2,j2,k2 for each cieobs in :cieobs:
            | If cieobs contains ndarrays, then keys in dict will be 
            | labeled 'cmf_0', 'cmf_1', ...
    """
    #-------------------------------------------------
    # Get Mi coefficients:
    #=====================
    # Get tristimulus values of daylight phase component functions:
    if S012_daylightphase is None:
        S012_daylightphase = _S012_DAYLIGHTPHASE
    if wl3 is not None:
        S012_daylightphase = cie_interp(S012_daylightphase,wl_new = wl3, kind='linear',negative_values_allowed = True)
    
    if cieobs is None: cieobs = _CMF['types']
    if not isinstance(cieobs, list):
        cieobs = [cieobs]
    Mcoeffs = {}
    i = 0
    for cieobs_ in cieobs:
        if isinstance(cieobs_,str):
            if 'scotopic' in cieobs_:
                continue
            if 'std_dev_obs' in cieobs_:
                continue
            key = cieobs_
        else:
            key = 'cmf_{:1.0f}'.format(i)
        
        xyz = spd_to_xyz(S012_daylightphase, cieobs = cieobs_, relative = False, K = 1)
        S = xyz.sum(axis=-1)
    
        # Get coefficients in Mi:
        f = 1000/S[0]**2
        r = 4 # rounding of i,j,k,i1,j1,k1,i2,j2,k2
        c = {'i': np.round(f*(xyz[2,0]*xyz[1,1] - xyz[1,0]*xyz[2,1]),r)}
        c['j'] = np.round(f*(xyz[2,1]*S[1] - xyz[1,1]*S[2]),r)
        c['k'] = np.round(f*(xyz[1,0]*S[2] - xyz[2,0]*S[1]),r)
        c['i1'] = np.round(f*(xyz[0,0]*xyz[2,1] - xyz[2,0]*xyz[0,1]),r)
        c['j1'] = np.round(f*(xyz[0,1]*S[2] - xyz[2,1]*S[0]),r)
        c['k1'] = np.round(f*(xyz[2,0]*S[0] - xyz[0,0]*S[2]),r)
        c['i2'] = np.round(f*(xyz[1,0]*xyz[0,1] - xyz[0,0]*xyz[1,1]),r)
        c['j2'] = np.round(f*(xyz[1,1]*S[0] - xyz[0,1]*S[1]),r)
        c['k2']= np.round(f*(xyz[0,0]*S[1] - xyz[1,0]*S[0]),r)
        Mcoeffs[key] = c
        i+=1
    return Mcoeffs


def _get_daylightphase_Mi_values(xD,yD, Mcoeffs = None, cieobs = None, S012_daylightphase = None):
    """
    Get daylight phase coefficients M1, M2 following Judd et al. (1964)
    
    Args:
        :xD,yD:
            | ndarray of x-coordinates, ndarray of y-coordinates of daylight phase
        :Mcoeffs:
            | Coefficients in M1 & M2 weights for specific cieobs.
            | If None and cieobs is not None: they will be calculated.
        :cieobs:
            | CMF set to use when calculating coefficients in M1, M2 weights.
            | If None: Mcoeffs must be supplied.
        :S012_daylightphase: 
            | ndarray with CIE S0, S1, S2 daylight phase component functions
    
    Returns:
        :M1,M2:
            | daylight phase coefficients M1, M2
        
    Reference:
        1. `Judd et al.(1964). Spectral Distribution of Typical Daylight as a 
        Function of Correlated Color Temperature. 
        JOSA A, 54(8), pp. 1031-1040 (1964) <https://doi.org/10.1364/JOSA.54.001031>`_
    """
    if (Mcoeffs is None) & (cieobs is  None):
        raise Exception("_get_daylightphase_Mi_values(): Mcoeffs and cieobs can't both be None")
    if (Mcoeffs is None) & isinstance(cieobs,str): # use pre-calculated coeffs.
        Mcoeffs = _DAYLIGHT_M12_COEFFS[cieobs]
    if isinstance(Mcoeffs,str) & (cieobs is not None): # calculate coeffs.
        if (Mcoeffs == 'calc'):
            Mcoeffs = get_daylightphase_Mi_coeffs(cieobs = cieobs, S012_daylightphase = S012_daylightphase)
            Mcoeffs = Mcoeffs[cieobs] if isinstance(cieobs,str) else Mcoeffs['cmf_0']
        else:
            raise Exception("Mcoeffs is a string, but not 'calc': unknown option.")
    if Mcoeffs is not None:
        c = Mcoeffs

    # Calculate M1, M2 and round to 3 decimals (CIE recommendation):
    denom = c['i'] + c['j']*xD + c['k']*yD
    rr = 3 # rounding of Mi
    M1 = np.round((c['i1'] + c['j1']*xD + c['k1']*yD) / denom, rr)
    M2 = np.round((c['i2'] + c['j2']*xD + c['k2']*yD) / denom, rr)
    
    # denom = (xyz[2,0]*xyz[1,1] - xyz[1,0]*xyz[2,1]) + (xyz[2,1]*S[1] - xyz[1,1]*S[2])*xD + (xyz[1,0]*S[2] - xyz[2,0]*S[1])*yD
    # M1 = np.round(((xyz[0,0]*xyz[2,1] - xyz[2,0]*xyz[0,1]) + (xyz[0,1]*S[2] - xyz[2,1]*S[0])*xD + (xyz[2,0]*S[0] - xyz[0,0]*S[2])*yD) / denom, 3)
    # M2 = np.round(((xyz[1,0]*xyz[0,1] - xyz[0,0]*xyz[1,1]) + (xyz[1,1]*S[0] - xyz[0,1]*S[1])*xD + (xyz[0,0]*S[1] - xyz[1,0]*S[0])*yD) / denom, 3)
              
    return M1, M2, c 
    
    
   
#------------------------------------------------------------------------------
def daylightphase(cct, wl3 = None, nominal_cct = False, force_daylight_below4000K = False, verbosity = None, 
                  n = None, cieobs = None, daylight_locus = None, daylight_Mi_coeffs = None):
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
        :nominal_cct:
            | False, optional
            | If cct is nominal (e.g. when calculating D65): multiply cct first
            | by 1.4388/1.4380 to account for change in 'c2' in definition of Planckian.
        :cieobs:
            | None or str or ndarray, optional
            | CMF set to use when calculating coefficients for daylight locus and for M1, M2 weights.
            | If None: use standard coefficients for CIE 1931 2° CMFs (for Si at 10 nm).
            | Else: calculate coefficients following Appendix C of CIE15-2004 and Judd (1964).
        :force_daylight_below4000K: 
            | False or True, optional
            | Daylight locus approximation is not defined below 4000 K, 
            | but by setting this to True, the calculation can be forced to 
            | calculate it anyway.
        :verbosity: 
            | None, optional
            |   If None: do not print warning when CCT < 4000 K.
        :n:
            | None, optional
            | Refractive index (for use in calculation of blackbody radiators).
            | If None: use the one stored in _BB['n']
        :daylight_locus:
            | None, optional
            | dict with xD(T) and yD(xD) parameters to calculate daylight locus 
            | for specified cieobs.
            | If None: use pre-calculated values.
            | If 'calc': calculate them on the fly.
        :daylight_Mi_coeffs:
            | None, optional
            | dict with coefficients for M1 & M2 weights for specified cieobs.
            | If None: use pre-calculated values.
            | If 'calc': calculate them on the fly.

    Returns:
        :returns: 
            | ndarray with daylight phase spectrum
            | (:returns:[0] contains wavelengths)

    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
        
        2. `Judd, MacAdam, Wyszecki, Budde, Condit, Henderson, & Simonds (1964). 
        Spectral Distribution of Typical Daylight as a Function of Correlated Color Temperature. 
        J. Opt. Soc. Am., 54(8), 1031–1040. 
        <https://doi.org/10.1364/JOSA.54.001031>`_
    """
    cct = float(cct)
    if wl3 is None: wl3 = _WL3 
    if (cct < (4000.0)) & (force_daylight_below4000K == False):
        if verbosity is not None:
            print('Warning daylightphase spd not defined below 4000 K. Using blackbody radiator instead.')
        return blackbody(cct,wl3, n = n)
    else:
        if nominal_cct: cct*=(1.4388/1.4380) # account for change in c2 in def. of Planckian

        wl = getwlr(wl3) 
        
        #interpolate _S012_DAYLIGHTPHASE first to wl range:
        if  not np.array_equal(_S012_DAYLIGHTPHASE[0],wl):
            S012_daylightphase = cie_interp(data = _S012_DAYLIGHTPHASE, wl_new = wl, kind = 'linear',negative_values_allowed = True)
        else:
            S012_daylightphase = _S012_DAYLIGHTPHASE

        # Get coordinates of daylight locus corresponding to cct:
        xD, yD = daylightlocus(cct, force_daylight_below4000K = force_daylight_below4000K, cieobs = cieobs, daylight_locus = daylight_locus)
        
        # Get M1 & M2 component weights:
        if (cieobs is None): # original M1,M2 for Si at 10 nm spacing and CIE 1931 xy
            Mcoeffs = {'i':0.0241,'j':0.2562,'k':-0.7341,
            'i1':-1.3515,'j1':-1.7703,'k1':5.9114,
            'i2':0.0300,'j2':-31.4424,'k2':30.0717}
        else:
            Mcoeffs = daylight_Mi_coeffs
        M1, M2, _ = _get_daylightphase_Mi_values(xD, yD, Mcoeffs = Mcoeffs, cieobs = cieobs, S012_daylightphase = S012_daylightphase) 
        
        # Calculate weigthed combination of S0, S1 & S2 components:
        Sr = S012_daylightphase[1,:] + M1*S012_daylightphase[2,:] + M2*S012_daylightphase[3,:]
        
        # Normalize to 1 at (or near) 560 nm:
        Sr560 = Sr[:,np.where(np.abs(S012_daylightphase[0,:] - 560.0) == np.min(np.abs(S012_daylightphase[0,:] - 560)))[0]]
        Sr /= Sr560
        Sr[Sr==float('NaN')] = 0
        return np.vstack((wl,Sr))

def get_daylightloci_parameters(ccts = None, cieobs = None, wl3 = [300,830,10], verbosity = 0):
    """
    Get parameters for the daylight loci functions xD(1000/CCT) and yD(xD).
    
    Args:
        :ccts:
            | None, optional
            | ndarray with CCTs, if None: ccts = np.arange(4000,25000,250)
        :cieobs:
            | None or list of str or list of ndarrays, optional
            | CMF sets to determine parameters for.
            | If None: get for all CMFs sets in _CMF (except scoptopic and deviate observer)
        :wl3:
            | [300,830,10], optional
            | Wavelength range and spacing of daylight phases to be determined
            | from '1931_2'. The default setting results in parameters very close
            | to that in CIE15-2004/2018.
        :verbosity:
            | 0, optional
            | print parameters and make plots.
            
    Returns:
        :dayloci:
            | dict with parameters for each cieobs
            | If cieobs contains ndarrays, then keys in dict will be 
            | labeled 'cmf_0', 'cmf_1', ...
    """
    if ccts is None:
        ccts = np.arange(4000,25000,250)
        
    # Get daylight phase spds using cieobs '1931_2':
    # wl3 = [300,830,10] # results in Judd's (1964) coefficients for the function yD(xD)x; other show slight deviations
    for i, cct  in enumerate(ccts):
        spd = daylightphase(cct, cieobs = None, wl3 = wl3, force_daylight_below4000K = False)
        if i == 0:
            spds = spd
        else:
            spds = np.vstack((spds,spd[1:]))
            
    if verbosity > 0:
        import matplotlib.pyplot as plt # lazy import
        fig,axs = plt.subplots(nrows = 2, ncols = len(_CMF['types']) - 2)   # -2: don't include scoptopic and dev observers     

    dayloci = {'WL':wl3}
    i = 0
    if cieobs is None: cieobs = _CMF['types']
    if not isinstance(cieobs, list):
        cieobs = [cieobs]
    for cieobs_ in cieobs:
        if isinstance(cieobs_,str):
            if 'scotopic' in cieobs_:
                continue
            if 'std_dev_obs' in cieobs_:
                continue
            key = cieobs_
        else:
            key = 'cmf_{:1.0f}'.format(i)
        
        # get parameters for cieobs:
        xy, pxy, pxT_l7, pxT_L7, l7, L7 = _get_daylightlocus_parameters(ccts, spds, cieobs_)
        dayloci[key] = {'pxT_l7k':pxT_l7, 'pxT_L7k':pxT_L7, 'pxy':pxy}
        
        if verbosity > 0:
            print('\n cieobs:', key)
            print('pxT_l7 (Tcp<7000K):',pxT_l7)
            print('pxT_L7 (Tcp>=7000K):',pxT_L7)
            print('p:xy',pxy)
                  
            axs[0,i].plot(ccts, xy[:,0],'r-', label = 'Data')
            axs[0,i].plot(ccts[l7], np.polyval(pxT_l7,1000/ccts[l7]),'b--', label = 'Fit (Tcp<7000K)')
            axs[0,i].plot(ccts[L7], np.polyval(pxT_L7,1000/ccts[L7]),'c--', label = 'Fit (Tcp>=7000K)')
            axs[0,i].set_title(key)
            axs[0,i].set_xlabel('Tcp (K)')
            axs[0,i].set_ylabel('xD')
            axs[0,i].legend()
            
            #plotSL(cieobs = cieobs_, cspace = 'Yxy', DL = False, axh = axs[1,i])
            axs[1,i].plot(xy[:,0],xy[:,1],'r-', label = 'Data')
            axs[1,i].plot(xy[:,0],np.polyval(pxy,xy[:,0]),'b--', label = 'Fit')
            # axs[1,i].plot(xy[:,0],np.polyval(pxy_31,xy[:,0]),'g:')
            axs[1,i].set_xlabel('xD')
            axs[1,i].set_ylabel('yD')
            axs[1,i].legend()
        
        i+=1
        
    return dayloci
    
#------------------------------------------------------------------------------
# Pre-calculate daylight loci parameters and Mi coefficients for each CMF in _CMF
# (except 'scotopic' and 'cie_std_dev_...'):
wl3 = [360,830,1]
_DAYLIGHT_LOCI_PARAMETERS = get_daylightloci_parameters(ccts = None, cieobs = None, wl3 = wl3, verbosity = 0)
_DAYLIGHT_M12_COEFFS = get_daylightphase_Mi_coeffs(cieobs = None, wl3 = wl3)

#------------------------------------------------------------------------------
def cri_ref(ccts, wl3 = None, ref_type = _CRI_REF_TYPE, mix_range = None, 
            cieobs = None, cieobs_Y_normalization = None, 
            norm_type = None, norm_f = None, 
            force_daylight_below4000K = False, n = None,
            daylight_locus = None):
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
            | None, optional
            | Required when calculating daylightphase (adjust locus parameters to cieobs)
            | If None: value in _CRI_REF_TYPES will be used (with None here corresponding to _CIEOBS).
        :cieobs_Y_normalization:
            | None, optional
            | Required for the normalization of the Planckian and Daylight SPDs 
            | when calculating a 'mixed' reference illuminant.
            | If None: value in _CRI_REF_TYPES will be used, 
            |   with None here resulting in the use of the value as specified in :cieobs:
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
        :n:
            | None, optional
            | Refractive index (for use in calculation of blackbody radiators).
            | If None: use the one stored in _BB['n']
        :daylight_locus:
            | None, optional
            | dict with xD(T) and yD(xD) parameters to calculate daylight locus 
            | for specified cieobs.
            | If None: use pre-calculated values.
            | If 'calc': calculate them on the fly.
    
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
        if cieobs is not None: cieobs = np.atleast_1d(cieobs)
        if cieobs_Y_normalization is not None: cieobs_Y_normalization = np.atleast_1d(cieobs_Y_normalization)

        if not (isinstance(ref_type,list) | isinstance(ref_type,dict)): ref_type = [ref_type]
   
        for i in range(len(ccts)):
            cct = ccts[i]

            # get ref_type and mix_range:
            if isinstance(ref_type,dict):
                raise Exception("cri_ref(): dictionary ref_type: Not yet implemented")
            else:

                ref_type_ = ref_type[i] if (len(ref_type)>1) else ref_type[0]

                if mix_range is None:
                    mix_range_ =  _CRI_REF_TYPES[ref_type_]['mix_range']
                else:
                    mix_range_ = mix_range[i] if (mix_range.shape[0]>1) else mix_range[0]  #must be np2d !!!            
      
                if cieobs is None:
                    cieobs_ = _CRI_REF_TYPES[ref_type_]['cieobs']
                else:
                    cieobs_ = cieobs[i] if (cieobs.shape[0]>1) else cieobs[0]
                    
                if cieobs_Y_normalization is None:
                    cieobs_Y_normalization_ = _CRI_REF_TYPES[ref_type_]['cieobs_Y_normalization']
                else:
                    cieobs_Y_normalization_ = cieobs_Y_normalization[i] if (cieobs_Y_normalization.shape[0]>1) else cieobs_Y_normalization[0]
                    
                if cieobs_Y_normalization_ is None: cieobs_Y_normalization_ = cieobs_
                if cieobs_Y_normalization_ is None: cieobs_Y_normalization_ = _CIEOBS # cieobs_Y_normalization_ might still be None as cieobs_ == None results in specific use of fixed published coeff. in the calculation of the daylight phase, while a string will result in calculation of these coeff.
                
            if (mix_range_[0] == mix_range_[1]) | (ref_type_[0:2] == 'BB') | (ref_type_[0:2] == 'DL'):
                if ((cct < mix_range_[0]) & (not (ref_type_[0:2] == 'DL'))) | (ref_type_[0:2] == 'BB'):
                    Sr = blackbody(cct, wl3, n = n)
                elif ((cct >= mix_range_[0]) & (not (ref_type_[0:2] == 'BB'))) | (ref_type_[0:2] == 'DL') :
                    Sr = daylightphase(cct,wl3,force_daylight_below4000K = force_daylight_below4000K, cieobs = cieobs_, daylight_locus = daylight_locus)
            else:
                SrBB = blackbody(cct, wl3, n = n)
                SrDL = daylightphase(cct,wl3,verbosity = None,force_daylight_below4000K = force_daylight_below4000K, cieobs = cieobs_, daylight_locus = daylight_locus)
                
                #cieobs_ = _CIEOBS if cieobs_ is None else cieobs_ # cieobs_ might still be None as that results in specific use of fixed published coeff. in the calculation of the daylight phase, while a string will result in calculation of these coeff.
                
                cmf = xyzbar(cieobs = cieobs_Y_normalization_, scr = 'dict', wl_new = wl3)
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

#------------------------------------------------------------------------------
def spd_to_indoor(spd):
    """
    Convert spd to indoor variant by multiplying it with the CIE spectral transmission for glass.
    """
    Tglass = cie_interp(_CIE_GLASS_ID['T'].copy(), spd[0,:], kind = 'rfl')[1:,:]
    spd_ = spd.copy()
    spd_[1:,:] *= Tglass
    return spd_