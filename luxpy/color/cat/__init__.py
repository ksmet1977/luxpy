# -*- coding: utf-8 -*-
"""
cat: Module supporting chromatic adaptation transforms (corresponding colors)
=============================================================================

 :_WHITE_POINT: default adopted white point

 :_LA:  default luminance of the adaptation field

 :_MCATS: default chromatic adaptation sensor spaces

     * 'hpe': Hunt-Pointer-Estevez: R. W. G. Hunt, The Reproduction of Colour: Sixth Edition, 6th ed. Chichester, UK: John Wiley & Sons Ltd, 2004.
     * 'cat02': from ciecam02: `CIE159-2004, “A Colour Apperance Model for Color Management System: CIECAM02,” CIE, Vienna, 2004. <http://onlinelibrary.wiley.com/doi/10.1002/col.20198/abstract>`_
     * 'cat02-bs':  cat02 adjusted to solve yellow-blue problem (last line = [0 0 1]): `Brill MH, Süsstrunk S. Repairing gamut problems in CIECAM02: A progress report. Color Res Appl 2008;33(5), 424–426. <http://onlinelibrary.wiley.com/doi/10.1002/col.20432/abstract>`_
     * 'cat02-jiang': cat02 modified to solve yb-probem + purple problem: `Jun Jiang, Zhifeng Wang,M. Ronnier Luo,Manuel Melgosa,Michael H. Brill,Changjun Li, Optimum solution of the CIECAM02 yellow–blue and purple problems, Color Res Appl 2015: 40(5), 491-503. <http://onlinelibrary.wiley.com/doi/10.1002/col.21921/abstract>`_
     * 'kries'
     * 'judd-1945': from `CIE16-2004 <http://www.cie.co.at/index.php/index.php?i_ca_id=436>`_, Eq.4, a23 modified from 0.1 to 0.1020 for increased accuracy
     * 'bfd': bradford transform :  `G. D. Finlayson and S. Susstrunk, “Spectral sharpening and the Bradford transform,” 2000, vol. Proceeding, pp. 236–242. <https://infoscience.epfl.ch/record/34077>`_
     * 'sharp': sharp transform:  `S. Süsstrunk, J. Holm, and G. D. Finlayson, “Chromatic adaptation performance of different RGB sensors,” IS&T/SPIE Electronic Imaging 2001: Color Imaging, vol. 4300. San Jose, CA, January, pp. 172–183, 2001. <http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=903890>`_
     * 'cmc':  `C. Li, M. R. Luo, B. Rigg, and R. W. G. Hunt, “CMC 2000 chromatic adaptation transform: CMCCAT2000,” Color Res. Appl., vol. 27, no. 1, pp. 49–58, 2002. <http://onlinelibrary.wiley.com/doi/10.1002/col.10005/abstract>`_
     * 'ipt':  `F. Ebner and M. D. Fairchild, “Development and testing of a color space (IPT) with improved hue uniformity,” in IS&T 6th Color Imaging Conference, 1998, pp. 8–13. <http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true>`_
     * 'lms':
     * 'bianco':  `S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010. <http://onlinelibrary.wiley.com/doi/10.1002/col.20573/full>`_
     * 'bianco-pc':  `S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010. <http://onlinelibrary.wiley.com/doi/10.1002/col.20573/full>`_
     * 'cat16': `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a. <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_

 :check_dimensions(): Check if dimensions of data and xyzw match. 

 :get_transfer_function(): | Calculate the chromatic adaptation diagonal matrix 
                             transfer function Dt.  
                           | Default = 'vonkries' (others: 'rlab', see Fairchild 1990)

 :smet2017_D(): | Calculate the degree of adaptation based on chromaticity. 
                | `Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017),
                  Study of chromatic adaptation using memory color matches, 
                  Part II: colored illuminants.
                  Opt. Express, 25(7), pp. 8350-8365 
                  <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-8350&origin=search>`_

 :get_degree_of_adaptation(): | Calculates the degree of adaptation. 
                              | D passes either right through or D is 
                                calculated following some D-function (Dtype) 
                                published in literature (cat02, cat16, cmccat, 
                                smet2017) or set manually.

 :parse_x1x2_parameters(): local helper function that parses input parameters 
                           and makes them the target_shape for easy calculation 

 :apply(): Calculate corresponding colors by applying a von Kries chromatic 
           adaptation transform (CAT), i.e. independent rescaling of 
           'sensor sensitivity' to data to adapt from current adaptation 
           conditions (1) to the new conditions (2). 
           
===============================================================================
"""
from .chromaticadaptation import *
__all__ = chromaticadaptation.__all__