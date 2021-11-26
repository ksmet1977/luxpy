# -*- coding: utf-8 -*-
"""
CLA2.0 efficiency functions
===========================

These were obtained from https://github.com/Light-and-Health-Research-Center/cscalc/blob/master/json/consts.json
(only the relevant data was kept)

Created on Fri Nov  5 20:08:09 2021

@author: ksmet1977 [a] gmail.com
"""

import numpy as np
from luxpy.spectrum.basics.spectral import cie_interp
import pandas as pd

_FILE_NAME = 'LRC2021_CS_CLa_efficiency_functions.dat'

CLA2d0_data =   {
      "ybar": {
    "wavelength": [
      360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
      375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
      390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
      405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
      420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434,
      435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,
      450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464,
      465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
      480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
      495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
      510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524,
      525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,
      540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554,
      555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
      570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
      585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,
      600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614,
      615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629,
      630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644,
      645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
      660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674,
      675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689,
      690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704,
      705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719,
      720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734,
      735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749,
      750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764,
      765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
      780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794,
      795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809,
      810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824,
      825, 826, 827, 828, 829, 830
    ],
    "value": [
      3.92e-6, 4.39e-6, 4.93e-6, 5.53e-6, 6.21e-6, 6.97e-6, 7.81e-6, 8.77e-6,
      9.84e-6, 1.1e-5, 1.24e-5, 1.39e-5, 1.56e-5, 1.74e-5, 1.96e-5, 2.2e-5,
      2.48e-5, 2.8e-5, 3.15e-5, 3.52e-5, 3.9e-5, 4.28e-5, 4.69e-5, 5.16e-5,
      5.72e-5, 6.4e-5, 7.23e-5, 8.22e-5, 9.35e-5, 0.000106136, 0.00012,
      0.000134984, 0.000151492, 0.000170208, 0.000191816, 0.000217, 0.000246907,
      0.00028124, 0.00031852, 0.000357267, 0.000396, 0.000433715, 0.000473024,
      0.000517876, 0.000572219, 0.00064, 0.00072456, 0.0008255, 0.00094116,
      0.00106988, 0.00121, 0.001362091, 0.001530752, 0.001720368, 0.001935323,
      0.00218, 0.0024548, 0.002764, 0.0031178, 0.0035264, 0.004, 0.00454624,
      0.00515932, 0.00582928, 0.00654616, 0.0073, 0.008086507, 0.00890872,
      0.00976768, 0.01066443, 0.0116, 0.01257317, 0.01358272, 0.01462968,
      0.01571509, 0.01684, 0.01800736, 0.01921448, 0.02045392, 0.02171824, 0.023,
      0.02429461, 0.02561024, 0.02695857, 0.02835125, 0.0298, 0.03131083,
      0.03288368, 0.03452112, 0.03622571, 0.038, 0.03984667, 0.041768, 0.043766,
      0.04584267, 0.048, 0.05024368, 0.05257304, 0.05498056, 0.05745872, 0.06,
      0.06260197, 0.06527752, 0.06804208, 0.07091109, 0.0739, 0.077016, 0.0802664,
      0.0836668, 0.0872328, 0.09098, 0.09491755, 0.09904584, 0.1033674, 0.1078846,
      0.1126, 0.117532, 0.1226744, 0.1279928, 0.1334528, 0.13902, 0.1446764,
      0.1504693, 0.1564619, 0.1627177, 0.1693, 0.1762431, 0.1835581, 0.1912735,
      0.199418, 0.20802, 0.2171199, 0.2267345, 0.2368571, 0.2474812, 0.2586,
      0.2701849, 0.2822939, 0.2950505, 0.308578, 0.323, 0.3384021, 0.3546858,
      0.3716986, 0.3892875, 0.4073, 0.4256299, 0.4443096, 0.4633944, 0.4829395,
      0.503, 0.5235693, 0.544512, 0.56569, 0.5869653, 0.6082, 0.6293456,
      0.6503068, 0.6708752, 0.6908424, 0.71, 0.7281852, 0.7454636, 0.7619694,
      0.7778368, 0.7932, 0.8081104, 0.8224962, 0.8363068, 0.8494916, 0.862,
      0.8738108, 0.8849624, 0.8954936, 0.9054432, 0.9148501, 0.9237348, 0.9320924,
      0.9399226, 0.9472252, 0.954, 0.9602561, 0.9660074, 0.9712606, 0.9760225,
      0.9803, 0.9840924, 0.9874182, 0.9903128, 0.9928116, 0.9949501, 0.9967108,
      0.9980983, 0.999112, 0.9997482, 1, 0.9998567, 0.9993046, 0.9983255,
      0.9968987, 0.995, 0.9926005, 0.9897426, 0.9864444, 0.9827241, 0.9786,
      0.9740837, 0.9691712, 0.9638568, 0.9581349, 0.952, 0.9454504, 0.9384992,
      0.9311628, 0.9234576, 0.9154, 0.9070064, 0.8982772, 0.8892048, 0.8797816,
      0.87, 0.8598613, 0.849392, 0.838622, 0.8275813, 0.8163, 0.8047947, 0.793082,
      0.781192, 0.7691547, 0.757, 0.7447541, 0.7324224, 0.7200036, 0.7074965,
      0.6949, 0.6822192, 0.6694716, 0.6566744, 0.6438448, 0.631, 0.6181555,
      0.6053144, 0.5924756, 0.5796379, 0.5668, 0.5539611, 0.5411372, 0.5283528,
      0.5156323, 0.503, 0.4904688, 0.4780304, 0.4656776, 0.4534032, 0.4412,
      0.42908, 0.417036, 0.405032, 0.393032, 0.381, 0.3689184, 0.3568272,
      0.3447768, 0.3328176, 0.321, 0.3093381, 0.2978504, 0.2865936, 0.2756245,
      0.265, 0.2547632, 0.2448896, 0.2353344, 0.2260528, 0.217, 0.2081616,
      0.1995488, 0.1911552, 0.1829744, 0.175, 0.1672235, 0.1596464, 0.1522776,
      0.1451259, 0.1382, 0.1315003, 0.1250248, 0.1187792, 0.1127691, 0.107,
      0.1014762, 0.09618864, 0.09112296, 0.08626485, 0.0816, 0.07712064,
      0.07282552, 0.06871008, 0.06476976, 0.061, 0.05739621, 0.05395504,
      0.05067376, 0.04754965, 0.04458, 0.04175872, 0.03908496, 0.03656384,
      0.03420048, 0.032, 0.02996261, 0.02807664, 0.02632936, 0.02470805, 0.0232,
      0.02180077, 0.02050112, 0.01928108, 0.01812069, 0.017, 0.01590379,
      0.01483718, 0.01381068, 0.01283478, 0.01192, 0.01106831, 0.01027339,
      0.009533311, 0.008846157, 0.00821, 0.007623781, 0.007085424, 0.006591476,
      0.006138485, 0.005723, 0.005343059, 0.004995796, 0.004676404, 0.004380075,
      0.004102, 0.003838453, 0.003589099, 0.003354219, 0.003134093, 0.002929,
      0.002738139, 0.002559876, 0.002393244, 0.002237275, 0.002091, 0.001953587,
      0.00182458, 0.00170358, 0.001590187, 0.001484, 0.001384496, 0.001291268,
      0.001204092, 0.001122744, 0.001047, 0.00097659, 0.000911109, 0.000850133,
      0.000793238, 0.00074, 0.000690083, 0.00064331, 0.000599496, 0.000558455,
      0.00052, 0.000483914, 0.000450053, 0.000418345, 0.000388718, 0.0003611,
      0.000335384, 0.00031144, 0.000289166, 0.000268454, 0.0002492, 0.000231302,
      0.000214686, 0.000199288, 0.000185048, 0.0001719, 0.000159778, 0.000148604,
      0.000138302, 0.000128793, 0.00012, 0.00011186, 0.000104322, 9.73e-5,
      9.08e-5, 8.48e-5, 7.91e-5, 7.39e-5, 6.89e-5, 6.43e-5, 6.0e-5, 5.6e-5,
      5.22e-5, 4.87e-5, 4.54e-5, 4.24e-5, 3.96e-5, 3.69e-5, 3.44e-5, 3.21e-5,
      3.0e-5, 2.8e-5, 2.61e-5, 2.44e-5, 2.27e-5, 2.12e-5, 1.98e-5, 1.85e-5,
      1.72e-5, 1.61e-5, 1.5e-5, 1.4e-5, 1.31e-5, 1.22e-5, 1.14e-5, 1.06e-5,
      9.89e-6, 9.22e-6, 8.59e-6, 8.01e-6, 7.47e-6, 6.96e-6, 6.49e-6, 6.05e-6,
      5.64e-6, 5.26e-6, 4.9e-6, 4.57e-6, 4.26e-6, 3.97e-6, 3.7e-6, 3.45e-6,
      3.22e-6, 3.0e-6, 2.8e-6, 2.61e-6, 2.43e-6, 2.27e-6, 2.11e-6, 1.97e-6,
      1.84e-6, 1.71e-6, 1.6e-6, 1.49e-6, 1.39e-6, 1.29e-6, 1.21e-6, 1.12e-6,
      1.05e-6, 9.77e-7, 9.11e-7, 8.49e-7, 7.92e-7, 7.38e-7, 6.88e-7, 6.42e-7,
      5.98e-7, 5.58e-7, 5.2e-7, 4.85e-7, 4.52e-7
    ],
  },
   "Vlambda": {
    "wavelength": [
      390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418,
      420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448,
      450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478,
      480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508,
      510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538,
      540, 542, 544, 546, 548, 550, 552, 554, 556, 558, 560, 562, 564, 566, 568,
      570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596, 598,
      600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622, 624, 626, 628,
      630, 632, 634, 636, 638, 640, 642, 644, 646, 648, 650, 652, 654, 656, 658,
      660, 662, 664, 666, 668, 670, 672, 674, 676, 678, 680, 682, 684, 686, 688,
      690, 692, 694, 696, 698, 700, 702, 704, 706, 708, 710, 712, 714, 716, 718,
      720, 722, 724, 726, 728, 730
    ],
    "value": [
      0.000120024, 0.000151522, 0.000191854, 0.000246955, 0.000318583,
      0.000396078, 0.000473117, 0.000572332, 0.000724703, 0.000941346,
      0.001210239, 0.001531054, 0.001935705, 0.002455285, 0.003118416, 0.00400079,
      0.005160339, 0.006547453, 0.008088105, 0.00976961, 0.011602292, 0.013585404,
      0.015718195, 0.018010918, 0.020457961, 0.023004545, 0.0256153, 0.028356852,
      0.031317017, 0.034527941, 0.038007508, 0.041776253, 0.045851728,
      0.050253608, 0.054991424, 0.060011855, 0.065290418, 0.070925101,
      0.077031218, 0.083683332, 0.090997977, 0.09906541, 0.107905917, 0.117555223,
      0.12801809, 0.139047469, 0.150499031, 0.162749851, 0.176277924, 0.191311294,
      0.208061102, 0.2267793, 0.2475301, 0.270238286, 0.295108799, 0.323063821,
      0.354755882, 0.389364419, 0.425714, 0.463485962, 0.503099387, 0.54461959,
      0.587081278, 0.629478954, 0.671007758, 0.710140288, 0.745610895,
      0.777990492, 0.808270074, 0.836472045, 0.862170322, 0.885137259,
      0.905622106, 0.92391732, 0.940108318, 0.9541885, 0.966198272, 0.976215351,
      0.984286846, 0.990508475, 0.995146691, 0.998295513, 0.999945739,
      1.000054261, 0.998522758, 0.995196601, 0.989938162, 0.982918276,
      0.974276168, 0.964047248, 0.952188105, 0.938684637, 0.923640065,
      0.907185615, 0.889380497, 0.870171902, 0.849559831, 0.827744821,
      0.804953719, 0.781346355, 0.757149575, 0.732567119, 0.707636294,
      0.682353999, 0.656804152, 0.631124679, 0.605434003, 0.57975243, 0.554070557,
      0.528457197, 0.503099387, 0.478124854, 0.453492788, 0.429164782, 0.40511203,
      0.381075281, 0.356897705, 0.332883361, 0.309399222, 0.286650228,
      0.265052361, 0.244937987, 0.226097466, 0.20820273, 0.19119297, 0.175034578,
      0.159677944, 0.145154575, 0.131526283, 0.118802669, 0.107021142,
      0.096207646, 0.086281895, 0.077135878, 0.068723656, 0.061012053,
      0.053965701, 0.047559045, 0.041766971, 0.036571065, 0.032006323,
      0.028082188, 0.024712932, 0.021805078, 0.01928489, 0.017003359, 0.014840112,
      0.012837316, 0.011070497, 0.009535195, 0.008211622, 0.007086824,
      0.006139698, 0.005344115, 0.004677328, 0.004102811, 0.003589808,
      0.003134712, 0.00273868, 0.002393717, 0.002091413, 0.001824941, 0.001590501,
      0.00138477, 0.00120433, 0.001047207, 0.000911289, 0.000793395, 0.000690963,
      0.000599614, 0.000520103
    ]
  },

  "Vprime": {
    "wavelength": [
      390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418,
      420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448,
      450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478,
      480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508,
      510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538,
      540, 542, 544, 546, 548, 550, 552, 554, 556, 558, 560, 562, 564, 566, 568,
      570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596, 598,
      600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622, 624, 626, 628,
      630, 632, 634, 636, 638, 640, 642, 644, 646, 648, 650, 652, 654, 656, 658,
      660, 662, 664, 666, 668, 670, 672, 674, 676, 678, 680, 682, 684, 686, 688,
      690, 692, 694, 696, 698, 700, 702, 704, 706, 708, 710, 712, 714, 716, 718,
      720, 722, 724, 726, 728, 730
    ],
    "value": [
      0.002209, 0.002939, 0.003921, 0.00524, 0.00698, 0.00929, 0.01231, 0.01619,
      0.02113, 0.0273, 0.03484, 0.0439, 0.0545, 0.0668, 0.0808, 0.0966, 0.1141,
      0.1334, 0.1541, 0.1764, 0.1998, 0.2243, 0.2496, 0.2755, 0.3017, 0.3281,
      0.3543, 0.3803, 0.406, 0.431, 0.455, 0.479, 0.502, 0.524, 0.546, 0.567,
      0.588, 0.61, 0.631, 0.653, 0.676, 0.699, 0.722, 0.745, 0.769, 0.793, 0.817,
      0.84, 0.862, 0.884, 0.904, 0.923, 0.941, 0.957, 0.97, 0.982, 0.99, 0.997, 1,
      1, 0.997, 0.99, 0.981, 0.968, 0.953, 0.935, 0.915, 0.892, 0.867, 0.84,
      0.811, 0.781, 0.749, 0.717, 0.683, 0.65, 0.616, 0.581, 0.548, 0.514, 0.481,
      0.448, 0.417, 0.3864, 0.3569, 0.3288, 0.3018, 0.2762, 0.2519, 0.2291,
      0.2076, 0.1876, 0.169, 0.1517, 0.1358, 0.1212, 0.1078, 0.0956, 0.0845,
      0.0745, 0.0655, 0.0574, 0.0502, 0.0438, 0.03816, 0.03315, 0.02874, 0.02487,
      0.02147, 0.01851, 0.01593, 0.01369, 0.01175, 0.01007, 0.00862, 0.00737,
      0.0063, 0.00538, 0.00459, 0.003913, 0.003335, 0.002842, 0.002421, 0.002062,
      0.001757, 0.001497, 0.001276, 0.001088, 0.000928, 0.000792, 0.000677,
      0.000579, 0.000496, 0.000425, 0.0003645, 0.0003129, 0.0002689, 0.0002313,
      0.0001991, 0.0001716, 0.000148, 0.0001277, 0.0001104, 0.0000954, 0.0000826,
      0.0000715, 0.000062, 0.0000538, 0.0000467, 0.0000406, 0.00003533,
      0.00003075, 0.00002679, 0.00002336, 0.00002038, 0.0000178, 0.00001556,
      0.0000136, 0.00001191, 0.00001043, 0.00000914, 0.00000802, 0.00000704,
      0.00000618, 0.00000544, 0.00000478, 0.00000421, 0.000003709, 0.00000327,
      0.000002884, 0.000002546
    ]
  },

  "Scone": {
    "wavelength": [
      3.9e2, 3.92e2, 3.94e2, 3.96e2, 3.98e2, 4.0e2, 4.02e2, 4.04e2, 4.06e2,
      4.08e2, 4.1e2, 4.12e2, 4.14e2, 4.16e2, 4.18e2, 4.2e2, 4.22e2, 4.24e2,
      4.26e2, 4.28e2, 4.3e2, 4.32e2, 4.34e2, 4.36e2, 4.38e2, 4.4e2, 4.42e2,
      4.44e2, 4.46e2, 4.48e2, 4.5e2, 4.52e2, 4.54e2, 4.56e2, 4.58e2, 4.6e2,
      4.62e2, 4.64e2, 4.66e2, 4.68e2, 4.7e2, 4.72e2, 4.74e2, 4.76e2, 4.78e2,
      4.8e2, 4.82e2, 4.84e2, 4.86e2, 4.88e2, 4.9e2, 4.92e2, 4.94e2, 4.96e2,
      4.98e2, 5.0e2, 5.02e2, 5.04e2, 5.06e2, 5.08e2, 5.1e2, 5.12e2, 5.14e2,
      5.16e2, 5.18e2, 5.2e2, 5.22e2, 5.24e2, 5.26e2, 5.28e2, 5.3e2, 5.32e2,
      5.34e2, 5.36e2, 5.38e2, 5.4e2, 5.42e2, 5.44e2, 5.46e2, 5.48e2, 5.5e2,
      5.52e2, 5.54e2, 5.56e2, 5.58e2, 5.6e2, 5.62e2, 5.64e2, 5.66e2, 5.68e2,
      5.7e2, 5.72e2, 5.74e2, 5.76e2, 5.78e2, 5.8e2, 5.82e2, 5.84e2, 5.86e2,
      5.88e2, 5.9e2, 5.92e2, 5.94e2, 5.96e2, 5.98e2, 6.0e2, 6.02e2, 6.04e2,
      6.06e2, 6.08e2, 6.1e2, 6.12e2, 6.14e2, 6.16e2, 6.18e2, 6.2e2, 6.22e2,
      6.24e2, 6.26e2, 6.28e2, 6.3e2, 6.32e2, 6.34e2, 6.36e2, 6.38e2, 6.4e2,
      6.42e2, 6.44e2, 6.46e2, 6.48e2, 6.5e2, 6.52e2, 6.54e2, 6.56e2, 6.58e2,
      6.6e2, 6.62e2, 6.64e2, 6.66e2, 6.68e2, 6.7e2, 6.72e2, 6.74e2, 6.76e2,
      6.78e2, 6.8e2, 6.82e2, 6.84e2, 6.86e2, 6.88e2, 6.9e2, 6.92e2, 6.94e2,
      6.96e2, 6.98e2, 7.0e2, 7.02e2, 7.04e2, 7.06e2, 7.08e2, 7.1e2, 7.12e2,
      7.14e2, 7.16e2, 7.18e2, 7.2e2, 7.22e2, 7.24e2, 7.26e2, 7.28e2, 7.3e2
    ],
    "value": [
      7.77e-3, 4.1e-2, 7.43e-2, 1.08e-1, 1.41e-1, 1.74e-1, 2.12e-1, 2.5e-1,
      2.87e-1, 3.25e-1, 3.63e-1, 4.23e-1, 4.82e-1, 5.42e-1, 6.02e-1, 6.61e-1,
      7.1e-1, 7.59e-1, 8.07e-1, 8.56e-1, 9.04e-1, 9.24e-1, 9.43e-1, 9.62e-1,
      9.81e-1, 1.0, 9.83e-1, 9.66e-1, 9.5e-1, 9.33e-1, 9.16e-1, 8.93e-1, 8.7e-1,
      8.48e-1, 8.25e-1, 8.02e-1, 7.8e-1, 7.58e-1, 7.37e-1, 7.15e-1, 6.93e-1,
      6.48e-1, 6.04e-1, 5.59e-1, 5.15e-1, 4.7e-1, 4.32e-1, 3.93e-1, 3.54e-1,
      3.16e-1, 2.77e-1, 2.55e-1, 2.32e-1, 2.1e-1, 1.87e-1, 1.65e-1, 1.51e-1,
      1.37e-1, 1.23e-1, 1.09e-1, 9.56e-2, 8.59e-2, 7.63e-2, 6.67e-2, 5.7e-2,
      4.74e-2, 4.3e-2, 3.87e-2, 3.43e-2, 3.0e-2, 2.56e-2, 2.3e-2, 2.04e-2,
      1.77e-2, 1.51e-2, 1.24e-2, 1.1e-2, 9.63e-3, 8.24e-3, 6.84e-3, 5.44e-3,
      4.82e-3, 4.2e-3, 3.57e-3, 2.95e-3, 2.33e-3, 2.18e-3, 2.02e-3, 1.86e-3,
      1.71e-3, 1.55e-3, 1.4e-3, 1.24e-3, 1.09e-3, 9.32e-4, 7.77e-4, 7.77e-4,
      7.77e-4, 7.77e-4, 7.77e-4, 7.77e-4, 7.77e-4, 7.77e-4, 7.77e-4, 7.77e-4,
      7.77e-4, 6.22e-4, 4.66e-4, 3.11e-4, 1.55e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
  },

  "Macula": {
    "wavelength": [
      4.0e2, 4.05e2, 4.1e2, 4.15e2, 4.2e2, 4.25e2, 4.3e2, 4.35e2, 4.4e2, 4.45e2,
      4.5e2, 4.55e2, 4.6e2, 4.65e2, 4.7e2, 4.75e2, 4.8e2, 4.85e2, 4.9e2, 4.95e2,
      5.0e2, 5.05e2, 5.1e2, 5.15e2, 5.2e2, 5.25e2, 5.3e2, 5.35e2, 5.4e2, 5.45e2,
      5.5e2, 5.55e2, 5.6e2, 5.65e2, 5.7e2, 5.75e2, 5.8e2, 5.85e2, 5.9e2, 5.95e2,
      6.0e2, 6.05e2, 6.1e2, 6.15e2, 6.2e2, 6.25e2, 6.3e2, 6.35e2, 6.4e2, 6.45e2,
      6.5e2, 6.55e2, 6.6e2, 6.65e2, 6.7e2, 6.75e2, 6.8e2, 6.85e2, 6.9e2, 6.95e2,
      7.0e2, 7.05e2, 7.1e2, 7.15e2, 7.2e2, 7.25e2, 7.3e2
    ],
    "value": [
      2.24e-1, 2.44e-1, 2.64e-1, 2.83e-1, 3.14e-1, 3.53e-1, 3.83e-1, 4.0e-1,
      4.17e-1, 4.4e-1, 4.66e-1, 4.9e-1, 5.0e-1, 4.83e-1, 4.62e-1, 4.38e-1,
      4.37e-1, 4.36e-1, 4.27e-1, 4.04e-1, 3.51e-1, 2.83e-1, 2.14e-1, 1.55e-1,
      9.6e-2, 6.8e-2, 4.0e-2, 2.85e-2, 1.7e-2, 1.3e-2, 9.0e-3, 8.5e-3, 8.0e-3,
      6.5e-3, 5.0e-3, 4.5e-3, 4.0e-3, 2.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
  },

  "Melanopsin": {
    "wavelength": [
      3.8e2, 3.81e2, 3.82e2, 3.83e2, 3.84e2, 3.85e2, 3.86e2, 3.87e2, 3.88e2,
      3.89e2, 3.9e2, 3.91e2, 3.92e2, 3.93e2, 3.94e2, 3.95e2, 3.96e2, 3.97e2,
      3.98e2, 3.99e2, 4.0e2, 4.01e2, 4.02e2, 4.03e2, 4.04e2, 4.05e2, 4.06e2,
      4.07e2, 4.08e2, 4.09e2, 4.1e2, 4.11e2, 4.12e2, 4.13e2, 4.14e2, 4.15e2,
      4.16e2, 4.17e2, 4.18e2, 4.19e2, 4.2e2, 4.21e2, 4.22e2, 4.23e2, 4.24e2,
      4.25e2, 4.26e2, 4.27e2, 4.28e2, 4.29e2, 4.3e2, 4.31e2, 4.32e2, 4.33e2,
      4.34e2, 4.35e2, 4.36e2, 4.37e2, 4.38e2, 4.39e2, 4.4e2, 4.41e2, 4.42e2,
      4.43e2, 4.44e2, 4.45e2, 4.46e2, 4.47e2, 4.48e2, 4.49e2, 4.5e2, 4.51e2,
      4.52e2, 4.53e2, 4.54e2, 4.55e2, 4.56e2, 4.57e2, 4.58e2, 4.59e2, 4.6e2,
      4.61e2, 4.62e2, 4.63e2, 4.64e2, 4.65e2, 4.66e2, 4.67e2, 4.68e2, 4.69e2,
      4.7e2, 4.71e2, 4.72e2, 4.73e2, 4.74e2, 4.75e2, 4.76e2, 4.77e2, 4.78e2,
      4.79e2, 4.8e2, 4.81e2, 4.82e2, 4.83e2, 4.84e2, 4.85e2, 4.86e2, 4.87e2,
      4.88e2, 4.89e2, 4.9e2, 4.91e2, 4.92e2, 4.93e2, 4.94e2, 4.95e2, 4.96e2,
      4.97e2, 4.98e2, 4.99e2, 5.0e2, 5.01e2, 5.02e2, 5.03e2, 5.04e2, 5.05e2,
      5.06e2, 5.07e2, 5.08e2, 5.09e2, 5.1e2, 5.11e2, 5.12e2, 5.13e2, 5.14e2,
      5.15e2, 5.16e2, 5.17e2, 5.18e2, 5.19e2, 5.2e2, 5.21e2, 5.22e2, 5.23e2,
      5.24e2, 5.25e2, 5.26e2, 5.27e2, 5.28e2, 5.29e2, 5.3e2, 5.31e2, 5.32e2,
      5.33e2, 5.34e2, 5.35e2, 5.36e2, 5.37e2, 5.38e2, 5.39e2, 5.4e2, 5.41e2,
      5.42e2, 5.43e2, 5.44e2, 5.45e2, 5.46e2, 5.47e2, 5.48e2, 5.49e2, 5.5e2,
      5.51e2, 5.52e2, 5.53e2, 5.54e2, 5.55e2, 5.56e2, 5.57e2, 5.58e2, 5.59e2,
      5.6e2, 5.61e2, 5.62e2, 5.63e2, 5.64e2, 5.65e2, 5.66e2, 5.67e2, 5.68e2,
      5.69e2, 5.7e2, 5.71e2, 5.72e2, 5.73e2, 5.74e2, 5.75e2, 5.76e2, 5.77e2,
      5.78e2, 5.79e2, 5.8e2, 5.81e2, 5.82e2, 5.83e2, 5.84e2, 5.85e2, 5.86e2,
      5.87e2, 5.88e2, 5.89e2, 5.9e2, 5.91e2, 5.92e2, 5.93e2, 5.94e2, 5.95e2,
      5.96e2, 5.97e2, 5.98e2, 5.99e2, 6.0e2, 6.01e2, 6.02e2, 6.03e2, 6.04e2,
      6.05e2, 6.06e2, 6.07e2, 6.08e2, 6.09e2, 6.1e2, 6.11e2, 6.12e2, 6.13e2,
      6.14e2, 6.15e2, 6.16e2, 6.17e2, 6.18e2, 6.19e2, 6.2e2, 6.21e2, 6.22e2,
      6.23e2, 6.24e2, 6.25e2, 6.26e2, 6.27e2, 6.28e2, 6.29e2, 6.3e2, 6.31e2,
      6.32e2, 6.33e2, 6.34e2, 6.35e2, 6.36e2, 6.37e2, 6.38e2, 6.39e2, 6.4e2,
      6.41e2, 6.42e2, 6.43e2, 6.44e2, 6.45e2, 6.46e2, 6.47e2, 6.48e2, 6.49e2,
      6.5e2, 6.51e2, 6.52e2, 6.53e2, 6.54e2, 6.55e2, 6.56e2, 6.57e2, 6.58e2,
      6.59e2, 6.6e2, 6.61e2, 6.62e2, 6.63e2, 6.64e2, 6.65e2, 6.66e2, 6.67e2,
      6.68e2, 6.69e2, 6.7e2, 6.71e2, 6.72e2, 6.73e2, 6.74e2, 6.75e2, 6.76e2,
      6.77e2, 6.78e2, 6.79e2, 6.8e2, 6.81e2, 6.82e2, 6.83e2, 6.84e2, 6.85e2,
      6.86e2, 6.87e2, 6.88e2, 6.89e2, 6.9e2, 6.91e2, 6.92e2, 6.93e2, 6.94e2,
      6.95e2, 6.96e2, 6.97e2, 6.98e2, 6.99e2, 7.0e2, 7.01e2, 7.02e2, 7.03e2,
      7.04e2, 7.05e2, 7.06e2, 7.07e2, 7.08e2, 7.09e2, 7.1e2, 7.11e2, 7.12e2,
      7.13e2, 7.14e2, 7.15e2, 7.16e2, 7.17e2, 7.18e2, 7.19e2, 7.2e2, 7.21e2,
      7.22e2, 7.23e2, 7.24e2, 7.25e2, 7.26e2, 7.27e2, 7.28e2, 7.29e2, 7.3e2
    ],
    "value": [
      1.21e-3, 1.52e-3, 1.88e-3, 2.27e-3, 2.71e-3, 3.2e-3, 3.74e-3, 4.35e-3,
      4.88e-3, 5.46e-3, 6.09e-3, 7.54e-3, 9.1e-3, 1.08e-2, 1.26e-2, 1.45e-2,
      1.66e-2, 1.88e-2, 2.11e-2, 2.34e-2, 2.59e-2, 3.06e-2, 3.54e-2, 4.05e-2,
      4.59e-2, 5.15e-2, 5.73e-2, 6.33e-2, 7.01e-2, 7.71e-2, 8.44e-2, 9.44e-2,
      1.05e-1, 1.15e-1, 1.26e-1, 1.37e-1, 1.49e-1, 1.61e-1, 1.74e-1, 1.88e-1,
      2.02e-1, 2.18e-1, 2.34e-1, 2.5e-1, 2.66e-1, 2.83e-1, 3.0e-1, 3.17e-1,
      3.35e-1, 3.54e-1, 3.72e-1, 3.88e-1, 4.04e-1, 4.2e-1, 4.36e-1, 4.52e-1,
      4.68e-1, 4.85e-1, 5.02e-1, 5.2e-1, 5.38e-1, 5.51e-1, 5.64e-1, 5.77e-1,
      5.91e-1, 6.05e-1, 6.19e-1, 6.33e-1, 6.47e-1, 6.62e-1, 6.76e-1, 6.89e-1,
      7.01e-1, 7.14e-1, 7.27e-1, 7.41e-1, 7.54e-1, 7.67e-1, 7.82e-1, 7.96e-1,
      8.1e-1, 8.22e-1, 8.34e-1, 8.45e-1, 8.57e-1, 8.69e-1, 8.8e-1, 8.91e-1,
      9.01e-1, 9.11e-1, 9.21e-1, 9.31e-1, 9.41e-1, 9.49e-1, 9.57e-1, 9.65e-1,
      9.72e-1, 9.79e-1, 9.84e-1, 9.89e-1, 9.94e-1, 9.97e-1, 9.99e-1, 1.0, 1.0,
      1.0, 9.99e-1, 9.97e-1, 9.95e-1, 9.92e-1, 9.89e-1, 9.85e-1, 9.81e-1, 9.75e-1,
      9.69e-1, 9.62e-1, 9.54e-1, 9.46e-1, 9.37e-1, 9.28e-1, 9.17e-1, 9.06e-1,
      8.95e-1, 8.83e-1, 8.71e-1, 8.58e-1, 8.44e-1, 8.3e-1, 8.16e-1, 8.01e-1,
      7.86e-1, 7.69e-1, 7.53e-1, 7.36e-1, 7.19e-1, 7.01e-1, 6.84e-1, 6.67e-1,
      6.5e-1, 6.32e-1, 6.14e-1, 5.97e-1, 5.8e-1, 5.63e-1, 5.45e-1, 5.28e-1,
      5.11e-1, 4.94e-1, 4.77e-1, 4.6e-1, 4.44e-1, 4.28e-1, 4.12e-1, 3.97e-1,
      3.81e-1, 3.66e-1, 3.52e-1, 3.38e-1, 3.24e-1, 3.1e-1, 2.97e-1, 2.84e-1,
      2.71e-1, 2.59e-1, 2.47e-1, 2.35e-1, 2.24e-1, 2.13e-1, 2.03e-1, 1.93e-1,
      1.83e-1, 1.74e-1, 1.64e-1, 1.56e-1, 1.47e-1, 1.39e-1, 1.32e-1, 1.24e-1,
      1.17e-1, 1.11e-1, 1.04e-1, 9.82e-2, 9.24e-2, 8.69e-2, 8.16e-2, 7.66e-2,
      7.19e-2, 6.74e-2, 6.32e-2, 5.92e-2, 5.54e-2, 5.18e-2, 4.84e-2, 4.53e-2,
      4.23e-2, 3.95e-2, 3.68e-2, 3.44e-2, 3.2e-2, 2.98e-2, 2.78e-2, 2.59e-2,
      2.41e-2, 2.24e-2, 2.08e-2, 1.93e-2, 1.79e-2, 1.66e-2, 1.54e-2, 1.43e-2,
      1.33e-2, 1.23e-2, 1.14e-2, 1.05e-2, 9.76e-3, 9.03e-3, 8.35e-3, 7.72e-3,
      7.15e-3, 6.61e-3, 6.11e-3, 5.65e-3, 5.22e-3, 4.83e-3, 4.46e-3, 4.12e-3,
      3.8e-3, 3.51e-3, 3.25e-3, 3.0e-3, 2.77e-3, 2.56e-3, 2.36e-3, 2.18e-3,
      2.01e-3, 1.86e-3, 1.72e-3, 1.58e-3, 1.46e-3, 1.35e-3, 1.25e-3, 1.16e-3,
      1.07e-3, 9.86e-4, 9.12e-4, 8.43e-4, 7.79e-4, 7.21e-4, 6.67e-4, 6.17e-4,
      5.71e-4, 5.29e-4, 4.9e-4, 4.54e-4, 4.2e-4, 3.9e-4, 3.61e-4, 3.35e-4,
      3.11e-4, 2.88e-4, 2.67e-4, 2.48e-4, 2.3e-4, 2.14e-4, 1.99e-4, 1.84e-4,
      1.71e-4, 1.59e-4, 1.48e-4, 1.38e-4, 1.28e-4, 1.19e-4, 1.11e-4, 1.03e-4,
      9.58e-5, 8.92e-5, 8.3e-5, 7.73e-5, 7.2e-5, 6.71e-5, 6.25e-5, 5.82e-5,
      5.43e-5, 5.06e-5, 4.71e-5, 4.4e-5, 4.1e-5, 3.82e-5, 3.57e-5, 3.33e-5,
      3.11e-5, 2.9e-5, 2.71e-5, 2.53e-5, 2.36e-5, 2.21e-5, 2.06e-5, 1.93e-5,
      1.8e-5, 1.68e-5, 1.57e-5, 1.47e-5, 1.38e-5, 1.29e-5, 1.21e-5, 1.13e-5,
      1.06e-5, 9.9e-6, 9.27e-6, 8.68e-6, 8.13e-6, 7.62e-6, 7.14e-6, 6.69e-6,
      6.28e-6, 5.89e-6, 5.52e-6, 5.17e-6, 4.86e-6, 4.56e-6, 4.28e-6, 4.02e-6,
      3.77e-6, 3.54e-6, 3.32e-6, 3.12e-6, 2.93e-6, 2.76e-6, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0
    ]
  },

  "setwavelength": [
    380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
    395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409,
    410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424,
    425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
    440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454,
    455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469,
    470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484,
    485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
    500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514,
    515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529,
    530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544,
    545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
    560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574,
    575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,
    590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604,
    605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,
    620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634,
    635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
    650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664,
    665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,
    680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694,
    695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709,
    710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
    725, 726, 727, 728, 729, 730
  ]
}



# calculate relevant quantities and linearly interpolate to 'setwavelengths':
#----------------------------------------------------------------------------
    
thickness = 1.0 # for macular pigment

keys = list(CLA2d0_data.keys())
wls = CLA2d0_data['setwavelength']
efs = np.zeros((len(keys),len(wls)))
efs[0] = np.array(wls)
cnt = 0
for i,key in enumerate(keys):
    if key != 'setwavelength':
        cnt +=1
        # get wavelengths and data:
        w = np.array(CLA2d0_data[key]['wavelength'])
        v = np.array(CLA2d0_data[key]['value'])
        wv = np.vstack((w,v))
        
        # process macula optical density to macula transmission:
        if key == 'Macula': 
            wv[1] = 10**(-thickness*wv[1]) 
            extrap_values = 1.0
        else: 
            extrap_values = 0.0
        
        # interpolate:
        efs[cnt] = cie_interp(wv, efs[0], kind = 'linear', extrap_values = extrap_values)[1:]
    
keys.remove('setwavelength')

# Pre-calculate Vl/macula & Scl/macula:
new_order = ['Vlambda', 'Vprime',  'Vlambda/mac', 'Scone/mac', 'Melanopsin','Scone','ybar']
p_mac = keys.index('Macula')
mac = efs[p_mac+1] # macula transmission
efs_new = np.zeros((len(new_order)+1,efs.shape[-1]))
efs_new[0] = efs[0]
for i,label in enumerate(new_order):
    label_i = label if '/mac' not in label else label[:-4]
    p = keys.index(label_i)
    efs_new[i+1] = efs[p+1]
    
    if '/mac' in label:
        efs_new[i+1] = efs_new[i+1]/mac
    
    efs_new[i+1] = efs_new[i+1]/efs_new[i+1].max()
        
df = pd.DataFrame(efs_new.T,columns = ['wl']+new_order)#
df.to_csv(_FILE_NAME,header=True, index=False,float_format='%1.9f')

        
        
     
    

        
        
    
    