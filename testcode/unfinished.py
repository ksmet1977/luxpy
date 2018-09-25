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
Temp
==================================================================


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

def plot_chrom():
    x = np.asarray([0.175596, 0.172787, 0.170806, 0.170085, 0.160343, 0.146958, 0.139149,\
                    0.133536, 0.126688, 0.115830, 0.109616, 0.099146, 0.091310, 0.078130,\
                    0.068717, 0.054675, 0.040763, 0.027497, 0.016270, 0.008169, 0.004876,\
                    0.003983, 0.003859, 0.004646, 0.007988, 0.013870, 0.022244, 0.027273,\
                    0.032820, 0.038851, 0.045327, 0.052175, 0.059323, 0.066713, 0.074299,\
                    0.089937, 0.114155, 0.138695, 0.154714, 0.192865, 0.229607, 0.265760,\
                    0.301588, 0.337346, 0.373083, 0.408717, 0.444043, 0.478755, 0.512467,\
                    0.544767, 0.575132, 0.602914, 0.627018, 0.648215, 0.665746, 0.680061,\
                    0.691487, 0.700589, 0.707901, 0.714015, 0.719017, 0.723016, 0.734674])
    y = np.asarray([ 0.005295, 0.004800, 0.005472, 0.005976, 0.014496, 0.026643, 0.035211,\
                    0.042704, 0.053441, 0.073601, 0.086866, 0.112037, 0.132737, 0.170464,\
                    0.200773, 0.254155, 0.317049, 0.387997, 0.463035, 0.538504, 0.587196,\
                    0.610526, 0.654897, 0.675970, 0.715407, 0.750246, 0.779682, 0.792153,\
                    0.802971, 0.812059, 0.819430, 0.825200, 0.829460, 0.832306, 0.833833,\
                    0.833316, 0.826231, 0.814796, 0.805884, 0.781648, 0.754347, 0.724342,\
                    0.692326, 0.658867, 0.624470, 0.589626, 0.554734, 0.520222, 0.486611,\
                    0.454454, 0.424252, 0.396516, 0.372510, 0.351413, 0.334028, 0.319765,\
                    0.308359, 0.299317, 0.292044, 0.285945, 0.280951, 0.276964, 0.265326])
    N = x.shape[0]
    i = 1
    e = 1/3
    steps = 25
    xy4rgb = np.zeros((N*steps*4, 5))
    for w in np.arange(N):                              # wavelength
        w2 = np.mod(w,N) + 1
        a1 = np.arctan2(y[w] - e, x[w] - e)             # start angle
        a2 = np.arctan2(y[w2] - e, x[w2] - e)           # end angle
        r1 = ((x[w] - e)**2 + (y[w] - e)**2)**0.5       # start radius
        r2 = ((x[w2] - e)**2 + (y[w2] - e)**2)**0.5     # end radius
        xyz = np.zeros((4,3))
        for c in np.arange(steps):                      # colorfulness
            # patch polygon
            xyz[0,0] = e + r1*np.cos(a1)*c/steps
            xyz[0,1] = e + r1*np.sin(a1)*c/steps
            xyz[0,2] = 1 - xyz[0,0] - xyz[0,1]
            xyz[1,0] = e + r1*np.cos(a1)*(c-1)/steps
            xyz[1,1] = e + r1*np.sin(a1)*(c-1)/steps
            xyz[1,2] = 1 - xyz[1,0] - xyz[1,1]
            xyz[2,0] = e + r2*np.cos(a2)*(c-1)/steps
            xyz[2,1] = e + r2*np.sin(a2)*(c-1)/steps
            xyz[2,2] = 1 - xyz[2,0] - xyz[2,1]
            xyz[3,0] = e + r2*np.cos(a2)*c/steps
            xyz[3,1] = e + r2*np.sin(a2)*c/steps
            xyz[3,2] = 1 - xyz[3,0] - xyz[3,1]
            # compute sRGB for vertices
            rgb = xyz_to_srgb(xyz)
            # store the results
            xy4rgb[i:i+2,0:2] = xyz[:,0:2]
            xy4rgb[i:i+2,2:5] = rgb
            i = i + 4


    rows = xy4rgb.shape[0]
    f = [1, 2, 3, 4]
    v = zeros((4,3))
#    for i = 1:4:rows
#        v(:,1:2) = xy4rgb(i:i+3,1:2)
#        patch('Vertices',v, 'Faces',f, 'EdgeColor','none', ...
#            'FaceVertexCData',xy4rgb[i:i+3,3:5],'FaceColor','interp')