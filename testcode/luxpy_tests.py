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
Module for testing LuxPy toolbox (v1.3.06)
==================================================================


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import luxpy as lx

# get some spectral data:
spds = lx._IESTM3018['S']['data'][:5]
rfls = lx._IESTM3018['R']['99']['5nm'][:6]

# check getdata():
rfls_df = lx.getdata(rfls,kind='df')
rfls_np = lx.getdata(rfls_df,kind='np')


# check spd_normalize:
rfls_norm = lx.spd_normalize(rfls_np)
rfls_np_cp = rfls_np.copy()
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'area', norm_f = 2)
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'lambda', norm_f = 560)
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'max', norm_f = 1)
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'pu', norm_f = 1)
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'ru', norm_f = 1)
rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'qu', norm_f = 1)

# check spd_to_xyz:
xyz = lx.spd_to_xyz(spds)
xyz = lx.spd_to_xyz(spds, relative = False)
xyz = lx.spd_to_xyz(spds, rfl = rfls)
xyz, xyzw = lx.spd_to_xyz(spds, rfl = rfls, cieobs = '1931_2', out = 2)

# check xyz_to_cct():
cct,duv = lx.xyz_to_cct(xyzw, cieobs='1931_2', out = 2)

# check xyz_to_..., ..._to_xyz:
labw = lx.xyz_to_Yxy(xyzw)
lab = lx.xyz_to_Yxy(xyz)
xyzw_ = lx.Yxy_to_xyz(labw)
xyz_ = lx.Yxy_to_xyz(lab)

labw = lx.xyz_to_Yuv(xyzw)
lab = lx.xyz_to_Yuv(xyz)
xyzw_ = lx.Yuv_to_xyz(labw)
xyz_ = lx.Yuv_to_xyz(lab)

labw = lx.xyz_to_lab(xyzw, xyzw=xyzw[:1,:])
lab = lx.xyz_to_lab(xyz, xyzw=xyzw[:1,:])
xyzw_ = lx.lab_to_xyz(labw, xyzw=xyzw[:1,:])
xyz_ = lx.lab_to_xyz(lab, xyzw=xyzw[:1,:])

labw = lx.xyz_to_luv(xyzw, xyzw=xyzw)
lab = lx.xyz_to_luv(xyz, xyzw=xyzw)
xyzw_ = lx.luv_to_xyz(labw, xyzw=xyzw)
xyz_ = lx.luv_to_xyz(lab, xyzw=xyzw)

labw = lx.xyz_to_ipt(xyzw)
lab = lx.xyz_to_ipt(xyz)
xyzw_ = lx.ipt_to_xyz(labw)
xyz_ = lx.ipt_to_xyz(lab)

labw = lx.xyz_to_wuv(xyzw, xyzw=xyzw)
lab = lx.xyz_to_wuv(xyz, xyzw=xyzw)
xyzw_ = lx.wuv_to_xyz(labw, xyzw=xyzw)
xyz_ = lx.wuv_to_xyz(lab, xyzw=xyzw)

labw = lx.xyz_to_Vrb_mb(xyzw)
lab = lx.xyz_to_Vrb_mb(xyz)
xyzw_ = lx.Vrb_mb_to_xyz(labw)
xyz_ = lx.Vrb_mb_to_xyz(lab)

labw = lx.xyz_to_Ydlep(xyzw)
lab = lx.xyz_to_Ydlep(xyz)
xyzw_ = lx.Ydlep_to_xyz(labw)
xyz_ = lx.Ydlep_to_xyz(lab)

labw = lx.xyz_to_lms(xyzw)
lab = lx.xyz_to_lms(xyz)
xyzw_ = lx.lms_to_xyz(labw)
xyz_ = lx.lms_to_xyz(lab)

labw = lx.xyz_to_srgb(xyzw)
lab = lx.xyz_to_srgb(xyz)
xyzw_ = lx.srgb_to_xyz(labw)
xyz_ = lx.srgb_to_xyz(lab)

print(labw.shape)
print(lab.shape)
print(xyzw-xyzw_)
print(xyz-xyz_)

# cam_sww6:
xyz, xyzw = lx.spd_to_xyz(spds, rfl = rfls, cieobs = '2006_10', out = 2)
jabw = lx.xyz_to_lab_cam_sww16(xyzw, xyzw=xyzw.copy())
jab = lx.xyz_to_lab_cam_sww16(xyz, xyzw=xyzw)
xyzw_ = lx.lab_cam_sww16_to_xyz(jabw, xyzw=xyzw)
xyz_ = lx.lab_cam_sww16_to_xyz(jab, xyzw=xyzw)
print(jabw.shape)
print(jab.shape)
print(xyzw-xyzw_)
print(xyz-xyz_)

# cam15u:
qabw = lx.xyz_to_qabW_cam15u(xyzw, fov=10)
qab = lx.xyz_to_qabW_cam15u(xyz, fov=10)
xyzw_ = lx.qabW_cam15u_to_xyz(qabw, fov=10)
xyz_ = lx.qabW_cam15u_to_xyz(qab, fov=10)
print(qabw.shape)
print(qab.shape)
print(xyzw-xyzw_)
print(xyz-xyz_)

#test IES Rf, CIE Rf, CRI2012, MCRI, CQS:
S = lx._CIE_ILLUMINANTS['F4']
rf = lx.cri.spd_to_iesrf_tm30_15(S)
print(rf)
rf = lx.cri.spd_to_cierf_224_2017(S)
print(rf)
rf = lx.cri.spd_to_cri2012(S)
print(rf)
rm = lx.cri.spd_to_mcri(spds, D=0.65)
print(rm)
qa = lx.cri.spd_to_cqs(S,version='v7.5', out='Qp')
print(qa)