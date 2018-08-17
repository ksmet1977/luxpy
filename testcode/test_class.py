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
import pytest
import numpy as np

class TestClass(object):
	
	@pytest.fixture
	def getvars(scope="module"):
		# get some spectral data:
		S = lx._CIE_ILLUMINANTS['F4']
		spds = lx._IESTM30['S']['data'][:5]
		rfls = lx._IESTM30['R']['99']['5nm'][:6]
		xyz, xyzw = lx.spd_to_xyz(spds, rfl = rfls, cieobs = '1931_2', out = 2)
		return S,spds,rfls,xyz,xyzw

	def test_getdata(self, getvars):
		# check getdata():
		S,spds,rfls,xyz,xyzw = getvars
		rfls_df = lx.getdata(rfls,kind='df')
		rfls_np = lx.getdata(rfls_df,kind='np')
		assert np.isclose(rfls_df,rfls_df.values).all()

	def test_spd_normalize(self,getvars):
		# check spd_normalize:
		S,spds,rfls,xyz,xyzw = getvars
		rfls_np_cp = rfls.copy()
		rfls_norm = lx.spd_normalize(rfls)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'area', norm_f = 2)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'lambda', norm_f = 560)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'max', norm_f = 1)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'pu', norm_f = 1)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'ru', norm_f = 1)
		rfls_norm = lx.spd_normalize(rfls_np_cp, norm_type = 'qu', norm_f = 1)

	def test_spd_to_xyz(self,getvars):
		# check spd_to_xyz:
		S,spds,rfls,xyz,xyzw = getvars
		xyz = lx.spd_to_xyz(spds)
		xyz = lx.spd_to_xyz(spds, relative = False)
		xyz = lx.spd_to_xyz(spds, rfl = rfls)
		xyz, xyzw = lx.spd_to_xyz(spds, rfl = rfls, cieobs = '1931_2', out = 2)


	def test_xyz_to_cct(self, getvars):
		# check xyz_to_cct():
		S,spds,rfls,xyz,xyzw = getvars
		cct,duv = lx.xyz_to_cct(xyzw, cieobs='1931_2', out = 2)

	def test_xyY(self, getvars):# check xyz_to_..., ..._to_xyz:
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_Yxy(xyzw)
		lab = lx.xyz_to_Yxy(xyz)
		xyzw_ = lx.Yxy_to_xyz(labw)
		xyz_ = lx.Yxy_to_xyz(lab)
		assert np.isclose(xyz,xyz_).all()
		assert np.isclose(xyzw,xyzw_).all()
		
	def test_Yuv(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_Yuv(xyzw)
		lab = lx.xyz_to_Yuv(xyz)
		xyzw_ = lx.Yuv_to_xyz(labw)
		xyz_ = lx.Yuv_to_xyz(lab)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 
	
	def test_lab(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_lab(xyzw, xyzw=xyzw[:1,:])
		lab = lx.xyz_to_lab(xyz, xyzw=xyzw[:1,:])
		xyzw_ = lx.lab_to_xyz(labw, xyzw=xyzw[:1,:])
		xyz_ = lx.lab_to_xyz(lab, xyzw=xyzw[:1,:])
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 
	
	def test_luv(self, getvars): 
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_luv(xyzw, xyzw=xyzw)
		lab = lx.xyz_to_luv(xyz, xyzw=xyzw)
		xyzw_ = lx.luv_to_xyz(labw, xyzw=xyzw)
		xyz_ = lx.luv_to_xyz(lab, xyzw=xyzw)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 

	def test_ipt(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_ipt(xyzw)
		lab = lx.xyz_to_ipt(xyz)
		xyzw_ = lx.ipt_to_xyz(labw)
		xyz_ = lx.ipt_to_xyz(lab)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 

	def test_wuv(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_wuv(xyzw, xyzw=xyzw)
		lab = lx.xyz_to_wuv(xyz, xyzw=xyzw)
		xyzw_ = lx.wuv_to_xyz(labw, xyzw=xyzw)
		xyz_ = lx.wuv_to_xyz(lab, xyzw=xyzw)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 

	def test_Vrb_mb(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_Vrb_mb(xyzw)
		lab = lx.xyz_to_Vrb_mb(xyz)
		xyzw_ = lx.Vrb_mb_to_xyz(labw)
		xyz_ = lx.Vrb_mb_to_xyz(lab)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 
	
	def test_Ydlep(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_Ydlep(xyzw)
		lab = lx.xyz_to_Ydlep(xyz)
		xyzw_ = lx.Ydlep_to_xyz(labw)
		xyz_ = lx.Ydlep_to_xyz(lab)
		assert np.isclose(xyz,xyz_,atol=1e-2).all() 
		assert np.isclose(xyzw,xyzw_,atol=1e-2).all() 
	
	def test_lms(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		labw = lx.xyz_to_lms(xyzw)
		lab = lx.xyz_to_lms(xyz)
		xyzw_ = lx.lms_to_xyz(labw)
		xyz_ = lx.lms_to_xyz(lab)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 

	def test_srgb(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		lab = lx.xyz_to_srgb(xyz[1:,:]) # first xyz row out of srgb gamut, so non-reversible!
		xyz_ = lx.srgb_to_xyz(lab)
		assert np.isclose(xyz[1:,:],xyz_).all() 

	def test_cam_sww16(self, getvars):
		# cam_sww16:
		S,spds,rfls,xyz,xyzw = getvars
		xyz, xyzw = lx.spd_to_xyz(S, rfl = rfls, cieobs = '2006_10', out = 2)
		jabw = lx.xyz_to_lab_cam_sww16(xyzw, xyzw=xyzw.copy())
		jab = lx.xyz_to_lab_cam_sww16(xyz, xyzw=xyzw)
		xyzw_ = lx.lab_cam_sww16_to_xyz(jabw, xyzw=xyzw)
		xyz_ = lx.lab_cam_sww16_to_xyz(jab, xyzw=xyzw)
		assert np.isclose(xyz,xyz_,atol=1e-1).all() 
		assert np.isclose(xyzw,xyzw_,atol=1e-1).all() 


	def test_cam15u(self, getvars):	
		# cam15u:
		S,spds,rfls,xyz,xyzw = getvars
		xyz, xyzw = lx.spd_to_xyz(spds, rfl = rfls, cieobs = '2006_10', out = 2)
		qabw = lx.xyz_to_qabW_cam15u(xyzw, fov=10)
		qab = lx.xyz_to_qabW_cam15u(xyz, fov=10)
		xyzw_ = lx.qabW_cam15u_to_xyz(qabw, fov=10)
		xyz_ = lx.qabW_cam15u_to_xyz(qab, fov=10)
		assert np.isclose(xyz,xyz_).all() 
		assert np.isclose(xyzw,xyzw_).all() 


	#test IES Rf, CIE Rf, CRI2012, MCRI, CQS:
	def test_rf_tm30_15(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		rf = lx.cri.spd_to_iesrf_tm30_15(S)
		print(rf)
		assert np.isclose(rf,np.asarray([[51.68898613]])).all()
	
	def test_rf_cie224(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		rf = lx.cri.spd_to_cierf_224_2017(S)
		print(rf)
		assert np.isclose(rf,np.asarray([[56.86193867]])).all()
		
	def test_cri2012(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		rf = lx.cri.spd_to_cri2012(S)
		print(rf)
		assert np.isclose(rf,np.asarray([[50.61447705]])).all()
		
	def test_mcri(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		rm = lx.cri.spd_to_mcri(S, D=0.65)
		print(rm)
		assert np.isclose(rm,np.asarray([[0.00036785]])).all()
		
	def test_scq(self, getvars):
		S,spds,rfls,xyz,xyzw = getvars
		qa = lx.cri.spd_to_cqs(S,version='v7.5', out='Qp')
		print(qa)
		assert np.isclose(qa,np.asarray([[54.60655527]])).all()