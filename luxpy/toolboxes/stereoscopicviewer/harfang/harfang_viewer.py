# -*- coding: utf-8 -*-
"""
Module for viewing equirectangular images in a steamVR supported head-mounted-display.
======================================================================================

Created on Sat Nov 19 09:45:11 2022

@author: u0032318
"""
import os
import sys
import copy
import time
from math import pi, sin, cos
import numpy as np
from ctypes import c_uint32

from luxpy.utils import is_importable 

# from imageio import imread, imsave
# import harfang as hg 

# import imageio (and if necessary install it):
success = is_importable('imageio', try_pip_install = True)
if success:
    try: 
        from imageio.v3 import imread, imsave
    except:
        from imageio import imread, imsave
   
# import harfang (and if necessary install it):
success = is_importable('harfang', try_pip_install = True)
if success:
    import harfang as hg


__all__ = ['_PKG_PATH', 'CreateSphereModel','CreatePlaneModel',
           'create_material','update_material_texture',
           'makeColorTex', 'split_SingleSphericalTex',
           'Shader', 'Scene','Camera','Material','Screen', 'Eye', 
           'HmdStereoViewer','generate_stimulus_tex_list',
           'generate_rgba_texs_iml','get_rgbFromTexPaths',
           'getRectMask', 'getRoiImage',
           'get_xyz_from_xyzmap_roi','get_rgb_from_rgbtexpath']

#==============================================================================

_PKG_PATH = os.path.dirname(__file__)
_SCREEN_TEX_WH = (800,600)
_TEXTURE_FLAGS = hg.TF_UBorder | hg.TF_VBorder | hg.TF_SamplerMinPoint | hg.TF_SamplerMagPoint

#==============================================================================

def CreateSphereModel(decl : hg.VertexLayout = None, radius : float = 1, subdiv_x : int = 2**8, subdiv_y : int = 2**8, flip_normals = False):
    """ 
    Create a Sphere Model.
    
    Args: 
        :decl:
            | VertexLayout declaration
            | If None: the following is created: PosFloatNormalFloatTexCoord0Float 
            |       (if using texture images: this is the one that is required)
        :radius:
            | 1, optional
            | Radius of sphere
        :subdiv_x:
            | 256, optional
            | Number of subdivisions along sphere axis 
        :subdiv_y:
            | 256, optional
            | Number of subdivision along sphere circumference.
        :flip_normals:
            | False, optional
            | If True: flip the direction of the normals of the vertices.
    
    Returns:
        :Model:
            | Harfang Sphere Model
    """
    if decl is None:
        decl = hg.VertexLayout()
        decl.Begin().Add(hg.A_Position, 3, hg.AT_Float).Add(hg.A_Normal, 3, hg.AT_Float).Add(hg.A_TexCoord0, 2, hg.AT_Float).End()


    builder = hg.ModelBuilder() 
    
    fn = -1.0 if flip_normals else 1.0
    fn3 = hg.Vec3(fn,fn,fn) # shorthand for later
    TwoPi = 2 * pi
    
    i_top = builder.AddVertex(hg.MakeVertex(hg.Vec3(0.,radius,0.), hg.Vec3(0., fn, 0.), hg.Vec2(0,0)))
    i_bottom = builder.AddVertex(hg.MakeVertex(hg.Vec3(0.,-radius,0.), hg.Vec3(0., -fn, 0.), hg.Vec2(0,1)))
    
    n = 1
    ref = np.empty(subdiv_y + n,dtype = np.uint16)
    old_ref = None
    for s in range(0, int(subdiv_x - 2 + 1)): # -2 because already have top and bottom
        t = (s + 1) / (subdiv_x + 1.)
        a = t * pi
        
        section_y = cos(a) * radius
        section_radius = sin(a) * radius 
        
        v = hg.Vec3(section_radius, section_y, 0.)
        ref[0] = i =  builder.AddVertex(hg.MakeVertex(v, fn3*hg.Normalize(v), hg.Vec2(0,t)))
     
        for c in range(1, int(subdiv_y + n)):
            c_a = float(c) * TwoPi / subdiv_y
            
            v = hg.Vec3(cos(c_a) * section_radius, section_y, sin(c_a) * section_radius)
            j = builder.AddVertex(hg.MakeVertex(v, fn3*hg.Normalize(v), hg.Vec2(c/(subdiv_y + n),t)))

            if (s == 0): # top section
                builder.AddTriangle(i_top, i, j)
            
            else: # middle section 
                array32 = [c_uint32(x).value for x in [old_ref[c], old_ref[c - 1], i, j]]
                builder.AddPolygon(hg.uint32_tList(array32))
                
            if (s == subdiv_x): # bottom section 
                builder.AddTriangle(i, i_bottom, j)
                
                
            ref[c] = i = j
            
        old_ref = ref.copy()
        
    builder.EndList(0)
    
    return builder.MakeModel(decl)

#------------------------------------------------------------------------------

def CreatePlaneModel(decl : hg.VertexLayout = None, width : float = 1, height : float = 1, subdiv_x : int = 2**8, subdiv_y : int = 2**8, flip_normals = False):
    """ 
    Create a Plane (Quad) Model.
    
    Args: 
        :decl:
            | VertexLayout declaration
            | If None: the following is created: PosFloatNormalFloatTexCoord0Float 
            |       (if using texture images: this is the one that is required)
        :width:
            | 1, optional
            | Width of plane
        :height:
            | 1, optional
            | height of plane
        :subdiv_x:
            | 256, optional
            | Number of subdivisions along plane height
        :subdiv_y:
            | 256, optional
            | Number of subdivision along plane width.
        :flip_normals:
            | False, optional
            | If True: flip the direction of the normals of the vertices.
    
    Returns:
        :Model:
            | Harfang Plane Model
    """
    if decl is None:
        decl = hg.VertexLayout()
        decl.Begin().Add(hg.A_Position, 3, hg.AT_Float).Add(hg.A_Normal, 3, hg.AT_Float).Add(hg.A_TexCoord0, 2, hg.AT_Float).End()

    builder = hg.ModelBuilder() 
    
    fn = -1.0 if flip_normals else 1.0
    fn3 = hg.Vec3(fn,fn,fn) # shorthand for later
      
    n = 1
    ref = np.empty(subdiv_y + n,dtype = np.uint16)
    old_ref = None
    for s in range(0, subdiv_x + 1): 
        t = (s) / (subdiv_x) 
        
        section_y = height*(t - 0.5) 
        section_x = 0
        
        v = hg.Vec3(section_x, section_y, 0)
        ref[0] = i =  builder.AddVertex(hg.MakeVertex(v, fn3*hg.Normalize(v), hg.Vec2(0,1-t)))
     
        for c in range(1, subdiv_y + n):
            c_a = (float(c) / subdiv_y - 0.5)*width
            
            v = hg.Vec3(c_a , section_y, 0)
            j = builder.AddVertex(hg.MakeVertex(v, fn3*hg.Normalize(v), hg.Vec2(c/(subdiv_y + n),1-t)))

            if s == 0: 
                pass
            else:
                array32 = [c_uint32(x).value for x in [old_ref[c], old_ref[c - 1], i, j]]
                builder.AddPolygon(hg.uint32_tList(array32))
                   
            ref[c] = i = j
            
        old_ref = ref.copy()
        
    builder.EndList(0)
    
    return builder.MakeModel(decl)
#==============================================================================

def create_material(prg_ref, res, ubc = None, orm = None, slf = None, tex = None,
                    blend_mode = hg.BM_Opaque, 
                    faceculling = hg.FC_CounterClockwise
                    ):
    """ 
    Create a Harfang material with specified color and texture properties.
    
    Args:
        :prg_ref:
            | shader program from assets (ref)
        :res:
            | resources
        :ubc:
            | uBaseOpacityColor
        :orm:
            | uOcclusionRoughnessMetalnessColor
        :slf:
            | uSelfColor
        :tex:
            | uSelfMap texture (if not None: any color input is ignored !)
        :blendmode:
            | hg.BM_Opaque, optional
            | Blend mode
        :faceculling:
            | hg.FC_CounterClockwise
            | Sets face culling (hg.FC_CounterClockwise, hg.FC_Clockwise, hg.FC_Disabled)
            
    Returns:
        :mat:
            | Harfang material (note that material program variant has been updated
            |                   accordingly; see: hg.UpdateMaterialPipelineProgramVariant)
    """
    mat = hg.Material()
    hg.SetMaterialProgram(mat, prg_ref)
    if tex is None: 
        if ubc is not None: hg.SetMaterialValue(mat, "uBaseOpacityColor", ubc)
        if slf is not None: hg.SetMaterialValue(mat, "uSelfColor", slf)
        if orm is not None: hg.SetMaterialValue(mat, "uOcclusionRoughnessMetalnessColor", orm)
    else:
        tex = hg.LoadTextureFromFile(tex, _TEXTURE_FLAGS, res)
        hg.SetMaterialTexture(mat,"uSelfMap", tex, 4)
    hg.UpdateMaterialPipelineProgramVariant(mat, res)
    hg.SetMaterialBlendMode(mat, blend_mode)
    hg.SetMaterialFaceCulling(mat, faceculling)
    return mat

def update_material_texture(node, res, tex, mat_idx = 0, name = "uSelfMap", stage = 4, texListPreloaded = None): 
    """ 
    Update the texture of a Harfang material.
    
    Args:
        :node:
            | Node to which material belongs
        :res:
            | Pipeline resources
        :tex:
            | New texture
        :mat_idx:
            | 0, optional
            | index of material in material table of object
        :name:
            | "uSelfMap", optional
            | name of material type (depends on shader used; the default is for the pbr shader)
       :stage:
            | 4, optional
            | Render stage: depends on features, shader, ... (see "writing a pipeline shader" in Harfang documentation)
       :texListPreloaded:
            | None, optional 
            | List with preloaded textures (to speed up texture update as it doesn't need to be read from file anymore while looping over frames)
                                            
    Returns:
        :mat:
            | Harfang material (note that material program variant has been updated
            |                   accordingly; see: hg.UpdateMaterialPipelineProgramVariant)

    """
    if isinstance(tex, str) & ((texListPreloaded is None) | (isinstance(texListPreloaded, list))):
        tex = hg.LoadTextureFromFile(tex, _TEXTURE_FLAGS, res)
        mat = node.GetObject().GetMaterial(mat_idx)
        hg.SetMaterialTexture(mat, name, tex, stage)
    # elif isinstance(tex, int): 
    #     # node.GetObject().SetMaterialCount(tex) # ????????
    #     # mat = node.GetObject().GetMaterial(tex)
    elif isinstance(tex, (int,str)) & (texListPreloaded is not None): 
        texPreloaded = texListPreloaded[tex] # indexing in List or Dict !
        mat = node.GetObject().GetMaterial(mat_idx)
        hg.SetMaterialTexture(mat, name, texPreloaded, stage)
    else:
        raise Exception('Tex {} could not be updated. (texListPreloaded was {})'.format(tex, texListPreloaded ))
    hg.UpdateMaterialPipelineProgramVariant(mat, res)
    return mat

def makeColorTex(color, texHeight = 100, texWidth = 100, save = None):
    """ 
    Make a full single-color texture.
    
    Args:
        :color:
            | uint8 RGB(A; ignored) color
        :texHeight,texWidth:
            | Height and width of texture
        :save:
            | None, optional
            | File path to save texture to.
            | If not None: save texture in supplied filepath.
            
    Returns:
        :text:
            | numpy ndarray with RGB texture.
    """
    
    if texWidth is None: texWidth = texHeight 
    tex = (np.ones((texHeight,texWidth,3))*color[None,...][...,:3]).astype(dtype = np.uint8)
    if save is not None: imsave(save,tex)
    return tex
    
def split_SingleSphericalTex(file, left_layout_pos = 'bottom'):
    """ 
    Split Image into left eye and right eye subimages 
    
    Args:
        :file:
            | Image file path
        :left_layout_pos:
            | Position of left eye sub-image in image specified in file.
            | options: 'bottom', 'top', 'left', 'right', None
            | If None: there is no left and right subimage in 
            |          the image specified in filePath -> don't split
            
    Returns:
        :file_L, file_R: 
            | filepaths to left and right eye sub-images 
            | (each indicated respectively by '_L', '_R' appended to the filename.)
    """
    if '_L.' not in file:
        file_L, file_R = file[:-4]+'_L'+file[-4:], file[:-4]+'_R'+file[-4:]
        if not os.path.exists(file_L): # only split when _L doesn't exist yet.
            tex = imread(file)
            texHeight,texWidth = tex.shape[:2]
            if left_layout_pos == 'top':
                tex_L, tex_R = tex[:texHeight//2,...], tex[(texHeight//2):,...]
            elif left_layout_pos == 'bottom':
                tex_R, tex_L = tex[:texHeight//2,...], tex[(texHeight//2):,...]
            elif left_layout_pos == 'left':
                tex_L, tex_R = tex[:,:texWidth//2,...], tex[:,(texWidth//2):,...]
            elif left_layout_pos == 'right':
                tex_R, tex_L = tex[:,:texWidth//2,...], tex[:,(texWidth//2):,...]
            elif left_layout_pos is None:
                tex_L, tex_R = tex, tex
            else:
                tex_L, tex_R = None, None
            if tex_L is not None: imsave(file_L,tex_L)
            if tex_R is not None: imsave(file_R,tex_R)
    else:
        file_L, file_R = file[:-6]+'_L'+file[-4:], file[:-6]+'_R'+file[-4:]
    return [file_L, file_R]


# def generateCubemapFiles(file_L, file_R):
#     files, Ids = [file_L,file_R], ['_L','_R'] 
#     generated_files = []
#     for file,Id in zip(files,Ids):
#         #if Id in file: 
#         Id = ''
#         if not os.path.exists((file[:-4] + '_posx' + file[-4:])):
#             tmp = imread(file)
#             for _posx in ['_posx','_negx','_posy','_negy','_posz','_negz']:
#                 file_posX = (file[:-4] + Id + _posx + file[-4:])
#                 imsave(file_posX, tmp)
#                 generated_files.append(file_posX)
#     return generated_files
    
def _cleanupGeneratedFiles(generated_files, keep_list = []):
    """ Clean up files generated by viewer (keep_list: files to keep) """
    gf = copy.copy(generated_files)
    for file in generated_files: 
        if file is not None: 
            if file not in keep_list: 
                os.remove(file)
        gf.remove(file)
    return gf

def _getLeftRightFileNames(filepath, left_is_right = False):
    """ Get filenames of left and right subimages"""
    _left_strId = '_L'
    _right_strId = '_L' if left_is_right else '_R'
    if '_L.' not in filepath: 
        file_L = filepath[:-4] + _left_strId  + filepath[-4:]
        file_R = filepath[:-4] + _right_strId + filepath[-4:]
    else:
        file_L = filepath[:-6] + _left_strId  + filepath[-4:]
        file_R = filepath[:-6] + _right_strId + filepath[-4:]
    
#     if os.path.exists(filepath) and (not os.path.exists(file_L)): # use original for both
#         file_L, file_R = filepath, filepath
    
    return file_L, file_R

def _processEquiRectImagePath(filePath, left_is_right, left_layout_pos):
    """ Process the path of the rectangular image.
    (splits image if no left ('_R') or right ('_L') sub-images are found in same folder as in filePath
     
     Args:
         :filePath:
             | Image file path
         :left_is_right:
             | If True: the image for the left eye is the same as for the right eye 
         :left_layout_pos:
             | Position of left eye sub-image in image specified in filePath.
             | options: 'bottom', 'top', 'left', 'right', None
             | If None: there is no left and right subimage in 
             |          the image specified in filePath -> don't split
     """
    if filePath is None:
        filePath = (None, None)
    elif isinstance(filePath,str): 
        file_L, file_R = _getLeftRightFileNames(filePath, left_is_right = left_is_right)
        if (not os.path.exists(file_L)) | (not os.path.exists(file_R)):
            file_L, file_R = split_SingleSphericalTex(filePath, left_layout_pos = left_layout_pos)
        filePath = (file_L, file_R)
    elif isinstance(filePath, list):
        filePath = tuple(filePath)
    elif isinstance(filePath, tuple):
        pass
    else:
        print(filePath)
        raise Exception('Only None, list, tuple or string input allowed for filePath')
    return filePath


#==============================================================================

class Shader:
    def __init__(self, resources, assetPath = 'core/shader/pbr.hps'):
        """ 
        Initialize / load a shader from assets.
        
        Args:
            :resources:
                | Resources; required to load shader.
            :assetPath:
                | 'core/shader/pbr.hps', optional
                | Asset path of shader.
        """
        # Load shader program
        self.assetPath = assetPath
        self.prgRef = hg.LoadPipelineProgramRefFromAssets(self.assetPath, resources, hg.GetForwardPipelineInfo())

class Scene:
    def __init__(self,
                 canvasColorI = [0,0,0,255],
                 ambientEnvColorI = [0, 0, 0, 0]):
        """
        Create a basic scene.
        
        Args:
            :canvasColorI:
                | [0,0,0,255], optional
                | uint8 RGBA color of canvas
            :ambientEnvColorI:
                | [0,0,0,0], optional
                | uint8 RGBA color of ambient environment light(ing)
        """
        self.canvasColorI = canvasColorI
        self.ambientEnvColorI = ambientEnvColorI
        
        self.scene = hg.Scene()
        self.scene.canvas.color = hg.ColorI(*self.canvasColorI)
        self.scene.environment.ambient = hg.ColorI(*self.ambientEnvColorI)

class Camera:
    def __init__(self, 
                 scene,
                 position = [0, 0, 0],
                 rotation = [0, 0, 0],
                 zNear = 0.01, 
                 zFar = 5000,
                 fov = 60):
        """ 
        Define / create a camera object
        
        Args:
            :scene:
                | scene object in which to create the camera.
            :position:
                | [0,0,0], optional
                | Position of camera 
            :rotation:
                | [0,0,0], optional
                | Euler rotation angles (°) of camera
            :zNear:
                | 0.01, optional
                | Distance of near plane of frustum.
            :zFar:
                | 5000,
                | Distance of far plane of frustum. 
            :fov:
                | 60, optional
                | Field-of-view of camera.
        """
        self.pos = position
        self.rot = rotation
        self.zNear = zNear 
        self.zFar = zFar
        self.fov = fov
        
        # Set camera position and orientation + create and add to scene:
        if not isinstance(self.pos, hg.Vec3): self.pos = hg.Vec3(*self.pos)
        if not isinstance(self.rot, hg.Vec3): self.rot  = hg.Deg3(*self.rot)
        self.mtx = hg.TransformationMat4(self.pos, self.rot)
        self.cam = hg.CreateCamera(scene, self.mtx, self.zNear, self.zFar, hg.Deg(self.fov))
  

class Material:
    def __init__(self, shader_prgRef, resources,
                 uSelfMapTexture = None,
                 uSelfMapTextureListPreloaded = None,
                 uBaseOpacityColor = [1.0,1.0,1.0,1.0],
                 uSelfColor = [1.0,1.0,1.0,1.0],
                 uOcclusionRoughnessMetalnessColor = [0.0, 0.0, 0.0, 1.0],
                 blend_mode = hg.BM_Opaque, 
                 faceculling = hg.FC_CounterClockwise):
        """ 
        Define a Harfang material with specified color and texture properties.
        
        Args:
            :ubc:
                | uBaseOpacityColor
            :orm:
                | uOcclusionRoughnessMetalnessColor
            :slf:
                | uSelfColor
            :prg_ref:
                | shader program from assets (ref)
            :res:
                | resources
            :tex:
                | uSelfMap texture (if not None: any color input is ignored !)
            :blendmode:
                | hg.BM_Opaque, optional
                | Blend mode
            :faceculling:
                | hg.FC_CounterClockwise
                | Sets face culling (hg.FC_CounterClockwise, hg.FC_Clockwise, hg.FC_Disabled)
                
        """
        
        self.shader_prgRef, self.resources = shader_prgRef, resources
        
        self.uSelfMapTexture = uSelfMapTexture 
        self.uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded
        self.uSelfColor = uSelfColor
        self.uBaseOpacityColor =  uBaseOpacityColor 
        self.uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor
        self.blend_mode = blend_mode
        self.faceculling = faceculling
        if not isinstance(self.uBaseOpacityColor, hg.Vec4): self.uBaseOpacityColor = hg.Vec4(*self.uBaseOpacityColor)
        if not isinstance(self.uSelfColor, hg.Vec4): self.uSelfColor = hg.Vec4(*self.uSelfColor)
        if not isinstance(self.uOcclusionRoughnessMetalnessColor, hg.Vec4): self.uOcclusionRoughnessMetalnessColor = hg.Vec4(*self.uOcclusionRoughnessMetalnessColor) 
        
        self.mat = create_material(shader_prgRef, resources, 
                                   ubc = self.uBaseOpacityColor, 
                                   slf = self.uSelfColor, 
                                   orm = self.uOcclusionRoughnessMetalnessColor, 
                                   tex = self.uSelfMapTexture,
                                   blend_mode = self.blend_mode, 
                                   faceculling = self.faceculling)
        
        self.texList = None
        
    def createMaterial(self,
                       uSelfMapTexture = None,
                       uBaseOpacityColor = None,
                       uSelfColor = None,
                       uOcclusionRoughnessMetalnessColor = None,
                       blend_mode = None, 
                       faceculling = None):
        """ 
        Create a Harfang material with specified color and texture properties.
        
        Args:
            :ubc:
                | uBaseOpacityColor
            :orm:
                | uOcclusionRoughnessMetalnessColor
            :slf:
                | uSelfColor
            :tex:
                | uSelfMap texture (if not None: any color input is ignored !)
            :blendmode:
                | hg.BM_Opaque, optional
                | Blend mode
            :faceculling:
                | hg.FC_CounterClockwise
                | Sets face culling (hg.FC_CounterClockwise, hg.FC_Clockwise, hg.FC_Disabled)
        """
        if uSelfMapTexture is not None: self.uSelfMapTexture = uSelfMapTexture
        if uBaseOpacityColor is not None: self.uBaseOpacityColor = uBaseOpacityColor 
        if uSelfColor is not None: self.uSelfColor = uSelfColor
        if uOcclusionRoughnessMetalnessColor is not None: self.uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor
        if blend_mode is not None: self.blend_mode = blend_mode 
        if faceculling is not None: self.faceculling = faceculling
        self.mat = create_material(self.shader_prgRef, 
                                   self.resources, 
                                   ubc = self.uBaseOpacityColor, 
                                   slf = self.uSelfColor, 
                                   orm = self.uOcclusionRoughnessMetalnessColor, 
                                   tex = self.uSelfMapTexture,
                                   blend_mode = self.blend_mode,
                                   faceculling = self.faceculling)

    def LoadTexturesFromFiles(self, texFileList, return_type = list):
        """ 
        Load textures specified in texFileList (return_type is either a list or dict) 
        """
        texList = [] if return_type == list else {}
        for texFile in texFileList: 
            if return_type == list: 
                texList.append(hg.LoadTextureFromFile(texFile, _TEXTURE_FLAGS, self.resources))
            else:
                texList[texFile] = hg.LoadTextureFromFile(texFile, _TEXTURE_FLAGS, self.resources)
        self.texList = texList
        self.uSelfMapTextureListPreloaded = texList


class Screen(Material):
    def __init__(self, 
                 scene,  shader_prgRef, resources,
                 geometry = 'sphere', aspect_ratio = [19,16],
                 radius = 4,
                 subdiv_x = 2**8,
                 subdiv_y = 2**8,
                 uSelfMapTexture = None,
                 uSelfMapTextureListPreloaded = None,
                 uBaseOpacityColor = [1.0,1.0,1.0,1.0],
                 uSelfColor = [1.0,1.0,1.0,1.0],
                 uOcclusionRoughnessMetalnessColor = [0.0, 0.0, 0.0, 1.0],
                 blend_mode = hg.BM_Opaque,
                 position = [0,0,0],
                 rotation = [0,0,0]
                 ):
        """
        Define / create a screen to show (image/single-color) texture on.
        
        Args:
            :scene:
                | scene object in which to create the camera.
            :shader_prgRef:
                | shader program from assets (ref)
            :resources:
                | Pipeline resources
            :geometry:
                | 'sphere', optional
                | Geometry of the screen ('sphere','quad')
            :aspect_ratio:
                | [1,1], optional
                | [width, height] relative dimensions of screen 
                | For 'sphere': [1,1])
                | For 'quad': is used to determine width and height of quad from :radius: (=1/2*diagonal size)
            :radius:
                | 4, optional
                | Radius of sphere or 1/2*length of diagonal of quad. 
            :subdiv_x:
                | 256, optional
                | Number of subdivisions along sphere axis or along quad height
            :subdiv_y:
                | 256, optional
                | Number of subdivision along sphere circumference or along quad width
            :uBaseOpacityColor:
                | [1.0,1.0,1.0,1.0], optional
                | uBaseOpacityColor
            :uOcclusionRoughnessMetalnessColor:
                | [0.0, 0.0, 0.0, 1.0], optional
                | uOcclusionRoughnessMetalnessColor
            :uSelfColor:
                | [1.0,1.0,1.0,1.0], optional
                | uSelfColor
            :uSelfMapTexture:
                | None, optional
                | uSelfMap texture (if not None: any color input is ignored !)
            :uSelfMapTextureListPreloaded:
                | None, optional
                | List with preloaded textures (to speed up texture update as it doesn't need to be read from file anymore while looping over frames)
            :blend_mode:
                | hg.BM_Opaque, optional
                | Blend mode
            :position:
                | [0,0,0], optional
                | Position of screen
            :rotation:
                | [0,0,0], optional
                | Euler rotation angles (°) of screen                                        
        """
        
        super().__init__(shader_prgRef = shader_prgRef,
                         resources = resources,
                         uSelfMapTexture = uSelfMapTexture,
                         uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded,
                         uBaseOpacityColor = uBaseOpacityColor,
                         uSelfColor = uSelfColor,
                         uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor,
                         blend_mode = blend_mode,
                         faceculling = hg.FC_CounterClockwise
                         )
        
        self.scene = scene
        
        self.geometry = geometry 
        self.aspect_ratio = aspect_ratio
        self.radius = radius
        self.subdiv_x = subdiv_x
        self.subdiv_y = subdiv_y
        self.position = position 
        self.rotation = rotation
        
        if not isinstance(self.position, hg.Vec3): self.position = hg.Vec3(*self.position)
        if not isinstance(self.rotation, hg.Vec3): self.rotation = hg.Vec3(*[hg.Deg(x) for x in self.rotation])
                
        if geometry == "sphere": 
            self.width = self.radius 
            self.height = self.radius
            self.aspect_ratio = [1,1]
            
            self.mdl = CreateSphereModel(None, self.radius, self.subdiv_x, self.subdiv_y)
            self.ref = resources.AddModel('sphere', self.mdl)
        
        elif geometry == "quad":
    
            scale_factor = self.radius/(0.5*(self.aspect_ratio[0]**2 + self.aspect_ratio[1]**2)**0.5) # scale width, height with radius, but take radius as half-diagonal
            self.width = np.ceil(self.aspect_ratio[0] * scale_factor).astype(int)
            self.height = np.ceil(self.aspect_ratio[1] * scale_factor).astype(int)
            
            self.mdl = CreatePlaneModel(None, self.width, self.height, self.subdiv_x, self.subdiv_y)
            self.ref = resources.AddModel('quad', self.mdl)
        
        self.mtx = hg.TransformationMat4(self.position, self.rotation)
        self.screen = hg.CreateObject(scene, self.mtx, self.ref, [self.mat])

    def updateScreenMaterial(self,
                            uSelfMapTexture = None,
                            uSelfColor = None, 
                            uBaseOpacityColor = None,
                            uOcclusionRoughnessMetalnessColor = None,
                            blend_mode = None
                            ):
        """ 
        Update Screen Material
        
        Args:
            
            :uBaseOpacityColor:
                | None, optional
                | uBaseOpacityColor
            :uOcclusionRoughnessMetalnessColor:
                | None, optional
                | uOcclusionRoughnessMetalnessColor
            :uSelfColor:
                | None, optional
                | uSelfColor
            :uSelfMapTexture:
                | None, optional
                | uSelfMap texture (if not None: any color input is ignored !)
            :blend_mode:
                | None, optional
                | Blend mode
                    
        Note:
            * If None: defaults set at initialization are used.
        """
        if uSelfMapTexture is not None: self.uSelfMapTexture = uSelfMapTexture
        if uBaseOpacityColor is not None: self.uBaseOpacityColor = uBaseOpacityColor 
        if uSelfColor is not None: self.uSelfColor = uSelfColor
        if uOcclusionRoughnessMetalnessColor is not None: self.uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor
        if blend_mode is not None: self.blend_mode = blend_mode 
        
        self.createMaterial(uSelfMapTexture = self.uSelfMapTexture,
                            uSelfColor = self.uSelfColor, 
                            uBaseOpacityColor = self.uBaseOpacityColor,
                            uOcclusionRoughnessMetalnessColor = self.uOcclusionRoughnessMetalnessColor,
                            blend_mode = self.blend_mode,
                            faceculling = self.faceculling)
        
        self.screen.RemoveObject()
        self.screen = hg.CreateObject(self.scene, self.mtx, self.ref, self.mat)
    
    def updateScreenMaterialTexture(self, uSelfMapTexture = None, uSelfMapTextureListPreloaded = None):
        """ 
        Update the texture of the Harfang material.
        
        Args:
           :uSelfMapTexture:
               | New texture (string with filename)
           :uSelfMapTextureListPreloaded:
                | None, optional 
                | List with preloaded textures (to speed up texture update as it doesn't need to be read from file anymore while looping over frames)
        """
        if uSelfMapTexture is not None:
            
            if uSelfMapTextureListPreloaded is None:
                if self.texList is not None:
                    if len(self.texList) > 0: 
                        uSelfMapTextureListPreloaded = self.texList
            # else:
            #     self.texList = uSelfMapTextureListPreloaded
            self.mat = update_material_texture(self.screen, self.resources, uSelfMapTexture, mat_idx = 0, 
                                               name = "uSelfMap", stage = 4, texListPreloaded = uSelfMapTextureListPreloaded)
            self.uSelfMapTexture = uSelfMapTexture
    

class Eye:
    def __init__(self, 
                 eye, 
                 vrFlag = True, 
                 shader_assetPath = 'core/shader/pbr.hps',
                 scene_canvasColorI = [0,0,0,255],
                 scene_ambientEnvColorI = [0, 0, 0, 0],
                 cam_pos = [0, 0, 0],
                 cam_rot = [0, 0, 0],
                 cam_zNear = 0.01, 
                 cam_zFar = 100,
                 cam_fov = 60,
                 screen_geometry = 'sphere',
                 screen_aspectRatio = 1,
                 screen_radius = 10,
                 screen_subdiv_x = 2**8,
                 screen_subdiv_y = 2**8,
                 screen_uSelfMapTexture = None,
                 screen_uSelfMapTextureListPreloaded = None,
                 screen_uBaseOpacityColor = [1.0,1.0,1.0,1.0],
                 screen_uSelfColor = [1.0,1.0,1.0,0],
                 screen_uOcclusionRoughnessMetalnessColor = [0.5, 0.0, 0.0, 1.0],
                 screen_blend_mode = hg.BM_Opaque,
                 screen_pos = [0,0,0],
                 screen_rot = [0,0,0]
                 ):
        """
        Initialize pipeline, get pipeline resources, render_data, OpenVR 
        and create scene, shader, camera, material and screen for a specified eye.
        
        Args:
            :eye:
                | 0 (left) or 1 (right) indicator
            :vrFlag:
                | False, optional
                | If True: use VR head-mounted-display
            :oher args:
                | See __doc__ for Scene, Shader, Camera, Material and Screen classes.
        """
        self.eye = eye # 0: left, 1: right
        
        self.pipeline = hg.CreateForwardPipeline() 
        self.resources = hg.PipelineResources() 
        self.render_data = hg.SceneForwardPipelineRenderData()  # this object is used by the low-level scene rendering API to share view-independent data with both eyes
        
        # Create OpenVrFrameBuffer:
        self.vrFlag = vrFlag
        if self.vrFlag:
            if not hg.OpenVRInit():
                sys.exit()
            self.vr_fb = hg.OpenVRCreateEyeFrameBuffer(hg.OVRAA_MSAA4x)
        else:
            self.vr_fb = None
          
            
        self.shader = Shader(self.resources, assetPath = shader_assetPath)
            
        self._scene = Scene(canvasColorI = scene_canvasColorI, ambientEnvColorI = scene_ambientEnvColorI)
        self.scene = self._scene.scene 
        
        
        self.camera = Camera(self.scene, position = cam_pos, rotation = cam_rot, zNear = cam_zNear, zFar = cam_zFar, fov = cam_fov)
        
        
        self._screen = Screen(self.scene, self.shader.prgRef, self.resources, 
                             geometry = screen_geometry, 
                             aspect_ratio = screen_aspectRatio,
                             radius = screen_radius, 
                             subdiv_x = screen_subdiv_x, 
                             subdiv_y = screen_subdiv_y,
                             uSelfMapTexture = screen_uSelfMapTexture, 
                             uSelfMapTextureListPreloaded = screen_uSelfMapTextureListPreloaded,
                             uBaseOpacityColor = screen_uBaseOpacityColor,
                             uSelfColor = screen_uSelfColor, 
                             uOcclusionRoughnessMetalnessColor = screen_uOcclusionRoughnessMetalnessColor,
                             blend_mode = screen_blend_mode,
                             position = screen_pos, 
                             rotation = screen_rot
                             )
        self.screen = self._screen.screen
     
    def updateScreenMaterial(self, 
                       uSelfMapTexture = None, 
                       uBaseOpacityColor = None,
                       uSelfColor = None,
                       uOcclusionRoughnessMetalnessColor = None,
                       blend_mode = None
                       ):
        """ Update Screen Material (see Screen.updateScreenMaterial.__doc__) """
        self._screen.updateScreenMaterial(uSelfMapTexture = uSelfMapTexture,
                                    uBaseOpacityColor = uBaseOpacityColor,
                                    uSelfColor = uSelfColor, 
                                    uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor,
                                    blend_mode = blend_mode)
        self.screen = self._screen.screen
        
    def updateScreenMaterialTexture(self, uSelfMapTexture = None, uSelfMapTextureListPreloaded = None):
        """ Update Screen MaterialTexture (see Screen.updtateScreenMaterialTexture.__doc__) """        
        self._screen.updateScreenMaterialTexture(uSelfMapTexture = uSelfMapTexture,
                                                 uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded)

        self.screen = self._screen.screen
        
        

    def SceneForwardPipelinePassViewId_PrepareSceneForwardPipelineCommonRenderData(self, vid = 0):
        # vid = 0  # keep track of the next free view id
        self.vid = vid
        self.passId = hg.SceneForwardPipelinePassViewId()
    
        # Prepare view-independent render data once
        self.vid, self.passId = hg.PrepareSceneForwardPipelineCommonRenderData(self.vid, self.scene, self.render_data, self.pipeline, self.resources, self.passId)
    
    def PrepareSceneForwardPipelineViewDependentRenderData_SubmitSceneToForwardPipeline(self, vs, vr_eye_rect, isMainScreen = False):
        self.vid, self.passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(self.vid, vs, self.scene, self.render_data, self.pipeline, self.resources, self.passId)
        if isMainScreen == False: 
            self.vid, self.passId = hg.SubmitSceneToForwardPipeline(self.vid, self.scene, vr_eye_rect, vs, self.pipeline, self.render_data, self.resources, self.vr_fb.GetHandle() )
        else: 
            self.vid, self.passId = hg.SubmitSceneToForwardPipeline(self.vid, self.scene, vr_eye_rect, vs, self.pipeline, self.render_data, self.resources )

    def DestroyForwardPipeline(self):
        hg.DestroyForwardPipeline(self.pipeline)
        
        
        
def _processMaterialInput(x, equiRectImageLeftIsRight, equiRectImageLeftPos):
    """ Process material input (helper function)"""
    if x is None: 
        x = [(None,None)]
    elif isinstance(x, (int, float)):
        raise Exception('Input must be list, tuple, hg.Vec3, str, None')
    elif isinstance(x,tuple): # already L, R splitted
        x = [x]
    elif isinstance(x, hg.Vec4): # single input
        x = [(x,x)]
    elif isinstance(x, str): # single input
        x = [_processEquiRectImagePath(x, equiRectImageLeftIsRight, equiRectImageLeftPos)]
    elif isinstance(x, list): # multiple input?
        if isinstance(x[0], (int, float)) & (len(x) == 4):
            x = [(x,x)]
        elif isinstance(x[0], (int, float)) & (len(x) != 4):
            raise Exception('Color specification requires 4 numbers')
        else:
            X = []
            for xi in x:
                if isinstance(xi, hg.Vec4):
                    X.append((xi,xi)) 
                elif isinstance(xi, str):
                    X.append(_processEquiRectImagePath(xi, equiRectImageLeftIsRight, equiRectImageLeftPos))
                elif isinstance(xi, list):
                    X.append((xi,xi)) 
                elif isinstance(xi, tuple):
                    X.append(xi)
                elif xi is None:
                    X.append((None,None))
            x = X
    return x[0], x
            
                
            

class HmdStereoViewer:
    def __init__(self, vrFlag = False, vsync = True, multisample = 4, cam_fov = 60,
                 windowWidth = _SCREEN_TEX_WH[0], windowHeight = _SCREEN_TEX_WH[1], 
                 windowTitle = "Harfang3d - Stereoscopic Viewer",
                 mainScreenIdx = 0,
                 screen_geometry = 'sphere',
                 screen_aspectRatio = [1,1],
                 screen_radius = 10,
                 screen_subdiv_x = 2**8,
                 screen_subdiv_y = 2**8,
                 equiRectImageLeftPos = 'bottom',
                 equiRectImageLeftIsRight = False,
                 screen_uSelfMapTexture = [None],
                 screen_uSelfMapTextureListPreloaded = [None],
                 screen_uBaseOpacityColor = [[1.0,1.0,1.0,1.0]],
                 screen_uSelfColor = [[1.0,1.0,1.0,1.0]],
                 screen_uOcclusionRoughnessMetalnessColor = [[0.0, 0.0, 0.0, 1.0]],
                 screen_blend_mode = hg.BM_Opaque, 
                 screen_position = [0,0,0],
                 screen_rotation = [0,0,0],
                 pipeFcns = None):
        """
        Initialize Stereo HMD Viewer.
        
        Args:
            :vrFlag:
                | False, optional
                | If True: use VR Head-Mounted_display
            :vsync:
                | True, optional
                | If True: Turn V-Sync on.
            :multisample:
                | 4, optional
                | Multi-sample anti-aliasing (options: None, 4, 8)
            :windowWidth, windowHeight:
                | _SCREEN_TEX_WH[0], _SCREEN_TEX_WH[1] , optional
                | Width and height of main screen window.
            :windowTitle:
                | "Harfang3d - Stereoscopic Viewer", optional
                | String with title of main screen window
            :mainScreenIdx:
                | 0, optional
                | Index to eye (0: left, 1: right) to display on main screen window.
            :screen_geometry:
                | 'sphere', optional
                | Geometry of the screen ('sphere','quad')
            :screen_aspectRatio:
                | [1,1], optional
                | [width, height] relative dimensions of screen 
                | For 'sphere': [1,1])
                | For 'quad': is used to determine width and height of quad from :radius: (=1/2*diagonal size)
            :screen_radius:
                | 10, optional
                | Radius of sphere or 1/2*length of diagonal of quad. 
            :screen_subdiv_x:
                | 256, optional
                | Number of subdivisions along sphere axis or along quad height
            :screen_subdiv_y:
                | 256, optional
                | Number of subdivision along sphere circumference or along quad width
            :equiRectImageLeftPos:
                | 'bottom', optional
                | Specifier for where in the texture image the left sub-image is located.
                | options: 'bottom', 'top', 'left', 'right', None
                | If None: there are no separate left/right sub-images in the texture image file.
            :equiRectImageLeftIsRight:
                | False, optional
                | If True: the image for the left and right eye is the same.
            :screen_uBaseOpacityColor:
                | [1.0,1.0,1.0,1.0], optional
                | uBaseOpacityColor
            :screen_uOcclusionRoughnessMetalnessColor:
                | [0.0, 0.0, 0.0, 1.0], optional
                | uOcclusionRoughnessMetalnessColor
            :screen_uSelfColor:
                | [1.0,1.0,1.0,1.0], optional
                | uSelfColor
            :screen_uSelfMapTexture:
                | None or str or list, optional
                | If None: no texture image used -> color is determine by color input arguments 
                | If str: 
                |     - filename of texture 
                | If list:
                |     - list of filenames to image textures.
                |   (if not None: any color input is ignored !)
            :screen_uSelfMapTextureListPreloaded:
                | None, optional
                | List with preloaded textures (to speed up texture update as it doesn't need to be read from file anymore while looping over frames)
            :screen_blend_mode:
                | hg.BM_Opaque, optional
                | Blend mode
            :screen_position:
                | [0,0,0], optional
                | Position of screen
            :screen_rotation:
                | [0,0,0], optional
                | Euler rotation angles (°) of screen   
            :pipeFcns:
                | None, optional
                | A list of functions to pipe during while-loop in the .run() class method.
                | If None: use defaults. (These will loop through all images)
                | Default has 3: 
                |   - init (initialize a state list),
                |   - update (update texture),
                |   - action (do some action, e.g. measure; default fcn causes a delay of the update to next texture)
        """
        
        self.vrFlag = vrFlag
        self.multisample = multisample 
        self.vsync = vsync
        self.cam_fov = cam_fov
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight 
        self.windowTitle = windowTitle +' - [{:s} eye image]'.format(['Left','Right'][mainScreenIdx])
        self._assetsFolder = os.path.join(_PKG_PATH,"assets_compiled")
        
        self.init_main()
        
        self.screen_geometry = screen_geometry
        self.screen_aspectRatio = screen_aspectRatio
        self.screen_radius = screen_radius
        self.screen_subdiv_x = screen_subdiv_x
        self.screen_subdiv_y = screen_subdiv_y
        self.screen_position = screen_position
        self.screen_rotation = screen_rotation
        
        # set textures
        self.prmatin = lambda x: _processMaterialInput(x, equiRectImageLeftIsRight, equiRectImageLeftPos)
        self.equiRectImageLeftPos = equiRectImageLeftPos
        self.equiRectImageLeftIsRight = equiRectImageLeftIsRight
        if screen_uSelfMapTexture is None: screen_uSelfMapTexture = [None]
        self.set_texture(screen_uSelfMapTexture,
                         equiRectImageLeftPos = equiRectImageLeftPos, 
                         equiRectImageLeftIsRight = equiRectImageLeftIsRight,
                         screen_uSelfMapTextureListPreloaded = screen_uSelfMapTextureListPreloaded
                         )

        self.screen_uBaseOpacityColor, self.screen_uBaseOpacityColorList = self.prmatin(screen_uBaseOpacityColor)
        self.screen_uSelfColor,self.screen_uSelfColorList = self.prmatin(screen_uSelfColor)
        self.screen_uOcclusionRoughnessMetalnessColor, self.screen_uOcclusionRoughnessMetalnessColorList = self.prmatin(screen_uOcclusionRoughnessMetalnessColor)
        self.screen_blend_mode = screen_blend_mode 
        
        self.leftEyeScreen = Eye(0, vrFlag = self.vrFlag, cam_fov = self.cam_fov, 
                        screen_geometry = screen_geometry,
                        screen_aspectRatio = screen_aspectRatio,
                        screen_radius = screen_radius,
                        screen_subdiv_x = screen_subdiv_x,
                        screen_subdiv_y = screen_subdiv_y,
                        screen_uBaseOpacityColor = self.screen_uBaseOpacityColor[0],
                        screen_uSelfColor = self.screen_uSelfColor[0],
                        screen_uOcclusionRoughnessMetalnessColor = self.screen_uOcclusionRoughnessMetalnessColor[0],
                        screen_uSelfMapTexture = self.screen_uSelfMapTexture[0],
                        screen_uSelfMapTextureListPreloaded = self.screen_uSelfMapTextureListPreloaded[0],
                        screen_blend_mode = self.screen_blend_mode,
                        screen_pos = self.screen_position,
                        screen_rot = self.screen_rotation)
        self.rightEyeScreen = Eye(1, vrFlag = self.vrFlag, cam_fov = self.cam_fov, 
                         screen_geometry = screen_geometry,
                         screen_aspectRatio = screen_aspectRatio,
                         screen_radius = screen_radius,
                         screen_subdiv_x = screen_subdiv_x,
                         screen_subdiv_y = screen_subdiv_y,
                         screen_uBaseOpacityColor = self.screen_uBaseOpacityColor[1],
                         screen_uSelfColor = self.screen_uSelfColor[1],
                         screen_uOcclusionRoughnessMetalnessColor = self.screen_uOcclusionRoughnessMetalnessColor[1],
                         screen_uSelfMapTexture = self.screen_uSelfMapTexture[1],
                         screen_uSelfMapTextureListPreloaded = self.screen_uSelfMapTextureListPreloaded[1],
                         screen_blend_mode = self.screen_blend_mode,
                         screen_pos = self.screen_position,
                         screen_rot = self.screen_rotation)
        
        self.mainScreenIdx = mainScreenIdx
        self.mainScreen = self.leftEyeScreen if (self.mainScreenIdx == 0) else self.rightEyeScreen
        self.mainScreen.scene.SetCurrentCamera(self.mainScreen.camera.cam) 
        
        self.keyboard = hg.Keyboard()
        self.mouse = hg.Mouse()
        
        self.frameCounter = 0
        
        self.pipeFcns = pipeFcns
        self.generate_defaultPipeFcns()
        
    def set_texture(self, screen_uSelfMapTexture,
                    equiRectImageLeftPos = None, 
                    equiRectImageLeftIsRight = None,
                    screen_uSelfMapTextureListPreloaded = None
                    ):
        
        if equiRectImageLeftPos is not None: self.equiRectImageLeftPos = equiRectImageLeftPos
        if equiRectImageLeftIsRight is not None: self.equiRectImageLeftIsRight = equiRectImageLeftIsRight
   
        #prmatin = lambda x: _processMaterialInput(x, equiRectImageLeftIsRight, equiRectImageLeftPos)
        if screen_uSelfMapTexture is not None: 
            self.screen_uSelfMapTexture, self.screen_uSelfMapTextureList = self.prmatin(screen_uSelfMapTexture)
        
        if screen_uSelfMapTextureListPreloaded is not None: 
            _, self.screen_uSelfMapTextureListPreloaded = self.prmatin(screen_uSelfMapTextureListPreloaded)
        
            # split in left & right textures
            tmpL, tmpR = [], []
            for tmap in self.screen_uSelfMapTextureListPreloaded:
                tmpL.append(tmap[0])
                tmpR.append(tmap[1])
            self.screen_uSelfMapTextureListPreloaded = (tmpL,tmpR)
    
    def init_main(self):
        """ Initialize Input and Window, add folder with compiled assets"""
        
        if os.path.exists(self._assetsFolder):
            
            hg.InputInit()
            hg.WindowSystemInit()
            RF_VSync = hg.RF_VSync if self.vsync else 0
            RF_MSAA = hg.RF_MSAA4X if (self.multisample == 4) else (hg.RF_MSAA8X if (self.multisample == 8) else 0) 
            self.window = hg.RenderInit(self.windowTitle, self.windowWidth, self.windowHeight, RF_VSync | hg.RF_MSAA4X)

            hg.AddAssetsFolder(self._assetsFolder)
            
        else:
            raise Exception ('Compiled assets folder "{:s}" not found.\nCheck path or create it by:\n 1)downloading and unzipping "AssetC" from https://dev.harfang3d.com/releases/;\n 2) then dragging the "assets" folder in this toolbox onto the assetc.exe file in the unzipped folder'.format(self._assetsFolder))
        
                    
    def shutdown(self):
        """ Shutdown Pipelines for left and right eyes, Shutdown Render and destroy Window"""
        self.leftEyeScreen.DestroyForwardPipeline()
        self.rightEyeScreen.DestroyForwardPipeline()
        hg.RenderShutdown()
        hg.DestroyWindow(self.window)
       
    
        
    def updateScreenMaterial(self, 
                       uSelfMapTexture = None, 
                       equiRectImageLeftIsRight = None, 
                       equiRectImageLeftPos = None,
                       uBaseOpacityColor = None,
                       uSelfColor = None,
                       uOcclusionRoughnessMetalnessColor = None,
                       blend_mode = None
                       ):
        """ 
        Update Screen Material
        
        Args:
            :uSelfMapTexture:
                | None, optional
                | uSelfMap texture (if not None: any color input is ignored !)
            :equiRectImageLeftPos:
                | 'bottom', optional
                | Specifier for where in the texture image the left sub-image is located.
                | options: 'bottom', 'top', 'left', 'right', None
                | If None: there are no separate left/right sub-images in the texture image file.
            :equiRectImageLeftIsRight:
                | False, optional
                | If True: the image for the left and right eye is the same.
            :uBaseOpacityColor:
                | None, optional
                | uBaseOpacityColor
            :uOcclusionRoughnessMetalnessColor:
                | None, optional
                | uOcclusionRoughnessMetalnessColor
            :uSelfColor:
                | None, optional
                | uSelfColor
            :blend_mode:
                | None, optional
                | Blend mode
                    
        Note:
            * If None: defaults set at initialization are used.
        """
        if equiRectImageLeftIsRight is not None: self.equiRectImageLeftIsRight = equiRectImageLeftIsRight 
        if equiRectImageLeftPos is not None: self.equiRectImageLeftPos = equiRectImageLeftPos
        uSelfMapTextures = _processEquiRectImagePath(uSelfMapTexture, self.equiRectImageLeftIsRight, self.equiRectImageLeftPos)
        if uSelfMapTextures[0] is not None: 
            if ('_L.' in uSelfMapTexture) or ('_R.' in uSelfMapTexture): self.screen_uSelfMapTexture = uSelfMapTexture
        if uBaseOpacityColor is not None: self.screen_uBaseOpacityColor = uBaseOpacityColor
        if uSelfColor is not None: self.screen_uSelfColor = uSelfColor
        if uOcclusionRoughnessMetalnessColor is not None: self.screen_uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor
        uBaseOpacityColor = (uBaseOpacityColor, uBaseOpacityColor) if not isinstance(self.screen_uBaseOpacityColor, tuple) else self.screen_uBaseOpacityColor
        uSelfColor = (uSelfColor, uSelfColor) if not isinstance(self.screen_uSelfColor, tuple) else self.screen_uSelfColor
        uOcclusionRoughnessMetalnessColor = (uOcclusionRoughnessMetalnessColor, uOcclusionRoughnessMetalnessColor) if not isinstance(self.screen_uOcclusionRoughnessMetalnessColor, tuple) else self.screen_uOcclusionRoughnessMetalnessColor 
        if blend_mode is not None: self.screen_blend_mode = blend_mode
        self.leftEyeScreen.updateScreenMaterial(uSelfMapTexture = uSelfMapTextures[0],
                                 uBaseOpacityColor = uBaseOpacityColor[0],
                                 uSelfColor = uSelfColor[0], 
                                 uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor[0],
                                 blend_mode = self.screen_blend_mode
                                 )
        self.rightEyeScreen.updateScreenMaterial(uSelfMapTexture = uSelfMapTextures[1],
                                 uBaseOpacityColor = uBaseOpacityColor[1],
                                 uSelfColor = uSelfColor[1], 
                                 uOcclusionRoughnessMetalnessColor = uOcclusionRoughnessMetalnessColor[1],
                                 blend_mode = self.screen_blend_mode
                                 )
        self.mainScreen = self.leftEyeScreen if (self.mainScreenIdx == 0) else self.rightEyeScreen
        
        
    def updateScreenMaterialTexture(self, 
                                    uSelfMapTexture = None, 
                                    equiRectImageLeftIsRight = None, 
                                    equiRectImageLeftPos = None,
                                    uSelfMapTextureListPreloaded = None,
                                    ):
        """ 
        Update the texture of the Harfang material.
        
        Args:
            :uSelfMapTexture:
                | New texture (string with filename)
            :equiRectImageLeftPos:
                | 'bottom', optional
                | Specifier for where in the texture image the left sub-image is located.
                | options: 'bottom', 'top', 'left', 'right', None
                | If None: there are no separate left/right sub-images in the texture image file.
            :equiRectImageLeftIsRight:
                | False, optional
                | If True: the image for the left and right eye is the same.
            :uSelfMapTextureListPreloaded:
                 | None, optional 
                 | List with preloaded textures (to speed up texture update as it doesn't need to be read from file anymore while looping over frames)
        """
        if equiRectImageLeftIsRight is not None: self.equiRectImageLeftIsRight = equiRectImageLeftIsRight 
        if equiRectImageLeftPos is not None: self.equiRectImageLeftPos = equiRectImageLeftPos
        uSelfMapTextures = _processEquiRectImagePath(uSelfMapTexture, self.equiRectImageLeftIsRight, self.equiRectImageLeftPos)

        if uSelfMapTextures[0] is not None: 
            if ('_L.' in uSelfMapTexture) or ('_R.' in uSelfMapTexture): self.screen_uSelfMapTexture = uSelfMapTexture

        if uSelfMapTextureListPreloaded is not None: 
            self.screen_uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded # must be supplied for both left and right !!
        uSelfMapTextureListPreloaded = (uSelfMapTextureListPreloaded, uSelfMapTextureListPreloaded) if not isinstance(self.screen_uSelfMapTextureListPreloaded,tuple) else self.screen_uSelfMapTextureListPreloaded

        self.leftEyeScreen.updateScreenMaterialTexture(uSelfMapTexture = uSelfMapTextures[0], uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded[0])
        self.rightEyeScreen.updateScreenMaterialTexture(uSelfMapTexture = uSelfMapTextures[1], uSelfMapTextureListPreloaded = uSelfMapTextureListPreloaded[1])
        self.mainScreen = self.leftEyeScreen if (self.mainScreenIdx == 0) else self.rightEyeScreen

        
    def resetFrameNumber(self):
        """ Reset the frame number """
        self.frameNumber = 0
    
    def getFrameNumber(self):
        """ Get the current frame number """
        return self.frameNumber
    
    def display(self):
        """ Display the texture (first one from list, use run() to loop through all of them) """
        self.resetFrameNumber()
        
        if (self.screen_uSelfMapTextureList[0][0] is not None) & (self.screen_uSelfMapTextureList[0][1] is not None):
            texL, texR = self.screen_uSelfMapTextureList[0][0], self.screen_uSelfMapTextureList[0][1] # get texture fileNames
            self.leftEyeScreen.updateScreenMaterialTexture( uSelfMapTexture = texL )
            self.rightEyeScreen.updateScreenMaterialTexture( uSelfMapTexture = texR )

        while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(self.window):
            self.frame()
        self.shutdown()
        
        
    def run(self, pipeFcns = None, pipeFcnsUpdate = None, only_once = False, 
            u_delay = None, a_delay = None, autoShutdown = True):
        """
        Run through all textures specified at initialization (and do some action) .
        
        Args:
            :pipeFcns:
                | None, optional
                | list of piped functions, one executed after the other
                | If None: use the defaults. This will cause all textures 
                |   specified at initialization to be shown one after the other, with
                |   delay time set by :delay:. 
                | If not None: use this set of user-defined pipeFcns (see code for example use)
            :pipeFcnsUpdate:
                | None, optional
                | Use this list or dictionary to update the pipeFcns specified by :pipeFcns:
                | This exists to keep e.g. the defaults but only change the 'action' part, e.g. to 
                | do a measurement. 
            :only_once:
                | False, optional
                | If True: loop through the set of textures once and then stop and shutdown.
            :u_delay:
                | None, optional
                | Delay in seconds for the update function in the pipeFcns. 
                | This delays the initialization of the action function after 
                | an update of the texture (e.g. to give some time display the update on the HMD)
                | If None: use whatever is set in the (default) pipeFcns update function.
                | Else override delay if update function as such a kwarg!
            :a_delay:
                | None, optional
                | Delay in seconds for the action function in the pipeFcns.
                | This delays the update to the next texture after the action 
                | has been started (e.g. to simulate some action duration)
                | If None: use whatever is set in the (default) pipeFcns action function.
                | Else override delay if action function as such a kwarg!
                
        """
        if pipeFcns is None: pipeFcns = self.pipeFcnsDef
        if pipeFcnsUpdate is not None: 
            for i, pipeFcnUpdate in enumerate(pipeFcnsUpdate):
                if pipeFcnUpdate is not None: 
                    if isinstance(pipeFcnUpdate,(tuple,list)):
                        if pipeFcnUpdate[0] is None: 
                            temp = [pipeFcns[i][0], pipeFcnUpdate[1]]
                        else:
                            temp = pipeFcnUpdate 
                    elif isinstance(pipeFcnUpdate,dict):
                        temp = [pipeFcns[i][0], pipeFcnUpdate]
                    elif isinstance(pipeFcnUpdate,(tuple,list)):
                        temp = pipeFcnUpdate
                    pipeFcns[i] = temp
        
        # Update delays in pipeFcns by those from kwargs:
        if (u_delay is not None) | (a_delay is not None):
            for i, pipeFcn in enumerate(pipeFcns): 
                if (i>=1) and ('delay' in pipeFcns[i][1]['check'][1].keys()):
                    delay = u_delay if (i == 1) else a_delay
                    if delay is not None: pipeFcns[i][1]['check'][1]['delay'] = delay

        self.resetFrameNumber()
        out = None
        self.run_only_once = only_once
        while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(self.window):
            
            
            if pipeFcns is not None: # execute a pipeline of functions:
                if len(pipeFcns)>0:
                    for fcnIdx, (pipeFcn, kwargs) in enumerate(pipeFcns): 
                        # print('in : ', self.frameNumber, fcnIdx, out) # code check
                        out = pipeFcn(self, self.frameNumber, out, **kwargs)
                        # print('out: ', self.frameNumber, fcnIdx, out) # code check
                    # if self.frameNumber >= 150: # code check
                    #     break                   # code check
             
            if isinstance(out[2],str) and (out[2] == 'break'): 
                break
            self.frame()
            
        if autoShutdown:    
            self.shutdown()
        

    def frame(self):        
        """ Run everything required to update a frame """
        dt = hg.TickClock()
        self.keyboard.Update()
        self.mouse.Update()
    
        hg.FpsController(self.keyboard, self.mouse, self.mainScreen.camera.pos, self.mainScreen.camera.rot, 20 if self.keyboard.Down(hg.K_LShift) else 8, dt)
    
        self.mainScreen.camera.cam.GetTransform().SetPos(self.mainScreen.camera.pos)
        self.mainScreen.camera.cam.GetTransform().SetRot(self.mainScreen.camera.rot)
    
        self.leftEyeScreen.scene.Update(dt)
        self.rightEyeScreen.scene.Update(dt)
        
        
        self.leftEyeScreen.SceneForwardPipelinePassViewId_PrepareSceneForwardPipelineCommonRenderData(vid = 0)
         
        self.rightEyeScreen.SceneForwardPipelinePassViewId_PrepareSceneForwardPipelineCommonRenderData(vid = 1)
        
        if (self.leftEyeScreen.vr_fb is not None) & (self.rightEyeScreen.vr_fb is not None):
            actor_body_mtx = hg.TransformationMat4(self.mainScreen.camera.pos, self.mainScreen.camera.rot)
            vr_state = hg.OpenVRGetState(actor_body_mtx, 0.01, 100)
            vs_left, vs_right = hg.OpenVRStateToViewState(vr_state)
            
            vr_eye_rect = hg.IntRect(0, 0, vr_state.width, vr_state.height)

            # Prepare the left eye render data then draw to its framebuffer
            self.leftEyeScreen.PrepareSceneForwardPipelineViewDependentRenderData_SubmitSceneToForwardPipeline(vs_left, vr_eye_rect)
            
            # Prepare the right eye render data then draw to its framebuffer
            self.rightEyeScreen.PrepareSceneForwardPipelineViewDependentRenderData_SubmitSceneToForwardPipeline(vs_right, vr_eye_rect)
            
    
        # Main screen:
        vs_main = self.mainScreen._scene.scene.ComputeCurrentCameraViewState(hg.ComputeAspectRatioX(self.windowWidth, self.windowHeight))
        self.mainScreen.PrepareSceneForwardPipelineViewDependentRenderData_SubmitSceneToForwardPipeline(vs_main, hg.IntRect(0, 0, self.windowWidth, self.windowHeight), isMainScreen = True)
        
        
        hg.Frame()
        if (self.leftEyeScreen.vr_fb is not None) & (self.rightEyeScreen.vr_fb is not None):
            hg.OpenVRSubmitFrame(self.leftEyeScreen.vr_fb,self.rightEyeScreen.vr_fb)
        
        hg.UpdateWindow(self.window)

        self.frameNumber += 1
            
    def generate_defaultPipeFcns(self, pipeFcnDef = None):
        """ Generate default pipeline functions (if pipeFcnDef not None: use these) """
        if pipeFcnDef is None:
            #==============================================================================
            # Execute some things during each frame -> control flow functions
    
            #------------------------------------------------------------------------------
            # Pipe function definitions
            #------------------------------------------------------------------------------
            # Interface: 
            #    (fcn, kwargs) 
            #       with fcn(self, frameNumber, output_previous_pipe_fcn, pipe_fcn_kwargs)
            #           with everything a tuple with elements: 
            #             - windowProps, keyboard/mouse, mainScreen, leftEyeScreen, rightEyescreen
            #------------------------------------------------------------------------------ 
    
            # Define pipe_fcn 0:
            def initWrapper(self, frameNumber, out, action = None):
                """ Initialize the state array"""
                if out is None:
                    return action[0](self, frameNumber, out, **action[1])     
                else:
                    return out  
                
            # Define pipe_fcn 1:   
            def updateWrapper(self, frameNumber, out, action = None, check = None):
                """ Wrapper function (cfr. control flow) around texture update function"""
                if out is None: return out
                if out[1][0]: # if action finished set new stimulus
                    out[0][1] = action[0](self, frameNumber, out, **action[1])

                else:
                    out[0][0] = check[0](self, frameNumber, out, **check[1])   
                    if out[0][0]: out[0][1] = 1e100
                return out
    
            # Define pipe_fcn 2
            def actionWrapper(self, frameNumber, out, action = None,  check = None):
                """ Wrapper function (cfr. control flow) around texture action function"""
                if out is None: return out
                if out[0][0]: # -> fcn1 finished
                    out[1][0] = False # action is only about to start (so not yet finished) -> don't change tex !
                    out[0][0] = False # reset texReady flag, next tex is not ready yet as we're still doing something with the current one
                    out[1][1] = action[0](self, frameNumber, out, **action[1]) # Normaly: this is where the start of the action (e.g. measurement) happens
                    
                else:
                    out[1][0] = check[0](self, frameNumber, out, **check[1]) # ? action finished ?
                    if out[1][0]:
                        out[2] += 1 # move to next tex
                        out[0][0] = False # reset texReady flag
                        out[0][1] = 1e100
                        out[1][1] = 1e100
                        if self.run_only_once: 
                            if out[2] == self.n_stimuli:
                                out[2] = 'break'
                 
                return out
    
            def i_action(self, frameNumber, out, **kwargs):
                """ Action to do at init (= setting the state array)"""
                return [[False, 1e100], [True, 1e100], 0] # [[updateReady, UpdateOut (=tstart)], [[ActionReady, ActionOut (=tstart)]], texNumber] 
            
            def u_action(self, frameNumber, out, **kwargs):
                """ Action to do at update (= change texture) """
                self.n_stimuli = len(self.screen_uSelfMapTextureList)
                self.texIdx = (out[2] % self.n_stimuli) # % n because only n images in example, normally: comment this line and use out[2] as the index into the texList
                texL, texR = self.screen_uSelfMapTextureList[self.texIdx][0], self.screen_uSelfMapTextureList[self.texIdx][1] # get texture fileNames
                self.leftEyeScreen.updateScreenMaterialTexture( uSelfMapTexture = texL )
                self.rightEyeScreen.updateScreenMaterialTexture( uSelfMapTexture = texR )
                return time.time()
            
            def u_check(self, frameNumber, out, delay = 2):
                """ Check to do at update (= check whether the delay before texture action function can be run, to make sure image has already displayed fully on HMD) """
                t_now = time.time() 
                dt = t_now - out[0][1] 
                return dt >= delay
    
            def a_action(self, frameNumber, out, **kwargs):
                """ Action to do at action (= start timer to create a delay as default or e.g. start some measurement when user creates his own a_action function and updates the default pipelineFcns) """
                t_start = time.time() # e.g. get time or start measurement
                return t_start # stored in out[3] so it is available for a later check
    
            def a_check(self, frameNumber, out, delay = 2):
                """ Check to do at action (= check if time since start of timer is larger than delay to create a delay as default or e.g. check whether measurement file exists (indicating finished measurement) when user creates his own a_action function and updates the default pipelineFcns) """
                t_now = time.time() 
                dt = t_now - out[1][1] 
                return dt >= delay
    
            self.pipeFcnsDef = [[initWrapper,   {'action' : [i_action, {'initial_out' : [False, True, 0, 1e100]}]}],
                                [updateWrapper, {'action' : [u_action, {}],
                                                 'check'  : [u_check,  {'delay' : 2}]}], # wait some time so that image can be show on HMD: find by trial-and error. 1 sec should be ok though 
                                [actionWrapper, {'action' : [a_action, {}],
                                                 'check'  : [a_check,  {'delay' : 2}]}]
                                ]
        else:
            self.pipeFcnsDef = pipeFcnDef 
            
        return self.pipeFcnsDef

#------------------------------------------------------------------------------
def generate_color_texs(rgba_list, folder = './'):
    """ Generate color textures from rgba_list """
    os.makedirs(folder, exist_ok = True)
    texFiles = []
    for i,color in enumerate(rgba_list): 
        texFiles.append('{:s}/tex_{:1.0f}_{:1.0f}_{:1.0f}.jpg'.format(folder,*color[:3]))
        makeColorTex(color, texHeight = 100, texWidth = 100, save = texFiles[-1])
    texFiles = [(texFile,texFile) for texFile in texFiles]
    return texFiles

def generate_stimulus_tex_list(stimulus_list = None, 
                               equiRectImageLeftIsRight = False, 
                               equiRectImageLeftPos = 'bottom', 
                               rgba_save_folder = None):
    """ 
    Generate a list of textures 
    Args:
        
        :stimulus_list:
            | None or str or list, optional
            | If None: generate a preset list of rgb colors: np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1],[1,0,1,1],[0,1,1,1]])*255 
            | If str: 
            |     - filename of texture 
            |     -  or, filename of .iml file  with a list of filenames to textures
            |           (first line in path should be: "path" followed by the path to the images in the file list)
            | If list:
            |     - list of filenames to image textures.
            |   (if not None: any color input is ignored !)
            | If ndarray with rgba stimuli :
            |     - (equiRectImageLeftIsRight, equiRectImageLeftPos) will be updated to (True, None)
            |     - texture files will be generated in folder
        :rgba_save_folder:
            | Folder to save the generated full single-color textures in when stimulus_list is an ndarray or None.
            
    Returns:
        :stimulus_list: 
            | list of stimuli file textures
        :(equiRectImageLeftIsRight, equiRectImageLeftPos):
            - equiRectImageLeftIsRight: bool (left image = right image)
            - equiRectImageLeftPos: string or None 
    """
    if stimulus_list is None:
        stimulus_list = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1],[1,0,1,1],[0,1,1,1]])*255 
        equiRectImageLeftIsRight, equiRectImageLeftPos = True, None # single colors so no diff. between left and right
        stimulus_list = generate_color_texs(rgba_list = stimulus_list, folder = rgba_save_folder)
    elif isinstance(stimulus_list,str): 
        path_file = stimulus_list
        mypath, myfile = os.path.split(path_file)
        if (myfile[-4:] == '.iml'): # get list from file
            with open(path_file) as file:
                stimulus_list = file.readlines()
            ci_path = stimulus_list[0].strip()
            ci_path = mypath + ci_path[(ci_path.index('path')+4):].replace('\\','/')
            stimulus_list = [os.path.join(ci_path,f).strip() for f in stimulus_list[1:] if not ('.iml' in f)]
        else: # get list as all files in dir
            stimulus_list = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) & (not (('_L' in f) or ('_R' in f))) & (not ('.iml' in f)))]
        
    elif isinstance(stimulus_list,np.ndarray):
        stimulus_list = generate_color_texs(rgba_list = stimulus_list, folder = rgba_save_folder)
        equiRectImageLeftIsRight, equiRectImageLeftPos = True, None # single colors so no diff. between left and right
    elif isinstance(stimulus_list,list):
        pass # assume list of tex-images
        
    return stimulus_list, (equiRectImageLeftIsRight, equiRectImageLeftPos)

#------------------------------------------------------------------------------

def get_rgb_from_rgbtexpath(path):
    """ Get rgb values from filename """
    path = path[(path.index('tex_')+4):]
    r, path = int(path[:path.index('_')]), path[(path.index('_')+1):]
    g, path = int(path[:path.index('_')]), path[(path.index('_')+1):]
    b = int(path[:path.index('.')])
    rgb = np.array([r,g,b])
    return rgb

def getRectMask(roi, shape):
    """ Get a boolean rectangular mask with mask-area determined by the (row,col) coordinates of the top-left & bottom-right corners of the ROI """
    rm, rM = roi[0][0], roi[1][0]
    cm, cM = roi[0][1], roi[1][1]
    r = np.arange(shape[0])
    c = np.arange(shape[1])
    mask = np.zeros(shape) 
    mask[(r>=rm) & (r<=rM),:] += 1
    mask[:,(c>=cm) & (c<=cM)] += 1
    mask[mask<2] = 0
    return mask.astype(bool)

def getRoiImage(img, roi):
    mask = getRectMask(roi, img.shape)
    subimg = img[mask].reshape((*(roi[1] - roi[0] + 1),3))
    return subimg

def get_xyz_from_xyzmap_roi(xyzmap, roi):
    """ Get xyz values of Region-Of-Interest in XYZ-map """
    m, n = xyzmap.shape[:2]
    M,N = np.arange(m).astype(int), np.arange(n).astype(int)
    if roi is None:
        f = 2*10 # 1/10 of size of each image dimension
        roi = [[M//2-M//f,N//2-N//f],[M//2+M//f,N//2+N//f]] # (upper-left, lower-right) XY of rectangular Region-Of-Interest coordinates
    mask = getRectMask(roi, (M,N))
    xyzmap_ = xyzmap.copy() 
    xyzmap_[mask] = np.nan
    xyz = xyzmap_[~mask[...,0],:]
    return xyz, xyzmap_

def get_rgbFromTexPaths(rgbatexFiles):
    """ Get rgb values read from the filenames of the tex-files """
    if isinstance(rgbatexFiles, str):
        with open(rgbatexFiles,'r') as fid:
            rgbatexFiles = fid.readlines()
        rgbatexFiles = rgbatexFiles[1:] # get rid of path line
        rgbatexFiles = [x[x.index('tex_'):][:-1] for x in rgbatexFiles]
    rgbs = []
    for i, rgbatexFile in enumerate(rgbatexFiles):
        rgbs.append(get_rgb_from_rgbtexpath(rgbatexFile))  
    rgbs = np.array(rgbs)
    return rgbs    
    
def get_avgXYZFromXYZmaps(xyzmaps, roi = None):
    """ Get average xyz values of Region-Of-Interest in XYZ-maps """
    xyzs = []
    for xyzmap in enumerate(xyzmaps):
        xyzs.append(get_xyz_from_xyzmap_roi(xyzmap, roi).mean(0, keepdims = True))
    return np.array(xyzs)  
        
    
def generate_rgba_texs_iml(rgb, rgba_save_folder):
    """ Generate rgba texture images, save them in a folder and return a list of texFiles and a .iml file with the paths to the texFiles"""
    rgba = np.hstack((rgb,np.ones_like(rgb[:,:1])*255)) # add alpha channel required by HMDviewer
    texFiles, _ = generate_stimulus_tex_list(rgba, rgba_save_folder = rgba_save_folder)
    stimulus_iml = rgba_save_folder + 'stimulus_list.iml'
    texFiles_ = ['path' + '' + '\n'] + [file[0]+'\n' for file in texFiles]
    with open(stimulus_iml,'w') as fid:
        fid.writelines(texFiles_)
    return texFiles, stimulus_iml

#------------------------------------------------------------------------------


if __name__ == '__main__':       
    
    ###############################################################################
    # Example code on how to:
    #    - update the action function in the pipeFcns
    #    - prepare / generate stimulus textures
    #    - use the Viewer with(out) VR HMD
    ###############################################################################
    
    #===========================================================================
    # Tests of different options:
    #===========================================================================
    script_path = os.path.dirname(os.path.realpath(__file__)) # LMK save requires absolute path !!!
    
    #=== Action definitions ===================================================
    use_LMK = False
    
    if use_LMK:
        ## Setup instance of LMK for measurements:
        from luxpy.toolboxes.technoteam_lmk import lmkActiveX, kill_lmk4_process
        
        modfrequency = 72 # check what freq. the headset is running on: Quest2: 72 Hz, Rift CV2: 90 Hz
        lmkX = lmkActiveX('tts20035', 'xvr', focusfactor = None,
                          autoscan = False, autoexposure = True,
                          modfrequency = modfrequency, maxtime = 10,
                          verbosity = None)
                           
        def a_action(self, frameNumber, out, 
                     lmkX = None, folder_name = None, file_name_base = None):
            if not os.path.exists(folder_name): 
                os.makedirs(folder_name, exist_ok = True)
            if lmkX is not None: 
                file = os.path.join(folder_name, '{:s}_{:1.0f}'.format(file_name_base, self.texIdx))
                lmkX.measureCaptureXYZmap(folder_name, '{:s}_{:1.0f}'.format(file_name_base, self.texIdx))
                
            return file + '.pcf'
    
        def a_check(self, frameNumber, out):
            if isinstance(out[1][1],(int, float)):
                return False
            else:
                return bool(os.path.exists(out[1][1]))
        
        actionWrapperDict = {'action' : [a_action, {'lmkX':lmkX, 'folder_name' : script_path+'./lmk_xyzmaps/','file_name_base':'lmk-xyzmap'}],
                             'check'  : [a_check,  {}]}
    else:
        #--------------------------------------------------------------------------
        def a_action(self, frameNumber, out, **kwargs):
            t_start = time.time() # e.g. get time or start measurement
            return t_start # stored in out[3] so it is available for a later check
    
        def a_check(self, frameNumber, out, delay = 3):
            t_now = time.time() 
            dt = t_now - out[1][1] 
            return dt >= delay
        
        actionWrapperDict =  {'action' : [a_action, {}],
                              'check'  : [a_check,  {'delay' : 0.1}]}
    
    #==== Prepare stimuli =====================================================
        
    stimulus_list = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]])*255
    texFiles, _ = generate_stimulus_tex_list(stimulus_list, rgba_save_folder = 'rgba_texs')
    
    # # Generate stimulus textures:
    # os.makedirs('./temp_color_stimulus_folder/', exist_ok = True)
    # texFiles = []
    # for i,color in enumerate(stimulus_list): 
    #     texFiles.append('./temp_color_stimulus_folder/tex_{:1.0f}_{:1.0f}_{:1.0f}.jpg'.format(*color[:3]))
    #     makeColorTex(color, texHeight = 100, texWidth = 100, save = texFiles[-1])
    # texFiles = [(texFile,texFile) for texFile in texFiles]
    
    # or load files from folder or list.iml file:
    texFiles, _ = generate_stimulus_tex_list('./spheremaps/list.iml')
    
    #==== Prepare hmdviewer ===================================================
    
    hmdviewer = HmdStereoViewer(screen_uSelfMapTexture = texFiles, 
                                screen_geometry = 'sphere',vrFlag = False)
    
    # pipeFcns = copy.deepcopy(hmdviewer.pipeFcnsDef) 
    # pipeFcns[-1] = (pipeFcns[-1][0], actionWrapperDict) # replace default action with custom action
    
    # === Display =============================================================
    
    # hmdviewer.display()
    
    #==== Run tests of display + action ========================================
    # hmdviewer.run()
    # hmdviewer.run(only_once = True)
    t_begin = time.time()
    
    hmdviewer.run(pipeFcnsUpdate = [None,None, [None, actionWrapperDict]],  only_once = False,
                  u_delay = 1, a_delay = 1)
       
    t_end = time.time()
    print('Total run time: ', t_end - t_begin)

