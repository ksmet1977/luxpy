# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with helper functions 
###############################################################################
#
# np2d(): Make a tuple, list or numpy array at least a 2D numpy array.
#
# np2dT(): Make a tuple, list or numpy array at least a 2D numpy array and tranpose.
#
# np3d(): Make a tuple, list or numpy array at least a 3D numpy array.
#
# np3dT(): Make a tuple, list or numpy array at least a 3D numpy array and tranpose (swap) first two axes.
#
# normalize_3x3_matrix(): Normalize 3x3 matrix M to xyz0 -- > [1,1,1]
#
# put_args_in_db(): Takes the **args with not-None input values of a function fcn and overwrites the values of the corresponding keys in dict db.
#                   See put_args_in_db? for more info.
#
# getdata(): Get data from csv-file or convert between pandas dataframe (kind = 'df') and numpy 2d-array (kind = 'np').
#
# dictkv(): Easy input of of keys and values into dict (both should be iterable lists).
#
# OD(): Provides a nice way to create OrderedDict "literals".
#
# meshblock(): Create a meshed black (similar to meshgrid, but axis = 0 is retained) to enable fast blockwise calculation.
#
# aplit(): Split numpy.array data on (default = last) axis.
#
# ajoin(): Join tuple of numpy.array data on (default = last) axis.
#
# broadcast_shape(): Broadcasts shapes of data to a target_shape. Useful for block/vector calculations in which numpy fails to broadcast correctly.
#
# todim():  Expand x to dimensions that are broadcast-compatable with shape_ of another array.
#
#------------------------------------------------------------------------------

Created on Sun Jun 18 21:12:30 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import np, pd, odict
__all__ = ['np2d','np3d','np2dT','np3dT','diagc','diagr','put_args_in_db','getdata','dictkv','OD','meshblock','asplit','ajoin','broadcast_shape','todim']

#--------------------------------------------------------------------------------------------------
def np2d(data):
    """
    Make a tuple, list or numpy array at least a 2D numpy array.
    
    Args:
        :data: tuple, list, numpy.ndarray
        
    Returns:
        :returns: numpy.ndarray with .ndim >= 2
    """
    if isinstance(data, np.ndarray):# assume already atleast_2d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=2):
            return data   
        else:
            return np.atleast_2d(data)
    else:
        return np.atleast_2d(np.array(data))


def np2dT(data):
    """
    Make a tuple, list or numpy array at least a 2D numpy array and transpose.
    
    Args:
        :data: tuple, list, numpy.ndarray
        
    Returns:
        :returns: numpy.ndarray with .ndim >= 2 and with transposed axes.
    """
    if isinstance(data, np.ndarray):# assume already atleast_2d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=2):
            return data.T   
        else:
            return np.atleast_2d(data).T
    else:
        return np.atleast_2d(np.array(data)).T

def np3d(data):
    """
    Make a tuple, list or numpy array at least a 3d numpy array.
    
    Args:
        :data: tuple, list, numpy.ndarray
        
    Returns:
        :returns: numpy.ndarray with .ndim >= 3
    """
    if isinstance(data, np.ndarray):# assume already atleast_3d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=3):
            return data   
        else:
            return np.expand_dims(np.atleast_2d(data),axis=0)
    else:
        return np.expand_dims(np.atleast_2d(np.array(data)),axis=0)
    
def np3dT(data): # keep last axis the same
    """
    Make a tuple, list or numpy array at least a 3d numpy array and transposed first 2 axes.
    
    Args:
        :data: tuple, list, numpy.ndarray
        
    Returns:
        :returns: numpy.ndarray with .ndim >= 3 and with first two axes transposed (axis=3 is kept the same).
    """
    if isinstance(data,np.ndarray):# assume already atleast_3d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=3):
            return data.transpose((1,0,2))
        else:
            return np.expand_dims(np.atleast_2d(data),axis=0).transpose((1,0,2))
    else:
        return np.expand_dims(np.atleast_2d(np.aray(data)),axis=0).transpose((1,0,2))


def diagc(x):
    """
    Take diagonal of numpy matrix x and store in numpy 2d-column vector.
    
    Args:
        :x: nump.ndarray 
        
    Returns:
        :returns: numpy.ndarray(2d column vector)
    
    """
    return np2dT(np.diag(x))

def diagr(x):
    """
    Take diagonal of numpy matrix x and store in numpy 2d-row vector.
    
    Args:
        :x: nump.ndarray column vector
        
    Returns:
        :returns: numpy.ndarray (2d row vector)
    
    """
    return np2d(np.diag(x))
#------------------------------------------------------------------------------
def normalize_3x3_matrix(M,xyz0 = np.array([[1.0,1.0,1.0]])):
    """
    Normalize 3x3 matrix to xyz0 -- > [1,1,1]
    If M == 1by9: reshape
    """
    M = np2d(M)
    if M.shape[-1]==9:
        M = M.reshape(3,3)
    return np.dot(diagm(1/np.dot(M,xyz0)),M)


#------------------------------------------------------------------------------
def normalize_3x3_matrix(M, xyz0 = np.array([[1.0,1.0,1.0]])):
    """
    Normalize 3x3 matrix M to xyz0 -- > [1,1,1]
    If M.shape == (1,9): M is reshaped to (3,3)
    
    Args:
        :M: numpy.array((3,3) or numpy.array((1,9))
        :xyz0: np.2darray, optional 
        
    Returns:
        :returns: normalized matrix such that M*xyz0 = [1,1,1]
    """
    M = np2d(M)
    if M.shape[-1]==9:
        M = M.reshape(3,3)
    return np.dot(np.diagflat(1/(np.dot(M,xyz0.T))),M)

#------------------------------------------------------------------------------
def put_args_in_db(db, args):
    """
    Takes the **args with not-None input values of a function fcn and overwrites the values of the corresponding keys in dict db.
    (**args are collected with the built-in function locals(), See example usage below)
    
    Args:
        :db: dict
    
    Returns:
        :returns: dict with the values of specific keys overwritten by the not-None values of corresponding **args of a function fcn.
    
    Example usage:
        _db = {'c' : 'c1', 'd' : 10, 'e' : {'e1':'hello', 'e2':1000}}

        def test_put_args_in_db(a, b, db = None, c = None,d = None,e = None):
    
            args = locals().copy() # get dict with keyword input arguments to function 'test_put_args_in_db'
            
            db = put_args_in_db(db,args) # overwrite non-None arguments in db copy
            
            if db is not None: # unpack db for further use
                c,d,e = [db[x] for x in sorted(db.keys())]
            
            print(' a : {}'.format(a))
            print(' b : {}'.format(b))
            print(' db: {}'.format(db))
            print(' c : {}'.format(c))
            print(' d : {}'.format(d))
            print(' e : {}'.format(e))
            print('_db: {}'.format(_db))
 
    """
    # overwrite not-'None' input arguments in db:
    if db is not None:
        dbc = db.copy()
        dbc_keys = dbc.keys()
        for argkey in args.keys():
            if args[argkey] is not None:
                if argkey in dbc_keys:
                    dbc[argkey] =  args[argkey]
        return dbc
    else:
        return db



#--------------------------------------------------------------------------------------------------
def getdata(data, kind = 'np', columns=None, header = None, sep = ',', datatype = 'S', verbosity = True):
    """
    Get data from csv-file or convert between pandas dataframe and numpy 2d-array.
    
    Args:
        :data: 
            - str with path to file containing data
            - numpy.ndarray with data
            - pandas.dataframe with data
        :kind: str ['np','df'], optional 
            Determines type(:returns:), np: numpy.ndarray, df: pandas.dataframe
        :columns: None or list[str] of column names for dataframe, optional
        :header: None (no header in data file) or 'infer' (infer headers from file), optional
        :sep: ',' or '\t' or other char, optional
            Column separator in data file
        :datatype': 'S' (light source spectrum) or 'R' (reflectance spectrum) or other, optional
            Specifies a type of data. Is used when creating column headers when :column: is None.
        :verbosity: True, False, optional
            Print warning when inferring headers from file.
    
    Returns:
        :returns: data as numpy.ndarray or pandas.dataframe
      
    """
    if isinstance(data,str):
        datafile = data
        data = pd.read_csv(data,names=None,index_col = None,header = header,sep = sep)

        # Set column headers:
        if header == 'infer':
            if verbosity == True:
                warnings.warn('getdata(): Infering HEADERS from data file: {}!'.format(datafile))
                columns = data.columns
        elif (columns is None):
            data.columns = ['{}{}'.format(datatype,x) for x in range(len(data.columns))] 
        if columns is not None:
            data.columns = columns

    if isinstance(data,np.ndarray) & (kind == 'df'):
        if columns is None:
            columns = ['{}{}'.format(datatype,x)  for x in range(data.shape[1])] 
        data = pd.DataFrame(data, columns = columns)

    elif isinstance(data,pd.DataFrame) & (kind == 'np'):
        data = data.values
    return data

#--------------------------------------------------------------------------------------------------
def dictkv(keys=None,values=None, ordered = True): 
    """
    Easy input of of keys and values into dict.
    
    Args:
        :keys: iterable list[str,...] of keys
        :values: iterable list[...,..., ] of values
        :ordered: True, False, optional
            True: creates an ordered dict using 'collections.OrderedDict()'
    Returns:
        :returns: (ordered) dict
    """
    if ordered is True:
        return odict(zip(keys,values))
    else:
        return dict(zip(keys,values))
    
class OD(object):
    """
    This class provides a nice way to create OrderedDict "literals".
    
    """
    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = slices,
        return odict((slice.start, slice.stop) for slice in slices)
# Create a single instance; we don't ever need to refer to the class.
OD = OD()

#--------------------------------------------------------------------------------------------------
def meshblock(x,y):
    """
    Create a meshed block from x and y.
    
    Is similar to meshgrid, but axis = 0 is retained to enable fast blockwise calculation.
    
    Args: 
        :x: numpy.ndarray with ndim == 2
        :y: numpy.ndarray with ndim == 2
        
    Returns:
        :X,Y: 2 numpy.ndarrays with ndim == 3 
            X.shape = (x.shape[0],y.shape[0],x.shape[1])
            Y.shape = (x.shape[0],y.shape[0],y.shape[1])
    """
    Y = np.transpose(np.repeat(y,x.shape[0],axis=0).reshape((y.shape[0],x.shape[0],y.shape[1])),axes = [1,0,2])
    X = np.repeat(x,y.shape[0],axis=0).reshape((x.shape[0],y.shape[0],x.shape[1]))
    return X,Y

#---------------------------------------------------------------------------------------------------
def asplit(data):
    """
    Split data on last axis
    
    Args:
        :data: numpy.ndarray
        
    Returns:
        :returns: numpy.ndarray, numpy.ndarray, ... (number of returns is equal data.shape[-1])
    """
    #return np.array([data.take([x],axis = len(data.shape)-1) for x in range(data.shape[-1])])
    return [data[...,x] for x in range(data.shape[-1])]


def ajoin(data):
    """
    Join data on last axis.
    
    Args:
        :data: tuple (numpy.ndarray, numpy.array, ...)
        
    Returns:
        :returns: numpy.array (shape[-1] is equal to tuple length)
    """
    if data[0].ndim == 2: #faster implementation
        return np.transpose(np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))),(1,2,0))
    elif data[0].ndim == 1:
        return np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))).T
    else:
        return np.hstack(data)[0]
    

#---------------------------------------------------------------------------------------------------
def broadcast_shape(data,target_shape = None, expand_2d_to_3d = None, axis0_repeats = None, axis1_repeats = None):
    """
    Broadcasts shapes of data to a target_shape.
    
    Useful for block/vector calculations in which numpy fails to broadcast correctly.
    
    Args:
        :data: np.ndarray 
        :target_shape: None (returns unchanged :data:) or tuple with requested shape, optional
        :expand_2d_to_3d: None (do nothing) or ... (if ndim == 2, expand from 2 to 3 dimensions), optional
        :axis0_repeats: None (keep axis=0 same size) or number of times to repeat axis=0, optional
        :axis1_repeats: None (keep axis=1 same size) or number of times to repeat axis=1, optional

    Returns:
        :returns: reshaped numpy.ndarray
    """
    data = np2d(data)
    
    # expand shape along axis (useful for some functions that allow block-calculations when the data is only 2d )
    if (expand_2d_to_3d is not None) & (len(data.shape) == 2):
        data = np.expand_dims(data, axis = expand_2d_to_3d)
     
    if target_shape is not None:
        dshape = data.shape
        if dshape != target_shape:
            axis_of_v3 = len(target_shape)-1
            if (dshape[0] != target_shape[0]): # repeat along axis 0
                if axis0_repeats is None:
                    axis0_repeats = (target_shape[0]-dshape[0] + 1)
                data = np.repeat(data,axis0_repeats,axis = 0)
            if (len(target_shape)>2) & (len(data.shape)==2): # repeat along axis 1, create axis if necessary
                data = np.expand_dims(data,axis = axis_of_v3-1) # axis creation
                dshape = data.shape
                if (dshape[1] != target_shape[1]):
                    if axis1_repeats is None:
                        axis1_repeats = (target_shape[1]-dshape[1] + 1) # repititon
                    data = np.repeat(data,axis1_repeats,axis = 1)

        for i in range(2):
            if (data.shape[i] > 1) & (data.shape[i] != target_shape[i]):
                raise Exception('broadcast_shape(): Cannot match dim of data with target: data.shape[i]>1  & ...  != target.shape[i]')
    return data


def todim(x,shape_, add_axis = 1, equal_shape = False): 
    """
    Expand x to dimensions that are broadcast-compatable with shape_ of another array.
    
    Args:
        :x: numpy.ndarray
        :shape_: tuple with target shape
        :add_axis: 1, optional
            Determines where in x.shape an axis should be added
        :equal_shape: False or True, optional
            True: expand :x: to identical dimensions (speficied by :shape_:)
            
    Returns:
        :returns: numpy.ndarray broadcast-compatable with shape_.
    """
    if x is None:
        return np.broadcast_arrays(x,np.ones(shape_))[0]
    else:
        x = np2d(x)
        sx = x.shape
        lsx = len(sx)
        lshape_ = len(shape_)
        if (sx == shape_):
            pass
        else:
            
            if ((lsx == 1) | (sx == (1,shape_[-1])) | (sx == (shape_[-1],1))): 
                if (sx == (shape_[-1],1)):
                    x = x.T
                if lsx != lshape_:
                    x = np.expand_dims(x,0)
            elif (lsx == 2):
                if (lshape_ == 3):
                    sd = np.setdiff1d(shape_, sx,assume_unique=True)
                    if len(sd) == 0:
                        ax = add_axis
                    else:
                        ax = np.where(shape_==sd)[0][0]
                    x = np.expand_dims(x,ax)
                else:
                    raise Exception("todim(x,shape_): dimensions do not match for 2d arrays.")  
            else:
                raise Exception("todim(x,shape_): no matching dimensions between 3d x and shape_.")
        if equal_shape == False:
            return x
        else:
            return np.ones(shape_)*x #make dims of x equal to those of a (shape_)