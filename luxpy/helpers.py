# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:12:30 2017

@author: kevin.smet
"""
###############################################################################
# helper functions 
###############################################################################
#
# np2d(): Make a tupple, list or numpy array at least 2d array.
#
# np2dT(): Make a tupple, list or numpy array at least 2d array and tranpose.
#
# np3d(): Make a tupple, list or numpy array at least 3d array.
#
# np3dT(): Make a tupple, list or numpy array at least 3d array and tranpose (swap) first two axes.
#
# put_args_in_db(): Overwrites values in dict db with 'not-None' input arguments from function (obtained with built-in locals()).
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
# aplit(): Split np.array data on (default = last) axis.
#
# ajoin(): Join tupple of np.array data on (default = last) axis.
#
# broadcast_shape(): Broadcasts shapes of data to a target_shape, expand_2d_to_3d if not None and data.ndim == 2, axis0,1_repeats specify how many times data much be repeated along axis (default = same axis size).
#                    Useful for block/vector calculation in which nupy fails to broadcast correctly.
#
#------------------------------------------------------------------------------

from luxpy import *
__all__ = ['np2d','np3d','np2dT','np3dT','put_args_in_db','getdata','dictkv','OD','meshblock','asplit','ajoin','broadcast_shape']

#--------------------------------------------------------------------------------------------------
def np2d(data):
    """
    Make a tupple, list or numpy array at least 2d array.
    """
    if isinstance(data,np.ndarray):# assume already atleast_2d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=2):
            return data   
        else:
            return np.atleast_2d(data)
    else:
        return np.atleast_2d(np.array(data))


def np2dT(data):
    """
    Make a tupple, list or numpy array at least 2d array and transpose.
    """
    if isinstance(data,np.ndarray):# assume already atleast_2d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=2):
            return data.T   
        else:
            return np.atleast_2d(data).T
    else:
        return np.atleast_2d(np.array(data)).T

def np3d(data):
    """
    Make a tupple, list or numpy array at least 3d array.
    """
    if isinstance(data,np.ndarray):# assume already atleast_3d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=3):
            return data   
        else:
            return np.expand_dims(np.atleast_2d(data),axis=0)
    else:
        return np.expand_dims(np.atleast_2d(np.aray(data)),axis=0)
    
def np3dT(data): # keep last axis the same
    """
    Make a tupple, list or numpy array at least 3d array.
    """
    if isinstance(data,np.ndarray):# assume already atleast_3d when nd.array (user has to ensure input is an array)
        if (len(data.shape)>=3):
            return data.transpose((1,0,2))
        else:
            return np.expand_dims(np.atleast_2d(data),axis=0).transpose((1,0,2))
    else:
        return np.expand_dims(np.atleast_2d(np.aray(data)),axis=0).transpose((1,0,2))

#------------------------------------------------------------------------------
def put_args_in_db(db,args):
    """
    Overwrites values in dict db with 'not-None' input arguments from function (obtained with built-in locals()).
    
    Example usage:
        _db = {'c' : 'c1', 'd' : 10, 'e' : {'e1':'hello', 'e2':1000}}

        def test_put_args_in_db(a,b,db = None, c=None,d=None,e=None):
    
            args = locals().copy() # get dict with keyword input arguments to function
            
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
def getdata(data,kind = 'np', index=None, columns=None,header = None,sep = ',',datatype = 'S',verbosity = True):
    """
    Get data from csv-file or convert between pandas dataframe and numpy 2d-array.
    """
    #idx = index
    if isinstance(data,str):
        datafile = data
        data = pd.read_csv(data,names=None,index_col=0,header = header,sep = sep)

        # Set column headers:
        if header == 'infer':
            if verbosity == True:
                warnings.warn('getdata(): Infering HEADERS from data file: {}!'.format(datafile))
                columns = data.columns
                index = data.index.name
        elif (columns is None):
            data.columns = ['{}{}'.format(datatype,x) for x in range(len(data.columns))] 
        if columns is not None:
            data.columns = columns    
        if (index == 'wl') | (index == 'wavelength') :
            data.index.name = 'wl'
            index = 'wl'

    if isinstance(data,np.ndarray) & (kind == 'df'):
        if index is None:
            idx = None
        elif isinstance(index,str):
            if (index == 'wl') | (index == 'idx'):
                idx = data[0]
                data = data[1:]
            else:
                idx = None
        else:
            idx = index
        if columns is None:
                columns = ['{}{}'.format(datatype,x)  for x in range(data.shape[0])] 
        data = pd.DataFrame(data,index = idx,columns = columns)
        if index == 'wl':
            data.index.name = 'wl'

    elif isinstance(data,pd.DataFrame) & (kind == 'np'):
        idx = data.index
        if index is not None:
            data = np.concatenate((np.atleast_2d(idx),data.values.transpose()),axis = 0)
        else:
            data.values.transpose()
    return data

#--------------------------------------------------------------------------------------------------
def dictkv(keys=None,values=None, ordered = True): 
    """
    Easy input of of keys and values into dict (both should be iterable lists).
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
    Create a meshed black (similar to meshgrid, but axis = 0 is retained) to enable fast blockwise calculation.
    """
    Y = np.transpose(np.repeat(y,x.shape[0],axis=0).reshape((y.shape[0],x.shape[0],y.shape[1])),axes = [1,0,2])
    X = np.repeat(x,y.shape[0],axis=0).reshape((x.shape[0],y.shape[0],x.shape[1]))
    return X,Y

#--------------------------------------------------------------------------------------------------
#def take_f(data,indices = 0, f = None, f_axis = None,t_axis = None, e_axis = None):
#    """
#    Take a subset of data determined by indices list along (t_axis: take()) of data,
#    additionally perform function f and expand dim of result of f to e_axis.
#    """  
#    axis_ = len(data.shape)-1
#    if t_axis == None: # axis along which to take indices
#        t_axis = axis_
#    if e_axis == None: # expansion axis
#        e_axis = axis_
#    if f_axis == None: # function axis
#        f_axis = axis_
#    out = data.take(indices,axis = t_axis)
#    if f is not None:
#        out = f(out,axis = f_axis)
#        out = np.expand_dims(out,axis = e_axis)
#    return out

#---------------------------------------------------------------------------------------------------
def asplit(data, axis = None):
    """
    Split np.array data on (last) axis
    """
    if axis is None:
        axis = len(data.shape)-1
    return np.array([data.take([x],axis = axis) for x in range(data.shape[-1])])


def ajoin(data, axis = None):
    """
    Join np.array data on (last) axis
    """
    if axis is None:
        axis = len(data[0].shape)-1
    return np.concatenate(data, axis = axis)

#---------------------------------------------------------------------------------------------------
def broadcast_shape(data,target_shape = None, expand_2d_to_3d = None, axis1_repeats = None, axis0_repeats = None):
    """
    Broadcasts shapes of data to a target_shape, expand_2d_to_3d if not None and data.ndim == 2, axis0,1_repeats specify how many times data much be repeated along axis (default = same axis size).
    Useful for block/vector calculation in which nupy fails to broadcast correctly.
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



#---------------------------------------------------------------------------------------------------
#def takea(data,indices,axis):
#    """
#    Take subset of data specified by indices and axis by []-indexing.
#    """
#    if len(data.shape) == 1:
#        if axis == 0:
#            return data[indices,None]
#        else:
#            raise Exception('axis = {}, but shape = {}'.format(axis,data.shape))
#    elif len(data.shape) == 2:
#        if axis == 0:
#            return data[indices,None]
#        elif axis == 1:
#            return data[:,indices,None]
#        else:
#            raise Exception('axis = {}, but shape = {}'.format(axis,data.shape))
#    elif len(data.shape) == 3:
#        if axis == 0:
#            return data[indices,None]
#        elif axis == 1:
#            return data[:,indices,None]
#        elif axis == 2:
#            return data[:,:,indices,None]
#        else:
#            raise Exception('axis = {}, but shape = {}'.format(axis,data.shape))
#    else:
#        raise Exception('axis > 3, not supported by takea')
            

#---------------------------------------------------------------------------------------------------
#def wrapper(f,*args,**kwargs):
#    """
#    Wrapper function for wrapped function f.
#    """
#    def wrapped():
#        return f(*args,**kwargs)
#    return wrapped

#----------------------------------------------------------------------------------------------------
#def np2dT0(data, T = 0, *args, **kwargs):
#    # make a tupple, list or numpy array at least 2d array and transpose (1) or not(0)
#    
#    if T == 0:
#        return np.atleast_2d(np.array(data))
#    elif T == 1:
#        return np.atleast_2d(np.array(data)).T
#    elif T is None:
#        return data # do nothing
#    else:
#        return data