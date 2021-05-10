# -*- coding: utf-8 -*-
"""

Module for reading and writing IES TM2714 SPDX files
====================================================

 :_SPDX_TEMPLATE: template dictionary for SPDX data.
 
 :read_spdx(): Read xml file or convert xml string with spdx data to dictionary.
     
 :write_spdx(): Convert spdx dictionary to xml string (and write to .spdx file)

Reference:
    1. http://www.ies.org/iestm2714


Created on Mon May 10 16:25:09 2021

@author: ksmet1977 [at] gmail.com
"""
import os
import io
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
import copy

# Setup general headers:
_XML_VERSION = 1.0
_XML_HEADER = '<?xml version="{:1.0f}"?>'.format(_XML_VERSION)
_IES_TM2714_VERSION = 1.0
_IES_TM2714_HEADER = '<IESTM2714 xmlns="iestm2714" version="{:1.0f}">'.format(_IES_TM2714_VERSION)
_IES_TM2714_CLOSE = '</IESTM2714>'

# default header dict:
_HEADER = {'Manufacturer'        : 'Unknown',
		  'CatalogNumber'        : 'Unknown',
		  'Description'          : 'Unknown',
		  'FileCreator'          : 'Unknown',
		  'Laboratory'           : 'Unknown',
		  'UniqueIdentifier'     : 'Unknown',
		  'ReportNumber'         : 'Unknown',
		  'ReportDate'           : 'Unknown',
		  'DocumentCreationDate' : 'Unknown',
		  'comments'             : 'Unknown'}

# default spectral distribution dict:
_SPECTRALDISTRIBUTION = {'SpectralQuantity'  : 'Unknown',
                        'BandwidthFWHM'      : 'Unknown',
                        'BandwidthCorrected' : 'Unknown',
                        'SpectralData'       :  'unknown'}
# default SPDX dict
_SPDX_TEMPLATE = {'Header' : _HEADER,
                'SpectralDistribution' : _SPECTRALDISTRIBUTION}


__all__ = ['_SPDX_TEMPLATE', 'read_spdx', 'write_spdx']

#------------------------------------------------------------------------------
# Writing xml (dict to xml)
#------------------------------------------------------------------------------
def _process_value(field, value, indent = ''):
    if (field == "SpectralData") & isinstance(value, np.ndarray):
        xml = ''
        for wi, vi in zip(value[0],value[1]):
            xml += indent + '<SpectralData wavelength="{:1.6f}">{:1.6f}</SpectralData>\n'.format(wi,vi)
        return xml
    else:
        return value
    
def _write_xml_field(field, value, xml = '', indent = '    ', value_fields = []):
    if field not in value_fields: xml = xml + indent + '<' + field + '>'
    if isinstance(value, dict):
        xml = xml + '\n'
        for key in value.keys():
            xml = _write_xml_field(key, value[key], xml = xml, 
                                   indent = indent + '    ', value_fields = value_fields)
        if field not in value_fields: xml = xml + indent + '</' + field + '>\n'
    else:
        xml = xml + _process_value(field, value, indent = indent) 
        if field not in value_fields: xml += '</' + field + '>\n'
    return xml
        
def write_spdx(spdx_dict, filename = None):
    """ 
    Convert spdx dictionary to xml string (and write to .spdx file).
    
    Args:
       :spdx_dict:
           | dictionary with spdx keys (see _SPDX for keys).
       :filename:
           | None, optional
           | string with filename to write xml data to.
           
    Returns:
       :spdx_xml:
           | string with xml data in spdx dictionary.
    """
    spdx_xml = ''
    indent = '    '
    value_fields = ['SpectralData'] # fields that have a value
    spdx_xml += _XML_HEADER + '\n' + _IES_TM2714_HEADER + '\n'
    for field in spdx_dict:
        spdx_xml = _write_xml_field(field, spdx_dict[field], xml = spdx_xml, indent = indent, 
                               value_fields = value_fields)
    spdx_xml += _IES_TM2714_CLOSE
    
    if filename is not None:
        file, ext = os.path.splitext(filename)
        filename = file + '.spdx'
        with open(filename, 'w') as f:
            f.write(spdx_xml)
    return spdx_xml
    


#------------------------------------------------------------------------------
# Reading xml (xml to dict)
#------------------------------------------------------------------------------

def _etree_to_dict(t):
    """ 
    Convert tree to dict
    
    from https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary/10077069 
    """ 
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(_etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
              d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def read_spdx(spdx):
    """ 
    Read xml file or convert xml string with spdx data to dictionary.
    
    Args:
       :spdx:
           | xml string or file with spdx data.
           
    Returns:
        :spdx_dict:
           | spdx data in a dictionary.
    """
    if isinstance(spdx,io.StringIO):
        spdx = spdx.read()
    if ('<?xml' not in spdx) & ('.spdx' in spdx[-5:]):
        with open(spdx) as f:
            spdx = f.readlines()
    else:
        spdx = spdx.split('\n')
        
    xml_header = [spdx[0]] # get xml header
    spdx = spdx[2:-1] # get rid of _IES_TM2714_HEADER and _IES_TM2714_CLOSE
    spdx = xml_header + ['<root>'] + spdx + ['</root>'] # add root field for easy conversion
    spdx = '\n'.join(spdx) # make string again
    root = ET.fromstring(spdx) #parse xml from string
    spdx_dict =  _etree_to_dict(root)['root'] # get dictionary located at key 'root'
    if 'SpectralDistribution' in spdx_dict.keys():  # process SpectralData data
        if 'SpectralData' in spdx_dict['SpectralDistribution'].keys():
            dspd = spdx_dict['SpectralDistribution']['SpectralData'] # make tmp view
            spdx_dict['SpectralDistribution']['SpectralData'] = np.array([[float(dspd[i]['@wavelength']),float(dspd[i]['#text'])] for i in range(len(dspd))]).T
    return spdx_dict

if __name__ == '__main__':
    import luxpy as lx
    
    # create spdx dict:
    spdx_dict = copy.copy(_SPDX_TEMPLATE)
    spdx_dict['Header']['Manufacturer'] = 'CIE'
    spdx_dict['Header']['Description'] = 'CIE D65 illuminant (5nm)'
    spdx_dict['Header']['UniqueIdentifier'] = 'CIE D65'
    spdx_dict['Header']['Laboratory'] = 'CIE'
    spdx_dict['Header']['FileCreator'] = 'luxpy.spdx_iestm2714'
    spdx_dict['SpectralDistribution']['SpectralQuantity'] = 'W/nm'
    spdx_dict['SpectralDistribution']['SpectralData'] = lx._CIE_D65[:,::5]
    spdx_xml = write_spdx(spdx_dict, filename = 'cie_d65_5nm.spdx')

    print('Spdx dictionary write test: spdx\n', spdx_xml)
    
    spdx_xml_from_string = read_spdx(spdx_xml)
    spdx_xml_from_file = read_spdx('cie_d65_5nm.spdx')
    print('Spdx string read test:\n', spdx_xml_from_string)
    print('Spdx file read test:\n', spdx_xml_from_file)
    
