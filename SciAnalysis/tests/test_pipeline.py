import tempfile
from PIL import Image
import os
import numpy as np

import time

from SciAnalysis.XSAnalysis.Data import Calibration

from databases import databases
from detectors import detectors2D

from copy import deepcopy


#from SciAnalysis.XSAnalysis.Protocols import load_SAXS_img

#import dask
from dask import delayed
from dask.delayed import Delayed
from toolz import curry

from databroker.broker import Header
#from distributed import Client
#_pipeline_client = Client('127.0.0.1:8788')

from nose.tools import assert_true, assert_false
from numpy.testing import assert_array_almost_equal

# caching stuff
#import chest
#_CACHE = chest.Chest()
#dask.set_options(cache=_CACHE)

# base class
class Protocol:
    pass

# TODO : add run_default
# TODO : add run_explicit
# TODO : how to handle implicit arguments? (Some global maybe?)
# TODO : add databroker keymap

''' For now, assume all incoming arguments are well defined each step.

There will be a case where this is not true.
For ex : linecut -> need to backpropagate until the latest data set that
is computed is found. It will be necessary to figure out what to fill in
for missing arguments for that data set.

Ideas introduced:
1. SciResult : this is a dictionary of results. It may contain data
    stored in filestore. 
2. new class specifiers : 
    _name : some unique name for protocol
    _depends : dependencies of the arguments of the protocol
        {'_arg0' : ..., '_arg1' : ..., ... 'foo' : ...}
        _argn means nth argument, rest are keywords
    _func_args : the explicit arguments for function
    _keymap : the keymap of results


PROBLEMS : 
1. Delayed object will modify instances of classes


QUESTIONS:
1.Object oriented or functional approach?
2. Two possible conventions for two step process:
    - explicit, using objects
    - curry function func(arg, **kwargs)
        only when arg is supplied does it compute



'''

#

# this is like a dict but I can identify it
# also, hashes well in dask (for nested dicts/SciResults too)
class SciResult(dict):
    ''' Something to distinguish a dictionary from
        but in essence, it's just a dictionary.'''

    def __init__(self, **kwargs):
        super(SciResult, self).__init__(**kwargs)
        # identifier, needed for Dask which transforms SciResult into a dict
        # TODO : Suggest to dask that classes should not be modified
        self['_SciResult'] = 'SciResult-version1'

# This decorator parses SciResult objects, indexes properly
# takes a keymap for args
# this unravels into arguments if necessary
# TODO : Allow nested keymaps
def parse_sciresults(keymap, output_names):
    # from keymap, make the decorator
    def decorator(f):
        # from function modify args, kwargs before computing
        def _f(*args, **kwargs):
            for i, val in enumerate(args):
                if isinstance(val, SciResult):
                    key = "_arg{}".format(i)
                    args[i] = val[keymap[key]]
            for key, val in kwargs.items():
                if isinstance(val, SciResult):
                    kwargs[key] = val[keymap[key]]
            result = f(*args, **kwargs)
            if len(output_names) == 1:
                result = {output_names[0] : result}
            else:
                result = {output_names[i] : res for i, res in enumerate(result)}

            return SciResult(**result)
        return _f
    return decorator

    
class load_saxs_image:
    _accepted_args = ['infile']
    _keymap = {'infile' : 'infile'}
    _output_names = ['image']
    _name = "XS:load_saxs_image"

    def __init__(self, **kwargs):
        self.kwargs= kwargs

    def run(self, **kwargs):
        new_kwargs = self.kwargs.copy()
        new_kwargs.update(kwargs)
        return self.run_explicit(_name=self._name, **new_kwargs)

    @delayed(pure=True)
    @parse_sciresults(_keymap, _output_names)
    # need **kwargs to allow extra args to be passed
    def run_explicit(infile = None, **kwargs):
        if isinstance(infile, Header):
            if 'detector' not in kwargs:
                raise ValueError("Sorry, detector must be passed if supplying a header")
            if 'database' not in kwargs:
                raise ValueError("Sorry, database must be passed if supplying a header")
            detector = kwargs.pop('detector')
            database = kwargs.pop('database')
            img = database.get_images(infile, detector['image_key']['value'])[0]
            img = np.array(img)
        elif isinstance(infile, np.ndarray):
            img = infile
        elif isinstance(infile, str):
            img = np.array(Image.open(infile))
        else:
            raise ValueError("Sorry, did not understand the input argument: {}".format(infile))

        return img

class load_calibration:
    # TODO: reevaluate if _accepted_args necessary
    _accepted_args = ['calibration']
    _keymap = {'calibration' : 'calibration'}
    _output_names = ['calibration']
    _name = "XS:calibration"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self.kwargs.copy())
        new_kwargs.update(kwargs)
        return self.run_explicit(_name=self._name, **new_kwargs)

    def add(self, name=None, value=None, unit=None):
        self.kwargs.update({name : {'value' : value, 'unit' : unit}})

    @delayed(pure=True)
    @parse_sciresults(_keymap, _output_names)
    def run_explicit(calibration={}, **kwargs):
        '''
            Load calibration data.
            The data must be a dictionary.
            either:
                load_calibration(calibration=myCalib)
            or:
                load_calibration(wavelength=fdsa) etc
            
            It is curried so you can keep overriding parameters.
    
    
            This is area detector specific.
        '''
        # defaults of function
        _defaults= {'wavelength' : {'value' : None, 'unit' : 'Angstrom'},
                     'beamx0' : {'value' : None, 'unit' : 'pixel'},
                     'beamy0' : {'value' : None, 'unit' : 'pixel'},
                     'sample_det_distance' : {'value' : None, 'unit' : 'm'},
                    # Area detector specific entries:
                     # width is columns, height is rows
                     'AD_width' : {'value' : None, 'unit' : 'pixel'},
                     'AD_height' : {'value' : None, 'unit' : 'pixel'},
                     'pixel_size_x' : {'value' : None, 'unit' : 'pixel'},
                     'pixel_size_y' : {'value' : None, 'unit' : 'pixel'},
                       #TODO : This assumes data has this detector, not good to use, remove eventually
                     'detectors' : {'value' : ['pilatus300'], 'unit' : None},
    
                    }
    
        if isinstance(calibration, Header):
            # a map from Header start doc to data
            # TODO : move out of function
            calib_keymap = {'wavelength' : {'key' : 'calibration_wavelength_A',
                                            'unit' : 'Angstrom'},
                            'detectors' : {'key' : 'detectors',
                                            'unit' : 'N/A'},
                            'beamx0' : {'key' : 'detector_SAXS_x0_pix', 
                                        'unit' : 'pixel'},
                            'beamy0' : {'key' : 'detector_SAXS_y0_pix',
                                        'unit' : 'pixel'},
                            'sample_det_distance' : {'key' : 'detector_SAXS_distance_m',
                                                     'unit' : 'pixel'}
                            }
    
            start_doc = calibration['start']
            calib_tmp = dict()
            # walk through defaults
            for key, entry in calib_keymap.items():
                start_key = entry['key'] # get name of key
                unit = entry['unit']
                val = start_doc.get(start_key, _defaults[key]['value'])
                calib_tmp[key] = {'value' : val,
                                  'unit' : unit}
    
            # finally, get the width and height by looking at first detector in header
            # TODO : add ability to read more than one detector, maybe in calib_keymap
            first_detector = start_doc[calib_keymap['detectors']['key']][0]
            detector_key = detectors2D[first_detector]['image_key']['value']
    
            # look up in local library
            pixel_size_x = detectors2D[first_detector]['pixel_size_x']['value']
            pixel_size_x_unit = detectors2D[first_detector]['pixel_size_x']['unit']
            pixel_size_y = detectors2D[first_detector]['pixel_size_y']['value']
            pixel_size_y_unit = detectors2D[first_detector]['pixel_size_y']['unit']
    
            img_shape = detectors2D[first_detector]['shape']
    
            calib_tmp['pixel_size_x'] = dict(value=pixel_size_x, unit=pixel_size_x_unit)
            calib_tmp['pixel_size_y'] = dict(value=pixel_size_y, unit=pixel_size_y_unit)
            calib_tmp['shape'] = img_shape.copy() #WARNING : copies only first level, this is one level dict
            calibration = calib_tmp
        
        # update calibration with all keyword arguments
        for key, val in kwargs.items():
            # make sure not a hidden parameter
            if not key.startswith("_") and key not in calibration:
                calibration[key] = _defaults[key]
    
        return calibration
        

class circular_average:
    _accepted_args = ['calib']
    _keymap = {'calibration': 'calibration', 'image' : 'image'}
    _output_names = ['sqx', 'sqy']
    _name = "XS:circular_average"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self.kwargs.copy())
        new_kwargs.update(kwargs)
        return self.run_explicit(_name=self._name, **new_kwargs)

    @delayed(pure=True)
    @parse_sciresults(_keymap, _output_names)
    def run_explicit(image=None, calibration=None, bins=100, mask=None, **kwargs):
        #print(calibration)
        x0, y0 = calibration['beamx0']['value'], calibration['beamy0']['value']
        from skbeam.core.accumulators.binned_statistic import RadialBinnedStatistic
        img_shape = calibration['shape']['value']
        print(img_shape)
        rbinstat = RadialBinnedStatistic(img_shape, bins=bins, origin=(y0,x0), mask=mask)
        sq = rbinstat(image)
        sqx = rbinstat.bin_centers
        return sqx, sq


    


    
def test_circular_average(plot=False):
    cmsdb = databases['cms']['data']
    # I randomly chose some header
    header = cmsdb['89e8caf6-8059-43ff-9a9e-4bf461ee95b5']


    # make dummy data
    tmpdir_data = tempfile.TemporaryDirectory().name
    os.mkdir(tmpdir_data)
    img_shape = (100,100)
    data = np.ones(img_shape, dtype=np.uint8)
    data[50:60] = 0
    data_filename = tmpdir_data + "/test_data.png"
    im = Image.fromarray(data)
    im.save(data_filename)


    calibres = load_calibration(calibration=header).run()
    image = load_saxs_image(infile=header, detector=detectors2D['pilatus300'], database=cmsdb).run()
    sq = circular_average(image=image, calibration=calibres).run().compute()

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(0);plt.clf()
        plt.loglog(sq['sqx'], sq['sqy'])

    return sq

# Completed tests (above are WIP)
#
def test_sciresult_parser():
    ''' This test ensures taht 
        The inputs and outputs of functions are properly 
            normalized using SciResult.

        The inputs can be SciResults or explicit arguments
        and the output is a sciresult with only one entry
            with name _output_name
    '''
    @parse_sciresults({'a' : 'a'}, 'a')
    def foo(a=1, **kwargs):
        return a

    test = SciResult(a=1)

    res = foo(a=test)
    assert res['a'] == 1


def test_sciresult():
    ''' Just ensure instance checking is fine for SciResult.'''
    # necessary when trying to distinguish SciResult from dict
    assert_true(isinstance(SciResult(), dict))
    assert_false(isinstance(dict(), SciResult))


# this will be False, so don't do, an issue with dask
def test_delayed_passthrough():
    ''' Test that a class that inherits dict isn't improperly interpreted and
        modified.
        This is from Issue https://github.com/dask/dask/issues/2107
    '''
    class MyClass(dict):
        pass

    @delayed(pure=True)
    def foo(arg):
        assert_true(isinstance(arg, MyClass))

    res = foo(MyClass())
    res.compute()

def test_calibration():
    # TODO : Replace with a portable db to to the db testing
    cmsdb = databases['cms']['data']
    # I randomly chose some header
    header = cmsdb['89e8caf6-8059-43ff-9a9e-4bf461ee95b5']
    calibres = load_calibration(calibration=header).run()
    assert isinstance(calibres, Delayed)
    calibres = calibres.compute()
    assert isinstance(calibres, SciResult)
    print(calibres)

    calibres = load_calibration()
    calibres.add(name='beamx0', value=50, unit='pixel')
    calibres.add(name='beamy0', value=50, unit='pixel')
    calibres.run().compute()
    print(calibres)
    
def test_load_saxs_img(plot=False):
    ''' test the load_saxs_img class'''
    cmsdb = databases['cms']['data']
    # I randomly chose some header
    header = cmsdb['89e8caf6-8059-43ff-9a9e-4bf461ee95b5']

    tmpdir_data = tempfile.TemporaryDirectory().name
    os.mkdir(tmpdir_data)

    # make dummy data
    img_shape = (100,100)
    data = np.ones(img_shape, dtype=np.uint8)
    data[50:60] = 0
    data_filename = tmpdir_data + "/test_data.png"
    im = Image.fromarray(data)
    im.save(data_filename)

    # testing that protocol can take a SciResult or data
    # test with data
    res_fileinput = load_saxs_image(infile=data_filename).run()
    # test with sciresult
    head = SciResult(infile=data_filename)
    res_sciresinput = load_saxs_image(infile=head).run()

    res_headerinput = load_saxs_image(infile=header, detector=detectors2D['pilatus300'], database=cmsdb).run()

    assert_true(isinstance(res_sciresinput, Delayed))
    assert_true(isinstance(res_fileinput, Delayed))
    assert_true(isinstance(res_headerinput, Delayed))

    # test with data
    res_fileinput = res_fileinput.compute()
    # test with sciresult
    res_sciresinput = res_sciresinput.compute()
    res_headerinput = res_headerinput.compute()

    assert_array_almost_equal(data, res_fileinput['image'])
    assert_array_almost_equal(data, res_sciresinput['image'])

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(0);plt.clf()
        plt.imshow(res_headerinput['image'])
