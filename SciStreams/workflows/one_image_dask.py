# TODO : make callback something else callback
# 
import matplotlib
matplotlib.use("Agg")

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from SciStreams.interfaces.plotting_mpl import store_results_mpl

# detecotr info
from SciStreams.detectors.detectors2D import detectors2D
from SciStreams.detectors.detectors2D import _make_detector_name_from_key

# for calibration
from SciStreams.data.Calibration import Calibration
from numbers import Number

# for the mask
from SciStreams.detectors.mask_generators import generate_mask

# for circular average
from SciStreams.processing.circavg import circavg

# for qphiavg
from SciStreams.processing.qphiavg import qphiavg
from SciStreams.processing.nn_fbbenet import infer_from_dict
from SciStreams.tools.image import normalizer

from SciStreams.loggers import logger

from celery.contrib import rdb

import yaml
# quick fix for a bug i dont understand
keymaps = yaml.load('''
keymaps:
    cms:
        wavelength:
            name: calibration_wavelength_A
            default_value: None
            default_unit: Angstrom
        beamx0:
            name: detector_SAXS_x0_pix
            default_value: null
            default_unit: pixel
        beamy0:
            name: detector_SAXS_y0_pix
            default_value: null
            default_unit: pixel
        sample_det_distance:
            name: detector_SAXS_distance_m
            default_value: null
            default_unit: m
        pixel_size_x:
            name:
            default_value: null
            default_unit: pixel
        pixel_size_y:
            name:
            default_value: null
            default_unit: pixel
        detector_key:
            name: detector_key
            default_value: null
            default_unit: string
                    ''')['keymaps']

# helper functions
def normalize_calib_dict(external_keymap=None, **md):
    ''' Normalize the calibration parameters to a set of parameters that the
    analysis expects.

        Parameters
        ----------
        external_keymap : dict, optional
            external keymap to use to override
            (useful for testing mainly)

        It gives entries like:
            beamx0 : dict(value=a, unit=b)
        etc...
    '''
    if external_keymap is None:
        #from SciStreams.config import config
        #keymaps = config['keymaps']
        keymap_name = md.get("keymap_name", "cms")
        keymap = keymaps[keymap_name]
    else:
        keymap = external_keymap

    # make a new dict, only choose relevant data
    new_md = dict()
    new_md.update(md)
    for key, val in keymap.items():
        # print("looking for key {}".format(val))
        name = val['name']
        if name is not None:
            # for debugging
            # print("setting {} to {}".format(name, key))
            # swap out temp vals
            tmpval = md.pop(name, val['default_value'])
            default_unit = val['default_unit']
            new_md[key] = dict(value=tmpval, unit=default_unit)
    logger.info("Finished calibration normalization")
    return new_md


def add_detector_info(**md):
    '''
        Add detector information to the metadata, like shape etc.
        This is a useful step for 2D SAXS analysis, before making the
        calibration parameters.

    Expects:

        detector_name : the detector name
        img_shape : tuple, optional
            force the image shape. This is useful when the detector image
                has been transformed (i.e. image stitching)
    '''
    detector_key = md.get('detector_key', None)
    detector_key = detector_key['value']
    # TODO : remove dict("Value" "unit") etc and replace with a general
    # descriptor (or ignore overall)
    md['detector_key'] = detector_key

    # only do something is there is a detector key
    if detector_key is not None:
        detector_name = _make_detector_name_from_key(detector_key)

        md['detector_name'] = detector_name

        # use the detector info supplied
        # look up in local library
        md['pixel_size_x'] = detectors2D[detector_name]['pixel_size_x']
        md['pixel_size_y'] = detectors2D[detector_name]['pixel_size_y']

        # shape is just a tuple, not a dict(value=...,unit=...)
        if 'shape' not in md:
            md['shape'] = detectors2D[detector_name]['shape']['value']
    else:
        msg = "Warning : no detector key found,"
        msg += " not adding detector information"
        print(msg)

    return md


def make_calibration(**md):
    '''
        Update calibration with all keyword arguments fill in the defaults

        This expects a dictionary of a certain form with certain elements:
            'wavelength'
            'pixel_size_x'
            'sample_det_distance'
            'beamx0'
            'beamy0'

        img_shape : specify arbitrary shape (useful for stitched images)
    '''
    # TODO : move detector stuff into previous load routine
    # k = 2pi/wv
    wavelength = md['wavelength']['value']  # in Angs *1e-10  # m
    try:
        md['k'] = dict(value=2.0*np.pi/wavelength, unit='1/Angstrom')
    except Exception:
        errormsg = "Error, wavelength not "
        errormsg += "supported type: {}\n".format(wavelength)
        print(errormsg)
        raise
    # energy
    # h = 6.626068e-34  # m^2 kg / s
    c = 299792458  # m/s
    # E = h*c/wavelength  # Joules
    # E *= 6.24150974e18  # electron volts
    # E /= 1000.0  # keV
    # calib_tmp.update(Singlet('energy', E, 'keV'))
    # q per pixel (Small angle limit)
    '''Gets the delta-q associated with a single pixel. This is computed in
    the small-angle limit, so it should only be considered a approximate.
    For instance, wide-angle detectors will have different delta-q across
    the detector face.'''
    pixel_size = md['pixel_size_x']['value']/1e6
    sample_det_distance = md['sample_det_distance']['value']
    try:
        c = pixel_size/sample_det_distance
    except Exception:
        errormsg = "Error, cannot divide pixel_size and sample_det_distance"
        errormsg += " values : {}/{}".format(pixel_size, sample_det_distance)
        print(errormsg)
        raise
    twotheta = np.arctan(c)  # radians
    md['q_per_pixel'] = dict(value=2.0*md['k']['value']*np.sin(twotheta/2.0),
                             unit="1/Angstrom")

    # some post calculations

    pixel_size_um = md['pixel_size_x']['value']
    distance_m = md['sample_det_distance']['value']
    wavelength_A = wavelength

    # prepare the calibration object
    if not isinstance(wavelength_A, Number) \
            or not isinstance(distance_m, Number) \
            or not isinstance(pixel_size_um, Number):
        errormsg = "Error, one of the inputs is not a number:"
        errormsg += "{}, {}, {}".format(wavelength_A, distance_m,
                                        pixel_size_um)
        print(errormsg)
        raise TypeError
    calib_object = Calibration(wavelength_A=wavelength_A,
                               distance_m=distance_m,
                               pixel_size_um=pixel_size_um)
    # NOTE : width, height reversed in calibration
    try:
        height, width = md['shape']
    except Exception:
        msg = "Error in the shape element of metadata"
        raise ValueError(msg)
    calib_object.set_image_size(width, height)
    calib_object.set_beam_position(md['beamx0']['value'],
                                   md['beamy0']['value'])
    # print("calibration object: {}".format(calib_object))
    # print("calibration object members: {}".format(calib_object.__dict__))

    return calib_object


#######



# the main input
# TODO : is this necessary or can a DAG have multiple roots?
def input_func(data):
    # pass the stream name
    logger.debug("Beginning of one image pipeline")
    data['md']['stream_name'] = 'unnamed'
    return data

# this splits images into one image to send to tasks
def to_thumb_func(data):
    logger.debug("Making thumb of image")
    data_dict = dict(img=data['img'])
    attrs = data['md']
    store_results_mpl(data_dict, attrs, images=['img'])

    # pass the stream name
    data['md']['stream_name'] = 'thumb'
    return data


def parse_attributes_func(data):
    logger.debug("Parsing attributes")
    md = data['md']
    md = normalize_calib_dict(**md)
    md = add_detector_info(**md)
    #print("parse attributes, final metadata: {}".format(md))
    data['md'] = md

    # pass the stream name
    data['md']['stream_name'] = 'unnamed'
    return data


def make_calibration_func(data):
    logger.debug("Making calibration")
    md = data['md']
    #print("making calibration from metadata {}".format(md))
    calibration = make_calibration(**md)
    calibration.generate_maps()
    #print("done")
    data['calibration'] = calibration

    # pass the stream name
    data['md']['stream_name'] = 'unnamed'
    return data

def generate_mask_func(data):
    logger.debug("Generating mask")
    md = data['md']
    mask = generate_mask(**md)['mask']
    data['mask'] = mask

    # pass the stream name
    data['md']['stream_name'] = 'unnamed'
    return data

#def save_mask_func(data, store, signal, context):
    #print("Saving thumb of mask")
    #data_dict = dict(mask=data['mask'])
    #attrs = data['md']
    #store_results_mpl(data_dict, attrs, images=['mask'])

def circavg_func(data):
    logger.debug("Computing circular average")
    image = data['img']
    calibration = data['calibration']
    q_map = calibration.q_map
    r_map = calibration.r_map
    mask = data['mask']
    #print("computing circavg")
    #print("q_map: {}".format(q_map))
    #print("r_map: {}".format(r_map))
    #print("image: {}".format(image))
    #print("mask: {}".format(mask))
    res = circavg(image, q_map=q_map, r_map=r_map, mask=mask)
    #print("done")
    data['sqx'] = res['sqx']
    data['sqxerr'] = res['sqxerr']
    data['sqy'] = res['sqy']
    data['sqyerr'] = res['sqyerr']

    # pass the stream name
    data['md']['stream_name'] = 'unnamed'
    return data

def circavg_plot_func(data):
    logger.debug("Plotting circular average")
    data_dict = dict()
    data_dict['sqx'] = data['sqx']
    data_dict['sqy'] = data['sqy']
    attrs = data['md']
    xlbl = "$q\,(\AA^\{-1\})$"
    ylbl = "$I(q)$"
    store_results_mpl(data_dict, attrs,
                      lines=[('sqx', 'sqy')],
                      scale='loglog', xlabel=xlbl,
                      ylabel=ylbl,)
    return data


# try peak finding code
def peakfind_func(data):
    logger.debug("Finding peaks")
    from SciStreams.processing.peak_finding import peak_finding

    sqx = data['sqx']
    sqy = data['sqy']

    res = peak_finding(intensity=sqy, frac=0.0001).peak_position()

    model = res[0]
    y_origin = res[1]
    inds_peak = res[2]
    xdata = res[3]
    ratio = res[4]
    ydata = res[5]
    wdata = res[6]
    bkgd = res[7]
    variance = res[8]
    variance_mean = res[9]

    peaksx = list()
    peaksy = list()

    for ind in inds_peak:
        peaksx.append(sqx[ind])
        peaksy.append(sqy[ind])

    res_dict = dict(
            model=model,
            y_origin=y_origin,
            inds_peak=inds_peak,
            xdata=xdata,
            ratio=ratio,
            ydata=ydata,
            wdata=wdata,
            bkgd=bkgd,
            variance=variance,
            variance_mean=variance_mean,
            peaksx=peaksx,
            peaksy=peaksy,
            )
    for key in res_dict.keys():
        data[key] = res_dict[key]
    data['md']['stream_name'] = 'unnamed'
    return data

def peakfind_plot_func(data):
    logger.debug("Plotting found peaks")
    data['md']['stream_name'] = 'peakfind'
    data_dict = dict()
    data_dict['sqx'] = data['sqx']
    data_dict['sqy'] = data['sqy']
    data_dict['peaksx'] = data['peaksx']
    data_dict['peaksy'] = data['peaksy']
    xlbl = "$q,(\AA^{-1})$"
    ylbl = "I(q)"
    store_results_mpl(data_dict, data['md'], lines=[dict(x='sqx', y='sqy'),
                                                    dict(x='peaksx',
                                                         y='peaksy',
                                                         marker='o', color='r',
                                                         linewidth=0)],
                      xlabel=xlbl, ylabel=ylbl, scale='loglog',)
    return data


def qphiavg_func(data):
    logger.debug("making qphiavg")
    image = data['img']
    calibration = data.get_by_alias('calibration')['calibration']
    q_map = calibration.q_map
    phi_map = calibration.angle_map
    mask = data.get_by_alias('mask')['mask']

    res = qphiavg(image, q_map=None, phi_map=None, mask=None,
                            bins=(800, 360), origin=None, range=None,
                            statistic='mean')
    data['sqphi'] = res['sqphi']
    data['qs'] = res['qs']
    data['phis'] = res['phis']

    data['md']['stream_name'] = 'unnamed'
    return data

def qphiavg_plot_func(data):
    logger.debug("Plotting qphiavg")
    data_dict = dict(sqphi = data['sqphi'])
    attr = data['md']
    attr['stream_name'] = 'qphiavg'
    store_results_mpl(data_dict, attr, img_norm=normalizer, aspect='auto',
                      images=['sqphi'],
                      xlabel="$\phi\,$(radians)", ylabel="$q\,$(pixel)",)
    return data

def inference(data):
    '''
        Runs an inference algorithm using tensorflow.
    '''
    data['tags'] = infer_from_dict(data['img'])
    return data
