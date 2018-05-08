"""
"""
# TODO : make callback something else callback
# 
# we don't need datbaroker
#from databroker import Broker
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

from workflows.one_image_dask import (input_func, to_thumb_func,
                                      parse_attributes_func,
                                      make_calibration_func,
                                      generate_mask_func, circavg_func,
                                      circavg_plot_func, peakfind_func,
                                      peakfind_plot_func, inference)

import numbers


# TODO : put in config files in this repo
required_attributes = {'main': {}}
typesdict = {'float': float, 'int': int, 'number': numbers.Number, 'str': str}


# filter a streamdoc with certain attributes (set in the yml file)
# required_attributes, typesdict globals needed
def filter_attributes(attr, type='main'):
    '''
        Filter attributes.

        Note that this ultimately checks that attributes match what is seen in
        yml file.
    '''
    #print("filterting attributes")
    # get the sub required attributes
    reqattr = required_attributes['main']
    for key, val in reqattr.items():
        if key not in attr:
            print("bad attributes")
            print("{} not in attributes".format(key))
            return False
        elif not isinstance(attr[key], typesdict[val]):
            print("bad attributes")
            print("key {} not an instance of {}".format(key, val))
            return False
    #print("good attributes")
    return True



# this splits images into one image to send to tasks
def primary_func(data):
    futures = list()
    data_folder = data['data_folder']
    print("Going through directory contents")

    files = os.listdir(data_folder)
    files = [data_folder + "/" + filename for filename in files
             if os.path.isfile(data_folder+'/'+filename)]

    # limit to 1 for now
    files = files[:1]
    print(files)
    detector_key = 'pilatus2M_image'
    for filename in files:
        f = h5py.File(filename, "r")

        md = dict()
        for attr, val in f['attributes'].items():
            md[attr] = val.value

        md['detector_key'] = detector_key

        img = np.array(f['img'].value)

        # give it some provenance and data
        new_data = dict(img=img)
        new_data['md'] = md.copy()
        good_attr = filter_attributes(new_data['md'])
        if good_attr:
            result = input_func(new_data)

            # thumbnail
            result2 = to_thumb_func(result)

            res_attr = parse_attributes_func(result)
            res_cal = make_calibration_func(res_attr)

            res_mask = generate_mask_func(res_attr)

            # three inputs
            input_circavg = dict()
            input_circavg.update(res_cal)
            input_circavg.update(res_mask)
            input_circavg.update(result)

            res_circavg = circavg_func(input_circavg)
            result = circavg_plot_func(res_circavg)
            res_peak = peakfind_func(res_circavg)

            res_peakplot = peakfind_plot_func(res_peak)
            res_infer = inference(new_data)
            print("Good attributes")
            #futures.append(future)
        else:
            print("Bad attributes!")
