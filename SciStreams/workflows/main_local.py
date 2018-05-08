"""
    This is meant to be run locally

    This DAG is the primary DAG. It will:
        - pick out images from the stream from detectors.
        - ensure that the attributes coming in have all needed keys
            If they don't, it won'process them

        - This spawns the one_image_dag DAG for events
            with the images expected and attributes
"""

from lightflow.models import Dag, Parameters, Option
from lightflow.models import Dag
from lightflow.tasks import PythonTask
from lightflow.models.task_data import TaskData, MultiTaskData

# TODO : make callback something else callback
# 
from databroker import Broker
import matplotlib.pyplot as plt
import numpy as np

from SciStreams.workflows.one_image import one_image_dag

import numbers

parameters = Parameters([
        Option('data_folder', help='Specify data folder', type=str),
])


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



import os
import h5py
# this splits images into one image to send to tasks
def primary_func(data, store, signal, context):
    dag_names = list()

    data_folder = store.get('data_folder')
    print("Going through directory contents")

    files = os.listdir(data_folder)
    files = [data_folder + "/" + filename for filename in files
             if os.path.isfile(data_folder+'/'+filename)]

    # limit to 1 for now
    files = files[:1]
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
        #new_data = TaskData(data=new_data)
        new_data = MultiTaskData(dataset=new_data)
        good_attr = filter_attributes(new_data['md'])
        if good_attr:
            print("got a good image")
            # one image dags should go here
            dag_name = signal.start_dag(one_image_dag, data=new_data)
            print("primary node, dag name: {}".format(dag_name))
            dag_names.append(dag_name)
        else:
            print("Bad attributes!")

    signal.join_dags(dag_names)

# create the main DAG that spawns others
#img_dag = Dag('img_dag')
primary_task = PythonTask(name="primary_task",
                          callback=primary_func, queue='cms-primary-task')
primary_dag_dict = {
    primary_task: None,
    }

primary_dag = Dag("primary_dag", autostart=True, queue='cms-primary')
primary_dag.define(primary_dag_dict)
