# these read the configuration file and setup extra configuration of parameters
import yaml
import os
import os.path
import numpy as np
import numbers

# these are static things to the library
from dask import set_options
from collections import deque

print("SciStreams: Importing config")


# First step is to read yaml file from user directory
cwd = os.getcwd()
filename = cwd + "/scistreams.yml"
try:
    f = open(filename)
    config = yaml.load(f)
except FileNotFoundError:
    print("Warning, scistreams.yml in current working directory not found")
    print("Not loading any configuration.")
    config = dict()


cwd = os.getcwd()
filename_masks = cwd + "/masks.yml"
try:
    fmask = open(filename_masks)
    masks_config = yaml.load(fmask)
except FileNotFoundError:
    print("Warning, could not find {}".format(filename_masks))
    print("Will proceed without any masks")
    masks_config = {}



detector_names = dict(pilatus300='saxs', psccd='waxs', pilatus2M='saxs')


# no databases present, only add if config file gives some
default_databases = {}


_DEFAULTS = {
    'delayed': True,
    'debug': False,
    'required_attributes': dict(main=dict()),
    'default_timeout': None,
    'resultsroot': os.path.expanduser("/GPFS/xf11bm/pipeline"),
    'delayed': True,
    'server': None,
    'databases': default_databases,
    # tensorflow storage stuff
    'TFLAGS': {'out_dir': '/GPFS/xf11bm/pipeline/ml-tmp',
               'num_batches': 16},
    'logfiles' : {
        'info': '/home/xf11bm/logs/scistreams.info.log',
        'debug': '/home/xf11bm/logs/scistreams.debug.log',
    }
}

default_timeout = config.get('default_timeout', _DEFAULTS['default_timeout'])
delayed = config.get('delayed', _DEFAULTS['delayed'])
resultsroot = config.get('resultsroot', _DEFAULTS['resultsroot'])
required_attributes = config.get('required_attributes',
                                 _DEFAULTS['required_attributes'])
debug = config.get('debug', _DEFAULTS['debug'])
logfiles = config.get('logfiles', _DEFAULTS['logfiles'])

# TODO : need way of dynamically doing this
modules = config.get('modules', {})
tensorflow = modules.get('tensorflow', {})


# TODO : formalize this with some global existing python structure?
# currently used for metadata validation
typesdict = {'int': int,
             'float': float,
             'str': str,
             'number': numbers.Number,
             }


def validate_md(md, name="main", validate_dict=None):
    ''' This just validates metadata
        Cycles through internal dictionary of key : typestr
        pairs where typestr is either 'int', 'float', 'str' or something else
        etc.

        name : the name from validation dictionary to pull from
        validate_dict : an external validation dictionary to use
    '''
    # first check kwarg, then define if not
    if validate_dict is None:
        validate_dict = required_attributes.get(name, {})
    for key, val in validate_dict.items():
        if key not in md:
            errormsg = "Error, key {} not in metadata".format(key)
            raise KeyError(errormsg)
        valtype = typesdict.get(val, None)
        if valtype is None:
            errormsg = "Error, type not understood for validation"
            errormsg += "\n Please check your validation definitions for "
            errormsg += "the class {}".format(name)
            errormsg += "\nThe value type is '{}'".format(val)
            errormsg += "\nPlease ensure it looks correct."
            raise ValueError(errormsg)

        if not isinstance(md[key], typesdict[val]):
            errormsg = "Error, key {}".format(key)
            errormsg += " is not an instance of {}".format(key, val)
            raise TypeError(errormsg)

    return True


TFLAGS_tmp = dict()
TFLAGS_tmpin = config.get("TFLAGS", _DEFAULTS['TFLAGS'])
for key in _DEFAULTS['TFLAGS']:
    TFLAGS_tmp[key] = TFLAGS_tmpin.get(key, _DEFAULTS['TFLAGS'][key])


class TFLAGS:
    pass


for key, val in TFLAGS_tmp.items():
    setattr(TFLAGS, key, val)

if isinstance(resultsroot, list):
    resultsrootmap = resultsroot
    resultsroot = None
else:
    resultsrootmap = None


server = config.get('server', _DEFAULTS['server'])
databases = config.get('databases', _DEFAULTS['databases'])


if delayed:
    from dask import delayed
else:
    def delayed(pure=None, pure_default=None):
        def dec(f):
            def fnew(*args, **kwargs):
                return f(*args, **kwargs)
            return fnew
        return dec


# client information
# TODO : remove this client information

#MAX_FUTURE_NUM = 1000

if server is not None:
    try:
        print("Adding a client: {}".format(server))
        from distributed import Client
        client = Client(server)
    except ValueError:
        print("Tried to start server but failed.")
        print("Tried to connect to {}".format(server))
        print("Please remove this line in the yml file if no")
        print(" server connection is desired.")
# no client, compute should compute and return nothing
else:
    print("No client supported, running locally")

    class Client:
        # make unbound method

        def submit(self, f, *args, **kwargs):
            return f(*args, **kwargs)

        def gather(self, future):
            # it's not a future, just a regular result
            return future

    client = Client()

#futures_total = 0
class _COUNTER:
    COUNT=0
    def inc(self, *args, **kwargs):
        self.COUNT+=1
    def __call__(self):
        return self.COUNT

futures_total = _COUNTER()
futures_cache = deque()#maxlen=MAX_FUTURE_NUM)
# allow for 100,000 sinks for these (so we don't lose them)
futures_cache_sinks = deque()#maxlen=100000)

# assume all functions are pure globally
try:
    from dask.cache import Cache
    cache = Cache(1e6)
    cache.register()
except ImportError:
    print("Error cachey not available. Will not be caching")
    pass

# make everything pure by default
set_options(delayed_pure=True)

# TAU STUFF Profiler dictionaries
profile_dict = dict()

last_run_time = None

