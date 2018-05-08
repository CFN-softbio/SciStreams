from SciStreams.config import logfiles

import logging
import logging.handlers
logger = logging.getLogger('scistreams')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# where to write debug messages
debug_file = logging.handlers.RotatingFileHandler(
    logfiles['debug'], maxBytes=10000000, backupCount=9)
debug_file.setLevel(logging.DEBUG)
debug_file.setFormatter(formatter)
logger.addHandler(debug_file)

# where to write info messages
info_file = logging.handlers.RotatingFileHandler(
    logfiles['info'],
    maxBytes=10000000, backupCount=9)
info_file.setLevel(logging.INFO)
info_file.setFormatter(formatter)
logger.addHandler(info_file)
