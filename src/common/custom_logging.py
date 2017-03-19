'''
Simple adapter to the :mod:`logging` python module. 

Two aditional debug levels are defined, along with a custom logging format.
'''


import logging

logging.basicConfig(format='%(levelname)s %(asctime)s in "%(module)s.%(funcName)s" | %(message)s')
logging.DEBUG2 = 9 # Create a more detailed debug level
logging.addLevelName(logging.DEBUG2, "DEBUG2")
logging.debug2 = (lambda str: logging.log(logging.DEBUG2, str))
logging.DEBUG3 = 8 # Create a *even more* detailed debug level
logging.addLevelName(logging.DEBUG3, "DEBUG3")
logging.debug3 = (lambda str: logging.log(logging.DEBUG3, str))

MEASURE_LOGGER_NAME= 'measure_logger'

