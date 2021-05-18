""" global project wide config, including logging

Just by importing this module, logging will be config'ed
"""
import numpy as np
import logging
from logging import getLogger


#
logger = logging.getLogger()


BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE * BOARD_SIZE
# w/ pass, but no resign
TOTAL_MOVES = BOARD_SIZE_SQUARED + 1

# dirs
# raw sgfs
EXP_HOME = '/Users/hyu/PycharmProjects/dlgo/9x9'
DATA_DIR = '/Users/hyu/PycharmProjects/dlgo/9x9/games'
FEATURES_DIR = '/Users/hyu/PycharmProjects/dlgo/9x9/features'


def config_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    # logger.info('logging configed')


config_logging()
np.set_printoptions(precision=2, suppress=True)
