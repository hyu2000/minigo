""" global project wide config, including logging

Just by importing this module, logging will be config'ed
"""
import numpy as np
import logging


BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE * BOARD_SIZE
# w/ pass, but no resign
TOTAL_MOVES = BOARD_SIZE_SQUARED + 1


def corner_focus(corner_size=BOARD_SIZE):
    """ """
    mat = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    mat[:corner_size, :corner_size] = 1
    return mat


FULL_BOARD_FOCUS = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)

# dirs
EXP_HOME = '/Users/hyu/PycharmProjects/dlgo/9x9'
DATA_DIR = f'{EXP_HOME}/games'
FEATURES_DIR = f'{EXP_HOME}/features'
MODELS_DIR = f'{EXP_HOME}/checkpoints'
SELFPLAY_DIR = f'{EXP_HOME}/selfplay'


def config_logging():
    """ import order matters. This needs to be the first to config logging!

    Otherwise it won't have any effect. We can set level explicitly, but format is still lost
    logger.setLevel(logging.INFO)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    logger = logging.getLogger()
    # why do we need to explicitly set rootlogger level?
    logger.info('logging configed')


config_logging()
np.set_printoptions(precision=2, suppress=True)
