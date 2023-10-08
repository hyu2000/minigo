""" global project wide config, including logging

Just by importing this module, logging will be config'ed
"""
import numpy as np
import logging
from logging import getLogger


logger = logging.getLogger()
# why do we need to explicitly set rootlogger level?
logger.setLevel(logging.INFO)


BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE * BOARD_SIZE
# w/ pass, but no resign
TOTAL_MOVES = BOARD_SIZE_SQUARED + 1
FULL_BOARD_FOCUS = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)

# dirs
EXP_HOME = '/Users/hyu/PycharmProjects/dlgo/9x9'
DATA_DIR = f'{EXP_HOME}/games'
FEATURES_DIR = f'{EXP_HOME}/features'
MODELS_DIR = f'{EXP_HOME}/checkpoints'
SELFPLAY_DIR = f'{EXP_HOME}/selfplay'


def config_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    logger.info('logging configed')


config_logging()
np.set_printoptions(precision=2, suppress=True)
