# General purpose global configuration information
import logging

DATASETS = "./datasets"
LOGDIR = "./experiment_logs"

def get_logger(name):
    # https://github.com/omz/Pythonista-Issues/issues/243 idk?
    logging.root.handlers=[]
    logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d %(levelname)s %(message)s')
    return logging.getLogger(name)
