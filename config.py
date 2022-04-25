# General purpose global configuration information
import logging

DATASETS = "./datasets"
LOGDIR = "./experiment_logs"

def get_logger(name):
    logging.basicConfig(format='%(filename)s:%(lineno)d %(levelname)s %(message)s', level=logging.INFO)
    return logging.getLogger(name)
