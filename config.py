# General purpose global configuration information
import logging

DATASETS = "./datasets"
LOGDIR = "./experiment_logs"

def get_logger(name):
    logging.basicConfig(format='\x1b[36m%(module)s/%(filename)s:\x1b[34m%(lineno)d %(levelname)s \x1b[0m %(message)s', level=logging.INFO)
    return logging.getLogger(name)
