# General purpose global configuration information
import typing as t
from dataclasses import dataclass, asdict
import logging

from experiment.scenario import scenario

def get_logger(name):
    # https://github.com/omz/Pythonista-Issues/issues/243 idk?
    logging.root.handlers=[]
    logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d %(levelname)s %(message)s')
    return logging.getLogger(name)
