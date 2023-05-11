import os

import atexit
import shutil
import tempfile
collect_ignore = ["network/hvae"]
os.environ.setdefault("DATASETS", "/local/scratch/antonlee/datasets")


def tmp_log_dir():
    """Create a temporary log directory that will be deleted after the test"""
    log_dir = tempfile.mkdtemp()
    def delete_log_dir():
        shutil.rmtree(log_dir)
    atexit.register(delete_log_dir)
    return log_dir

ROOT_LOG_DIR = tmp_log_dir()
