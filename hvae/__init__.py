import sys

# Make sure that hvaeoodd is in the path. This is necessary because hvaeoodd
# is not a package, so it cannot be imported as a package. This __init__.py
# file is a workaround to make hvaeoodd importable as a package.
sys.path.append("hvae/hvaeoodd")

__all__ = ["hvaeoodd"]
