import numpy as np

# Compatibility patch for numpy >= 1.24 where np.float, np.int, np.bool were removed
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

from .eval import Evaluator
from . import datasets
from . import metrics
from . import plotting
from . import utils
