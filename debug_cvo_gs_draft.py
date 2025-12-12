
import yaml
import numpy as np
import logging
import sys
import os

# Adjust path to find modules
sys.path.append(os.getcwd())

import magcalc as mc
from generic_model import GenericSpinModel
import magcalc
from numpy.linalg import eigh

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_cvo_gs():
    # Load config (simplified, assuming we can load the full model)
    config_path = "aCVO/CVO_declarative.yaml" # Assuming this exists or similar
    # If not, I'll fallback to loading the manual model logic or just using the parameters if I can find them.
    # Actually the user ran `python aCVO/sw_CVO.py`. Let's look at that file first to see how it loads.
    pass

if __name__ == "__main__":
    pass
