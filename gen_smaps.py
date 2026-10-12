from mirtorch.linear import Identity, Diff3dgram, NuSense, NuSenseGram
from mirtorch.prox import Const, L1Regularizer
from obj.RawData import RawData
from obj.FSLViewer import FSLViewer
from obj.Smap import Smaps
from obj.pblms import PBNuSense, PBNuSenseGram
from obj.helpers import *
from copy import deepcopy
import torch

# Flags
disp = True     # Display Images?
save = True     # Save Images?
verbose = True  # Print output stuff?

# Check for the best CUDA device
device = best_available_device(verbose=True)
torch.manual_seed(0)

# Instantiate FSLViewer
viewer = FSLViewer(halted=(not disp))

# Load Data
RawData.set_verbose(verbose)
y = RawData.from_dir("/home/clouly/DEV/asldata/20260731/asldata_*smap*", verbose=True)

# Make Smap Dir
#   For my workflow, they live in the parent of the data directory.
#   E.g. if the data is in './20260731/asldata_123/', the sensetivity maps will go in
#   './20260731/smaps/' (because they may be relevant to all scans located in './20260731/').
smap_dir = y.dir.parent / 'smaps'
smap_dir.mkdir(exist_ok=True)

# Clip off some navigator data
y = y.initial_clip()

# Get Smaps
# TODO: Write ESPIRIT alg (or find existing) to finish this script
