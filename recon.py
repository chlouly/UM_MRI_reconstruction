import torch
from torch import Tensor
from torch.types import Device
from copy import deepcopy
from pathlib import Path
from numbers import Number
from typing import Literal
from collections.abc import Iterable

from obj.RawData import RawData
from obj.FSLViewer import FSLViewer
from obj.Smap import Smaps
from obj.helpers import *

def validate_indices(x, name):
    if x is None:
        return None

    try:
        x = list(x)
    except TypeError:
        raise TypeError(f"{name} must be iterable or None") from None

    if not all(type(v) in (int, bool) for v in x):
        raise TypeError(f"{name} must contain only integers or booleans")

    if x and any(type(v) != type(x[0]) for v in x):
        raise TypeError(f"{name} cannot mix integers and booleans")

    if any(v < 0 for v in x if type(v) is int):
        raise ValueError(f"{name} indices must be non-negative")

    return x

def recon(
    data_dir: str | Path | None = None,
    smaps: Tensor | str | Path | None = None,
    niter: int = 500,
    reg: Number = 0.,
    regtype: Literal["l1", "l2", "tv"] = "l2",
    coilwise: bool = False,
    resfac: Number = 1.,
    ccfac: Number = 1.,
    mean: bool = False,
    frames: Iterable[int] | None = None,
    views: Iterable[int] | None = None,
    initial_clip: int = 50,
    k0correct: int = 0,
    aqdel: int = 0,
    verbose: bool = True,
    name = str | None,
    device: Device | None = None,
) -> Tensor:
    r"""
        Standalone Image Reconstruction Function.  TODO: Make this more descriptive

        Inputs:
            - data_dir:     A string or pathlib.Path object. This is the path where the data to be reconstructed lives.
                            If None is passed, the current working directory is used instead. Default is None (read from cwd)
            - smaps:        Coil Sensetivity Maps. If a torch.Tensor is passed, this function will use it as the sensetivity maps (TODO: smap dims).
                            If a string or a pathlib.Path is passed, we try to load the sensetivity maps from that path. If None is passed, we do
                            a coil sum reconstruction instead of SENSE.
            - niter:        Number of iterations for the conjugate gradient descent reconstruction algorithm. Default is 500 iterations.
            - reg:          Regularization strength. Default is 0 (no regularization)
            - regtype:      Type of regularizer to use:
                                - "l2": l2 norm regularization (Default)
                                - "l1": l1 norm regularization (TODO: Not yet implemented)
                                - "tv": Total Variation regularization (TODO: Not yet implemented)
            - coilwise:     Reconstruct individual coil images. If coilwise == True, the value of smaps is ignored, and this is no longer a SENSE
                            reconstruction). Default is False.
            - resfac:       Image space resolution upsampling factor. Default is 1 (No resolution scaling)
            - ccfac:        Coil compression factor (i.e. ccfac = 4 will compressed data to 1/4 of the channels). Default is 1 (No coil compression)
            - mean:         Pool all frames into one 'mean image' instead of reconstructing individual frames. Default is False 
            - frames:       Frame indices to be reconstructed. Default is None (All frames will be reconstructed)
            - views:        View indices to be reconstructed. Default is None (All views will be reconstructed)
            - initial_clip: Number of Kspace samples to remove from the beginning of each echo (they can get corrupted sometimes). Default is 50.
            - k0correct:    kspace sample scaling. scale each of the FIDs (views) according to the mean signal
                            in the navigator segment (redundant points collected in the center of k space)
                                - 0:    Do nothing. (Default)
                                - 1:    Use the complex mean of the navigator segment
                                - 2:    Use only the phase of the segments.
            - aqdel:        Adjust acquisition delay by shifting the readout by aqdel points in time (it will result in losing aqdel data points 
                            per view). Default is 0
            - verbose:      Flag to print status messages while the reconstruction runs. Default is True.
            - name:         Filename for saving the reconstructed images. If this field is None, the images will not be saved, only returned.
                            The default is 'img' which would result in 'img_mag.nii.gz' (magnitude image) and 'img_ang.nii.gz' (phase image) 
                            being saved to the cwd.
            - device:       Device on which the bulk of the computation will be performed. If device is None, the best available device will be chosen.

        Outputs:
            - x:    Reconstructed Image. This will be a complex torch.Tensor of shape [nf, nx, ny, nz]

        Usage: TODO

    """
    # --- Validate Inputs --- #
    if isinstance(niter, bool) or not isinstance(niter, int):
        raise TypeError("niter must be an integer")
    if niter < 1:
        raise ValueError("niter must be positive")

    if not isinstance(reg, Number):
        raise TypeError("reg must be a number")
    if reg < 0:
        raise ValueError("reg must be non-negative")

    if regtype not in ("l1", "l2", "tv"):
        raise ValueError("regtype must be one of 'l1', 'l2', 'tv'")
    if regtype != "l2":
        # TODO: Implement "l1" and "tv"
        raise NotImplementedError(f"regtype='{regtype}' is not implemented yet")

    if not isinstance(coilwise, bool):
        raise TypeError("coilwise must be bool")

    if not isinstance(resfac, Number):
        raise TypeError("resfac must be a number")
    if resfac <= 0:
        raise ValueError("resfac must be positive")

    if not isinstance(ccfac, Number):
        raise TypeError("ccfac must be a number")
    if ccfac <= 0:
        raise ValueError("ccfac must be positive")

    if not isinstance(mean, bool):
        raise TypeError("mean must be bool")

    if isinstance(initial_clip, bool) or not isinstance(initial_clip, int):
        raise TypeError("initial_clip must be an integer")
    if initial_clip < 0:
        raise ValueError("initial_clip must be non-negative")

    if isinstance(k0correct, bool) or not isinstance(k0correct, int) or k0correct not in (0, 1, 2):
        raise TypeError("k0correct must be an integer (either 0, 1, or 2)")

    if isinstance(aqdel, bool) or not isinstance(aqdel, int):
        raise TypeError("aqdel must be an integer")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be bool")

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be str or None")

    if device is not None and not isinstance(device, (torch.device, str, int)):
        raise TypeError("device must be a torch.device, str, int, or None")


    # --- Clean Inputs --- #
    if device is None:
        # No device given, we choose the best one.
        device = best_available_device(verbose)
    else:
        # Cast it as an actual torch device, not string or int
        device = torch.device(device)

    if coilwise:
        # If we're reconstructing coilwise, we are NOT doing a SENSE recon
        # so we throw away the smaps (or possibly the directory in which they live)
        smaps = None

    if smaps is not None and isinstance(smaps, Tensor):
        # Just putting out sensetivity maps in this wrapper class (Smaps) that handles
        # finding and storing the matrix of virtual coil basis vecotrs.
        smaps = Smaps._from_tensors(smaps, None)
    else:
        # If we get here, smaps has to be a filepath, so we load in the sensetivity maps
        smaps = Smaps.load(smaps, cdim=-1, dtype=torch.complex64, device="cpu")
    

    # --- Load Data --- #
    # y.data: Tensor [ns, ne, nf, nc]
    # y.traj: Tensor [ns, ne, nf, 3]
    RawData.set_verbose(verbose)
    y = RawData.from_dir(data_dir)

    # Choose Frames to reconstruct
    if frames is not None:
        y = y.retain(dim=2, ret=frames)

    # Choose Views to reconstruct
    if views is not None:
        y = y.retain(dim=1, ret=views)

    # Clip off some of the first kspace samples
    y = y.initial_clip(initial_clip)

    # k0 correction
    y = y.k0correct(k0correct)


    # --- Reconstruction --- #
    w = y.get_dcf(device=device)

    if smaps is None:
        print()
    else:
        x = y.recon_CGSENSE(smaps, reg, w, device, niter)[0]


if __name__ == "__main__":
    # Parse Input Arguments

    img = recon()