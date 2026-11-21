import torch
import argparse
from torch import Tensor
from torch.types import Device
from pathlib import Path
from numbers import Number
from typing import Literal
from collections.abc import Iterable

from obj.RawData import RawData
from obj.FSLViewer import FSLViewer
from obj.Smap import Smaps
from obj.helpers import *


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
    name: str | None = None,
    save_sub: bool = False,
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
                            If name is 'img' we would get 'img_mag.nii.gz' (magnitude image) and 'img_ang.nii.gz' (phase image) 
                            saved to the cwd.
            - save_sub:     Flag to save the subtraction of the even and odd frames. Default is False, and if True and name is None, the True input is 
                            ignored. This will save the subtraction images as '{name}_sub_mag.nii.gz' and '{name}_sub_ang.nii.gz'.
            - device:       Device on which the bulk of the computation will be performed. If device is None, the best available device will be chosen.

        Returns:
            - x:    Reconstructed Image. This will be a complex torch.Tensor of shape [nf, nx, ny, nz]

        Saves:
            - Images [if name is not None]
                - {name}_mag.nii.gz    Magnitude Coil Images
                - {name}_ang.nii.gz    Phase Coil Images
            - Subtraction Images [if save_sub and name is not None]
                - {name}_sub_mag.nii.gz    Magnitude Coil Images
                - {name}_sub_ang.nii.gz    Phase Coil Images

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

    if coilwise or smaps is None:
        # If we're reconstructing coilwise, we are NOT doing a SENSE recon
        # so we throw away the smaps (or possibly the directory in which they live)
        #
        # Or, we aren't reconstructing coil images and we also dont have a sensetivity map
        # in which case we do this and the reconstruction will give us a mean coil image.
        smaps = None
    elif smaps is not None and isinstance(smaps, Tensor):
        # Just putting out sensetivity maps in this wrapper class (Smaps) that handles
        # finding and storing the matrix of virtual coil basis vecotrs.
        smaps = Smaps._from_tensors(smaps, None)
    else:
        # If we get here, smaps has to be a filepath, so we load in the sensetivity maps
        smaps = Path(smaps) if not isinstance(smaps, Path) else smaps
        smaps = Smaps.load(smaps, cdim=0, dtype=torch.complex64, device="cpu")
    

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
    # k0 correction and aquisition delay
    y = y.initial_clip(initial_clip).aqdel(aqdel).k0correct(k0correct)

    # Coil Compression
    if smaps is not None:
        smaps = smaps.compress(ccfac)
        y = y.coil_compress(smaps)

    # Mean Image
    if mean:
        y = y.flatten(1, 2).unsqueeze(2)

    # Coil Images
    if coilwise:
        y = y.coilwise()


    # --- Reconstruction --- #
    if smaps is None:
        smaps = torch.ones((y.nc,)+tuple(y.matsize), device=y.device, dtype=y._data_type)
        smaps = Smaps._from_tensors(smaps, None)

    y = y.set_mirt_shape()
    w = y.get_dcf(device=device)    

    x, _ = y.recon_CGSENSE(smaps, reg, w, device, niter, batchmode=True)

    if name is not None:
        viewer = FSLViewer(halted=True, save_dir=Path.cwd())
        viewer.add_complex(x, name, tdim=0)

        if save_sub:
            sub = x[0::2, ...] - x[1::2, ...]
            viewer.add_complex(sub, name + "_sub", tdim=0)

    return x



# If we RAN this file, not imported it.
if __name__ == "__main__":
    # DEFAULTS    
    data_dir = Path.cwd()
    smaps = None
    niter = 500
    reg = 1.
    regtype = "l2"
    coilwise = False
    resfac = 1
    ccfac = 1
    mean = False
    frames = None
    views = None
    initial_clip = 50
    k0correct = 0
    aqdel = 0
    verbose = True
    name = "img"
    save_sub = False
    device = None

    # ARGUMENT PARSER
    p = argparse.ArgumentParser(
        prog="recon",
        description="MRI Image Reconstruction User Inputs"
    )

    # DATA DIRECTORY
    p.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=data_dir,
        help="Directory where the raw data is stored."
    )


    # SENSITIVITY MAPS
    p.add_argument(
        "sense_map_path",
        type=Path,        
        nargs="?",
        default=smaps,
        help="Path to sensitivity maps to be used in reconstruction. If nothing is passed, this program looks for 'smaps.py' in the current working directory"
    )


    # ITERATIONS
    p.add_argument(
        "-n", "--num_iter",
        type=int,
        default=niter,
        help="Maximum number of conjugate gradient iterations."
    )


    # REGULARIZATION
    reg_group = p.add_mutually_exclusive_group()
    rtypes = ("l2", "l1", "tv")
    for rtype in rtypes:
        reg_group.add_argument(
            f"--{rtype}",
            dest=f"reg_{rtype}",
            type=float,
            metavar="STRENGTH",
            #action=RegularizationAction,
            #regtype=rtype,
            help=f"Use {rtype.upper()} regularization with the specified strength."
        )


    # COILWISE RECONSTRUCTION
    p.add_argument(
        "-c", "--coilwise",
        action="store_true",
        default=coilwise,
        help="Reconstruct individual coil images. All frames are pooled."
    )


    # RESOLUTION OVERSAMPLING
    p.add_argument(
        "--res",
        type=float,
        default=resfac,
        help="Resolution oversampling factor."
    )


    # COIL COMPRESSION
    p.add_argument(
        "--cc",
        type=float,
        default=ccfac,
        help="Coil compression factor."
    )


    # MEAN IMAGE
    p.add_argument(
        "-m", "--mean",
        action="store_true",
        default=mean,
        help="Pool all frames together into a mean image."
    )


    # FRAMES
    p.add_argument(
        "-f", "--frames",
        type=int,
        nargs="+",
        default=frames,
        help="Frames to reconstruct."
    )


    # VIEWS
    p.add_argument(
        "-v", "--views",
        type=int,
        nargs="+",
        default=views,
        help="Views to use in reconstruction."
    )


    # INITIAL CLIPPING
    p.add_argument(
        "-i", "--initial_clip",
        type=int,
        default=initial_clip,
        help="Number of initial samples to clip from each FID."
    )


    # K0 CORRECTION
    p.add_argument(
        "-k", "--k0correct",
        type=int,
        choices=(0, 1, 2),
        default=k0correct,
        help="FID scaling (0: none, 1: amplitude scaling, 2: phase-only scaling)."
    )


    # ACQUISITION DELAY
    p.add_argument(
        "-a", "--aqdel",
        type=float,
        default=aqdel,
        help="Acquisition delay compensation."
    )


    # VERBOSITY
    p.add_argument(
        "-s", "--silent",
        action="store_false",
        dest="verbose",
        default=verbose,
        help="Do not print to stdout."
    )


    # OUTPUT NAME
    p.add_argument(
        "-N", "--name",
        type=str,
        default=name,
        help="Name for reconstructed image file."
    )


    # SAVE SUBTRACTIONS
    p.add_argument(
        "-sub", "--save_sub",
        action="store_true",
        dest="save_sub",
        default=save_sub,
        help="Save subpraction images too"
    )


    # DEVICE
    p.add_argument(
        "-d", "--device",
        type=str,
        default=device,
        help="Torch device (e.g. cpu, cuda, cuda:0)."
    )


    # PARSE ARGUMENTS
    args = p.parse_args()

    # ASSIGN VARIABLES
    data_dir = args.data_dir
    smaps = args.sense_map_path
    niter = args.num_iter

    for rtype in rtypes:
        val = getattr(args, f"reg_{rtype}", None)
        if val is not None:
            reg = val
            regtype = rtype

    coilwise = args.coilwise
    resfac = args.res
    ccfac = args.cc
    mean = args.mean
    frames = args.frames
    views = args.views
    initial_clip = args.initial_clip
    k0correct = args.k0correct
    aqdel = args.aqdel
    verbose = args.verbose
    name = args.name
    save_sub = args.save_sub
    device = args.device

    # RUN THE RECONSTRUCTION
    recon(
        data_dir,
        smaps,
        niter,
        reg,
        regtype,
        coilwise,
        resfac,
        ccfac,
        mean,
        frames,
        views,
        initial_clip,
        k0correct,
        aqdel,
        verbose,
        name,
        save_sub,
        device,
    )

    if verbose:
        print(grn("ALL DONE!") + "\n")