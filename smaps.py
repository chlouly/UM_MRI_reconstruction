import torch
import argparse
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device
from copy import deepcopy
from pathlib import Path
from numbers import Number
from typing import Literal
from collections.abc import Iterable

from recon import recon
from obj.FSLViewer import FSLViewer
from obj.helpers import *

def espirit(
    coil_images,
    calib_width=20,
    kernel_width=5,
    thresh=0.02,
    crop=0.,
    max_iter=50,
    kernel_batch=4,
):
    """
    GPU ESPIRiT calibration from reconstructed coil images.

    Parameters
    ----------
    coil_images : complex Tensor
        [ncoils, nx, ny, nz]

    Returns
    -------
    smaps : complex Tensor
        [ncoils, nx, ny, nz]
    eigenvalue : real Tensor
        [nx, ny, nz]
    """

    device = coil_images.device
    nc, nx, ny, nz = coil_images.shape
    k = kernel_width
    dims = (-3, -2, -1)

    with torch.no_grad():

        # --------------------------------------------------
        # 1. Coil images -> centered Cartesian k-space
        # --------------------------------------------------

        kspace = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(coil_images, dim=dims),
                dim=dims,
                norm="ortho",
            ),
            dim=dims,
        )

        # --------------------------------------------------
        # 2. Extract calibration region
        # --------------------------------------------------

        starts = [
            (n - calib_width) // 2
            for n in (nx, ny, nz)
        ]

        calib = kspace[
            :,
            starts[0]:starts[0] + calib_width,
            starts[1]:starts[1] + calib_width,
            starts[2]:starts[2] + calib_width,
        ]

        # --------------------------------------------------
        # 3. Calibration matrix
        # --------------------------------------------------

        patches = (
            calib
            .unfold(1, k, 1)
            .unfold(2, k, 1)
            .unfold(3, k, 1)
        )

        A = patches.permute(
            1, 2, 3, 0, 4, 5, 6
        ).reshape(-1, nc * k**3)

        # --------------------------------------------------
        # 4. SVD and kernel selection
        # --------------------------------------------------

        _, S, Vh = torch.linalg.svd(
            A, full_matrices=False
        )

        kernels = Vh[S > thresh * S[0]].reshape(
            -1, nc, k, k, k
        )

        # --------------------------------------------------
        # 5. Construct ESPIRiT covariance matrix
        # --------------------------------------------------

        # Shape: [nx, ny, nz, nc, nc]
        AHA = torch.zeros(
            nx, ny, nz, nc, nc,
            dtype=coil_images.dtype,
            device=device,
        )

        pad = []
        for n in (nz, ny, nx):
            left = (n - k) // 2
            right = n - k - left
            pad.extend([left, right])

        for batch in kernels.split(kernel_batch):

            # Center-pad kernels to image dimensions.
            padded = F.pad(batch, pad)

            # Centered inverse FFT.
            img_kernels = torch.fft.fftshift(
                torch.fft.ifftn(
                    torch.fft.ifftshift(padded, dim=dims),
                    dim=dims,
                    norm="ortho",
                ),
                dim=dims,
            )

            # Sum outer products across calibration kernels.
            AHA += torch.einsum(
                "q c x y z, q d x y z -> x y z c d",
                img_kernels,
                img_kernels.conj(),
            )

        AHA *= (nx * ny * nz) / k**3

        # --------------------------------------------------
        # 6. Principal eigenvector via power iteration
        # --------------------------------------------------

        v = torch.ones(
            nx, ny, nz, nc,
            dtype=coil_images.dtype,
            device=device,
        )

        v = F.normalize(v, dim=-1)

        for _ in range(max_iter):
            v = torch.einsum(
                "...ij,...j->...i", AHA, v
            )
            v = F.normalize(v, dim=-1)

        # Maximum eigenvalue (Rayleigh quotient).
        Av = torch.einsum(
            "...ij,...j->...i", AHA, v
        )

        eigenvalue = (
            v.conj() * Av
        ).sum(dim=-1).real

        # --------------------------------------------------
        # 7. Phase normalization and eigenvalue cropping
        # --------------------------------------------------

        smaps = v.permute(3, 0, 1, 2)

        phase = smaps[0].conj() / (
            smaps[0].abs() + 1e-8
        )

        smaps = smaps * phase.unsqueeze(0)

        smaps = smaps * (eigenvalue > crop).unsqueeze(0)

        return smaps, eigenvalue



def gen_smaps(
    data_dir: str | Path | None = None,
    niter: int = 500,
    reg: Number = 0.,
    regtype: Literal["l1", "l2", "tv"] = "l2",
    resfac: Number = 1.,
    frames: Iterable[int] | None = None,
    views: Iterable[int] | None = None,
    initial_clip: int = 50,
    k0correct: int = 0,
    aqdel: int = 0,
    verbose: bool = True,
    name: str | None = None,
    save_cims: bool = False,
    save_nii: bool = False,
    device: Device | None = None,
) -> Tensor:
    r"""
        Standalone Coil Sensetivity Estimation Function.  TODO: Make this more descriptive

        This function uses recon(), an end to end conjugate gradient descent image reconstruction function, to reconstruct coil images from a 
        kspace data. The coil images are then passed through ESPIRiT to estimate coil sensetivities (to be used in other image reconstructions).

        Inputs:
            - data_dir:     A string or pathlib.Path object. This is the path where the data to be reconstructed lives.
                            If None is passed, the current working directory is used instead. Default is None (read from cwd)
            - niter:        Number of iterations for the conjugate gradient descent reconstruction algorithm. Default is 500 iterations.
            - reg:          Regularization strength. Default is 0 (no regularization)
            - regtype:      Type of regularizer to use:
                                - "l2": l2 norm regularization (Default)
                                - "l1": l1 norm regularization (TODO: Not yet implemented)
                                - "tv": Total Variation regularization (TODO: Not yet implemented)
            - resfac:       Image space resolution upsampling factor. Default is 1 (No resolution scaling)
            - ccfac:        Coil compression factor (i.e. ccfac = 4 will compressed data to 1/4 of the channels). Default is 1 (No coil compression)
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
            - name:         Filename for saving the reconstructed images as .pt file. If this field is None (Default), the maps will not be saved. 
            - save_cims:    Flag to save the coil images before the coil sensetivity is estimated. Default is False. If name is None, passing True 
                            will be ignored for this parameter.
            - save_nii:     Flag to save the coil sensetivity maps as .nii.gz files aswell as the .py file. Default is False. If name is None, passing 
                            True will be ignored for this parameter.
            - device:       Device on which the bulk of the computation will be performed. If device is None, the best available device will be chosen.

        Returns:
            - s:    Reconstructed coil sensetivity maps. This will be a complex torch.Tensor of shape [nc, nx, ny, nz]

        Saves:
            - '{name}.pt'   torch.Tensor of complex floats containing the estimated coil sensetivity maps [nc, nx, ny, nz]
            - Coil Images [if save_cims and name is not None]
                - {name}_cims_mag.nii.gz    Magnitude Coil Images
                - {name}_cims_ang.nii.gz    Phase Coil Images
            - Sensetivity Maps [if save_nii and name is not None]
                - {name}_mag.nii.gz         Magnitude Coil Sensetivity Maps
                - {name}_ang.nii.gz         Phase Coil Sensetivity Maps

        Usage: TODO
    """
    # --- Validate Inputs --- #
    # Pretty much everything gets validated in recon()
    # We only have to validate a few things
    if not isinstance(save_cims, bool):
        raise TypeError("save_cims must be bool")

    if not isinstance(save_nii, bool):
        raise TypeError("save_nii must be bool")
    
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be str or None")

    # If we're saving coil images
    if name is not None and save_cims:
        cims_name = name + "_cims"
    else:
        cims_name = None

    if device is None:
        # No device given, we choose the best one.
        device = best_available_device(verbose)
    else:
        # Cast it as an actual torch device, not string or int
        device = torch.device(device)


    # --- Reconstruct Coil Images --- #
    cims = recon(
        data_dir,
        None,
        niter,
        reg,
        regtype,
        True,
        resfac,
        1.,
        False,
        frames,
        views,
        initial_clip,
        k0correct,
        aqdel,
        verbose,
        cims_name,
        False,
        device
    )

    # --- Estimate Coil Sensetivity --- #
    smaps, _ = espirit(cims.to(device))

    if name is not None:
        torch.save(smaps, name + ".pt")

        if save_nii:
            viewer = FSLViewer(halted=True, save_dir=Path.cwd())
            viewer.add_complex(smaps, name, tdim=0)

    return smaps



# If we RAN this file, not imported it.
if __name__ == "__main__":
    # DEFAULTS    
    data_dir = Path.cwd()
    niter = 500
    reg = 0.1
    regtype = "l2"
    resfac = 1
    frames = None
    views = None
    initial_clip = 50
    k0correct = 0
    aqdel = 0
    verbose = True
    name = "smaps"
    save_cims = False
    save_nii = True
    device = None

    # ARGUMENT PARSER
    p = argparse.ArgumentParser(
        prog="smaps",
        description="MRI sensetivity map estimation tool"
    )

    # DATA DIRECTORY
    p.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=data_dir,
        help="Directory where the raw data is stored."
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


    # RESOLUTION OVERSAMPLING
    p.add_argument(
        "--res",
        type=float,
        default=resfac,
        help="Resolution oversampling factor."
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


    # SAVE COIL IMAGES
    p.add_argument(
        "-cims", "--save_cims",
        action="store_true",
        dest="save_cims",
        default=save_cims,
        help="Save Coil Images"
    )


    # SAVE NIFTI
    p.add_argument(
        "-nii", "--save_nii",
        action="store_true",
        dest="save_nii",
        default=save_nii,
        help="Save nifti"
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
    niter = args.num_iter

    for rtype in rtypes:
        val = getattr(args, f"reg_{rtype}", None)
        if val is not None:
            reg = val
            regtype = rtype

    resfac = args.res
    frames = args.frames
    views = args.views
    initial_clip = args.initial_clip
    k0correct = args.k0correct
    aqdel = args.aqdel
    verbose = args.verbose
    name = args.name
    save_cims = args.save_cims
    save_nii = args.save_nii
    device = args.device

    # RUN THE RECONSTRUCTION
    gen_smaps(
        data_dir,
        niter,
        reg,
        regtype,
        resfac,
        frames,
        views,
        initial_clip,
        k0correct,
        aqdel,
        verbose,
        name,
        save_cims,
        save_nii,
        device,
    )

    if verbose:
        print(grn("ALL DONE!") + "\n")