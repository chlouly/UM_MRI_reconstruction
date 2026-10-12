#####################################################################################
#                                                                                   #
#   RawData Class Definition:                                                       #
#   Used to read and manipulate raw GE MRI data                                     #
#                                                                                   #
#   University of Michigan                                                          #
#   Christopher Louly - 2026                                                        #
#                                                                                   #
#####################################################################################



from pathlib import Path
import numpy as np
from typing import Self
import torch
from torch import Tensor, device
from numbers import Number
from collections.abc import Iterable, Iterator, Sequence
from mirtorch.linear import NuSense, NuSenseGram, Identity, LinearMap
from obj.Smap import Smaps
from copy import deepcopy
from GERecon import Pfile
from obj.helpers import *
from obj.pblms import PBNuSenseGram

# TODO:
#   - Handle Cases when KTraj Is the same from frame to frame
#   - Clean this mess up
#   - Read Dicom files maybe
#   - Fully handle the case where we mess with the sample or spatial dimsnions of tensors
#   - Modify set_mirt_shape():
#       Don't modify the internal shape of the object, just spit out the reshaped trajectory and data

ScalarLike = Tensor | Number
Device = device | str | int | None

def _is_scalar(value) -> bool:
    return isinstance(value, Number) or isinstance(value, Tensor) and value.ndim == 0


class RawData:
    r"""
        RawData:
        A class that is used to load, store, and manipulate raw MRI data. Note: in the descriptions below
            - nc [int]:     Number of receive coils
            - ns [int]:     Number of Kspace samples collected per echo
            - ne [int]:     Number of echos collected per repetition (frame)
            - nf [int]:     Number of repetitions (frames) 

        Attributes:
            - self.data ---- [torch.Tensor] [DAC units] -- SHAPE: [ns, ne, nf, nc]
                Complex Valued Tensor object that stores the raw Kspace Data from a scan.

            - self.traj ---- [torch.Tensor] [rad/vox] ---- SHAPE: [ns, ne, (nf or 1), 3]
                Real Valued Tensor Object that stores the Kspace coordinates to each sample in data.
                Note that if the third dimension has length 1, it is assumed that the same trajectory
                was used for every frame. 

            - self.fov ----- [torch.Tensor] [cm] --------- SHAPE: [3,]
                Field of View for the reconstructed image.

            - self.matsize - [torch.Tensor] [# voxels] --- SHAPE: [3,]
                Matrix size for the reconstructed image.

            - self.device -- [torch.device | str | int | None]
                The torch device on wich self.data and self.traj are stored

            - self.dir ----- [pathlib.Path | None]
                Path to the raw data directory (if any)

        Important Methods:
            - __init__():    
                Initializes a RawData object directly (without loading from disk).

            - from_dir() -> RawData:    
                Initializes a single RawData object by reading from disk at a provided data path.

            - from_glob() -> Iterator[RawData]:    
                Creates an iterator of RawData objects from a wildcard path. To save memory, 
                the RawData objects are constructed on demand, not up front.

            - to() -> RawData:
                Moves the Kspace data and trajectory to a different torch device.
    """

    verbose: bool = False           # Are we printing stuff?
    _data_type = torch.complex64    # dtype of Kspace Data
    _traj_type = torch.float32      # dtype of Kspace Trajectory
    _fov_type = torch.float32       # dtype of Image Field of View
    _matsize_type = torch.uint16    # dtype of Image Matrix Dimensions

    #####################################################################################
    #   Initialization and Data Loading                                                 #
    #####################################################################################

    def __init__(self, data: Tensor, traj: Tensor, fov: Tensor, matsize: Tensor, device: Device = "cpu", dir: Path | None = None, samp_dim: int = 0, spat_dim: int = 3):
        r"""
            Initializes a RawData Object.

            Inputs:
                - data ------ [torch.Tensor] [DAC units] -- SHAPE: [ns, ne, nf, nc]
                    Complex Valued Tensor object that stores the raw Kspace Data from a scan.
                    Automatically cast to torch.complex64.
    
                - traj ------ [torch.Tensor] [rad/vox] ---- SHAPE: [ns, ne, (nf or 1), 3]
                    Real Valued Tensor Object that stores the Kspace coordinates to each sample in data.
                    Note that if the third dimension has length 1, it is assumed that the same trajectory
                    was used for every frame. Automatically cast to torch.float32
    
                - fov ------- [torch.Tensor] [cm] --------- SHAPE: [3,]
                    Field of View for the reconstructed image. Automatically cast to torch.float32.
    
                - matsize - [torch.Tensor] [# voxels] --- SHAPE: [3,]
                    Matrix size for the reconstructed image. Automatically cast to torch.uint16
    
                - device ---- [torch.device | str | int | None]
                    The torch device on wich self.data and self.traj are stored
    
                - dir ------- [pathlib.Path | None]
                    Path to the raw data directory (if any)

                - samp_dim -- [int]
                    Sample dimension in the Kspace Data and Kspace Trajectory tensors (0: first dimension by default)

                - spat_dim -- [int]
                    Spatial Dimension in the Kspace Trajectory tensors (3: fourth dimension by default)
        """
        
        self.data = data.to(self._data_type).to(device)
        self.traj = traj.to(self._traj_type).to(device)
        self.fov = fov.to(self._fov_type)
        self.matsize = matsize.to(self._matsize_type)
        self.device = device
        self.dir=dir
        self.samp_dim = samp_dim
        self.spat_dim = spat_dim
        self.ns = self.data.shape[self.samp_dim]
        self.mirt_shape = False

    @classmethod
    def from_dir(cls, dir: str | Path | None = None) -> Self:
        r"""
            Create a RawData instance from a single directory.
            NOTE:   If there is a glob present in the path, this method reads the first matching directory.
                    If dir = None (Default case) then the method tries to read from the current working directory.

            Usage:
                y = RawData.from_dir("/path/to/data")

            Inputs:
                - dir [str | pathlib.Path | None]
                    The directory from which we attempt to read data. The directory must include:
                    - 'P*.7':           The Pfile in which the Kspace data and scan information are stored.
                    - 'ktraj*.txt':     The Text file that stores the kspace locations for a single echo.
                    - 'kviews*.txt':    The Text file that stores the kspace locations for a single echo.
    
            Returns:
                - y [RawData]
                    A new RawData instance using the data read from the three files found in the specified directory.
        """

        if dir is None:
            dir = Path.cwd()
        elif isinstance(dir, str):
            dir = Path(dir)
        elif not isinstance(dir, Path):
            raise ValueError("Path must be a string, pathlib.Path, or None.")

        if any(char in str(dir) for char in "*?["):
            matches = [p for p in dir.parent.glob(dir.name) if p.is_dir()]

            if not matches:
                raise FileNotFoundError(f"No directories match wildcard path: {dir}")

            dir = matches[0]

        return cls._from_single_dir(dir)

    @classmethod
    def from_glob(cls, pattern: str | Path) -> Iterator[Self]:
        r"""
            Create an iterator of RawData objects.
            NOTE:   To save on memory, each RawData instance is constructed when the iterator proceeds, i.e. when the 
                    for loop starts the next repetition or the __next__() method is called on the iterator. This way,
                    only one RawData object is loaded into memory at a time.

            Usage:
                for y in RawData.from_glob("/path/to/data*"):
                    y.do_stuff()

                OR

                y_itr = RawData.from_glob("/path/to/data*").__iter__()
                y_1 = y_itr.__next__()
                y_2 = y_itr.__next__()
                y_3 = y_itr.__next__()
                ...

            Inputs:
                - pattern [str | pathlib.Path]
                    The directory pattern from which we attempt to read data. The matching directories must include:
                    - 'P*.7':           The Pfile in which the Kspace data and scan information are stored.
                    - 'ktraj*.txt':     The Text file that stores the kspace locations for a single echo.
                    - 'kviews*.txt':    The Text file that stores the kspace locations for a single echo.
    
            Returns:
                - y_itr [Iterator[RawData]]
                    An iterator of new RawData instances using the data read from the three files found in the matching
                    directories.
        """
        if isinstance(pattern, str):
            pattern = Path(pattern)
        elif not isinstance(pattern, Path):
            raise ValueError("Pattern must be a string or pathlib.Path.")

        if not any(char in str(pattern) for char in "*?["):
            raise ValueError("RawData.from_glob() requires a wildcard pattern.")

        return cls._from_glob_dir(pattern)

    @classmethod
    def _from_single_dir(cls, dir: Path) -> Self:
        r"""
            Internal Method to load a RawData object from a single directory
        """
        # Get the data path
        Pf_path, kv_path, kt_path = cls._resolve_path(dir)

        # If we're being verbose
        if cls.verbose:
            print(f"Working on '{str(dir)}':")

        # Read Kspace Data and Kspace Trajectories
        data, fov, matsize = cls._read_Pfile(Pf_path)
        traj = cls._read_traj(kv_path, kt_path)
        traj = cls._scale_traj(traj, fov, matsize)

        # Return new class instance
        return cls(data, traj, fov, matsize, dir=dir)

    @classmethod
    def _from_glob_dir(cls, dir_glob: Path) -> Iterator[Self]:
        r"""
            Internal Method to load and yield a RawData object from a directory pattern
        """
        for dir_ind in sorted(dir_glob.parent.glob(dir_glob.name)): 
            # Only process directories 
            if not dir_ind.is_dir(): 
                continue 
            yield cls._from_single_dir(dir_ind)

    @staticmethod
    def _resolve_path(dir: str | Path | None = None) -> tuple[Path, Path, Path]:
        r"""
            Internal Method to look for 'P*.7', 'ktraj*.txt' and 'kviews*.txt' in a directory.
        """
        try:
            if not dir.exists():
                # Path was invalid error
                raise FileNotFoundError(f"Path does not exist: {dir}")

            if dir.is_file():
                # Path leads to file error
                raise ValueError(f"Path '{dir}' is a file, a directory was expected")

            if dir.is_dir():
                # Find Pfile
                pfiles = sorted(f for f in dir.glob("P*.7") if f.is_file())
                if not pfiles: raise FileNotFoundError(f"No file matching 'P*.7' found in: {dir}")

                # Find kviews file
                kviewsfiles = sorted(f for f in dir.glob("kviews*.txt") if f.is_file())
                if not kviewsfiles: raise FileNotFoundError(f"No file matching 'kviews*.txt' found in: {dir}")

                # Find klocs
                ktrajfiles = sorted(f for f in dir.glob("ktraj*.txt") if f.is_file())
                if not ktrajfiles: raise FileNotFoundError(f"No file matching 'ktraj*.txt' found in: {dir}")

                # Return the first one found of all 3
                return pfiles[0], kviewsfiles[0], ktrajfiles[0]
            
            raise ValueError(f"Path is neither a regular file nor a directory: {dir}")
        except OSError as error:
            raise OSError(f"Unable to process path '{dir}': {error}") from error

    @staticmethod
    def _read_Pfile(Pf_path: Path):
        r"""
            This static method loads the kspace data, field of view, and image matrix size from a P file.

            Inputs:
        """
        # TODO: Handle 2D aquisition case

        # Instantiate the Pfile Object
        p = Pfile(str(Pf_path))         # Pfile Reader Object
        md = p.MetaData()               # Pfile Metadata (dictionary)
        hdr = p.Header()                # Pfile Header (dictionary)

        ns = int(md['acquiredXRes'])    # Number of Kspace Samples
        ne = int(md['acquiredYRes'])    # Number of Echos
        nf = int(md['slices'])          # Number of Frames
        nc = int(md['channels'])        # Number of Coils

        fov = (hdr['rdb_hdr_image']['dfov'] / 10) * torch.ones((3,), dtype=RawData._fov_type)   # Image FOV [cm]
        matsize = (hdr['rdb_hdr_image']['dim_X'] * torch.ones((3,))).to(RawData._matsize_type)  # Image Matrix Size [voxels]

        # Delete these dictionaries (hdr is big and neither are needed at this point)
        del hdr, md
        clr_mem()

        # Load Kspace Data
        kdata = torch.zeros((ns, ne, nf, nc), dtype=RawData._data_type)
        for i in range(nf):
            kdata[:, :, i, :] = torch.from_numpy(p.KSpace(i, 0))

        # Remove Filler Echos and Frames
        echos_keep = (kdata != 0).any(dim=(0, 2, 3))
        frames_keep = (kdata != 0).any(dim=(0, 1, 3))
        kdata = kdata[:, echos_keep, :, :]
        kdata = kdata[:, :, frames_keep, :]

        return kdata, fov, matsize

    @staticmethod
    def _read_traj(kv_path: Path, kt_path: Path):
        r"""
            Internal Method to load a Kspace Trajectory from 'kviews*.txt' and 'ktraj*.txt'
        """
        # Load Data
        ktraj = np.loadtxt(kt_path)     # Kspace locations for each echo (before rotation) ---------------  [ns, 3]
        kviews = np.loadtxt(kv_path)    # Views File Data (Info about each echo's rotations and offsets) -  [ne x nf, 14]
                                        #       Each entry stores: [armn, shotn, echon, rz, dz, R[:]]
        nv = kviews.shape[0]

        # Get Number of Echos
        bof = np.argwhere(kviews[:, 2].astype(int).squeeze() == 0)
        ne = nv if bof.size == 1 else next((n.item() for n in bof if n != 0), -1)
        nf = int(nv / ne)

        # Check if the number of echos and frames makes sense
        if ne == -1:
            raise ValueError(f"Kviews file seems to be formatted incorrectly.")
        elif ne * nf != nv:
            raise ValueError(f"Kviews file at '{str(kv_path)}' has an irregular number of echos per frame.")

        # Get Rotation Matrices
        R = kviews[:, -9:].reshape((nf, ne, 3, 3))

        # Compute Kspace Trajectory and return it
        return torch.from_numpy(np.einsum('nmjk,lk->lnmj', R, ktraj)).swapdims(1,2)

    @staticmethod
    def _scale_traj(traj: Tensor, fov: Tensor, matsize: Tensor) -> Tensor:
        r"""
            Internal Method to convert a Kspace trajctory from [-1, 1] to [rad/vox]
        """
        slice_tup = tuple([None] * (traj.ndim - 1)) + (slice(None),)
        return (traj * (2 * torch.pi * fov)[slice_tup]) / matsize[slice_tup]
        #return (traj * (torch.pi * fov)[slice_tup]) / matsize[slice_tup]

    def to(self, device: Device) -> Self:
        self.data = self.data.to(device)
        self.traj = self.traj.to(device)
        self.device = device
        return self

    def img_dims(self):
        return tuple(self.matsize.tolist())

    @classmethod
    def set_verbose(cls, verbose: bool = True):
        r"""Set verbose?"""
        cls.verbose = verbose

    def printv(self, *args, **kwargs):
        r"""Print only if verbose flag is enabled"""
        if self.verbose: print(*args, **kwargs)

    def print_solnv(self, *args, **kwargs):
        r"""Print only if verbose flag is enabled"""
        if self.verbose: print_soln(*args, **kwargs)


    
    #####################################################################################
    #   OPERATOR OVERLOAD                                                               #
    #####################################################################################
    # TODO: Handle 3 different cases:
    #       LinearMap Object - For right multiplication
    #       RawData Object --- For everything (Check if Kspace Trajectories are the same)
    #       Tensor like ------ For everything

    def __repr__(self) -> str:
        return f"<RawData Object: On {str(self.data.device)} with shape {tuple(self.data.shape)}>"

    def __add__(self, other: Self | Tensor | ScalarLike) -> Self:
        if isinstance(other, RawData):
            RawData._same_dims(self, other)
            RawData._same_traj(self, other)
            out = deepcopy(self)
            out.data += other.data
            return out
        elif _is_scalar(other):
            out = deepcopy(self)
            out.data += other[tuple([None] * out.data.ndim)]
            return out
        elif isinstance(other, Tensor):
            out = deepcopy(self)
            out.data += other
            return out
        else:
            raise NotImplementedError

    def __mul__(self, other: Self | Tensor | ScalarLike) -> Self:
        if isinstance(other, RawData):
            RawData._same_dims(self, other)
            RawData._same_traj(self, other)
            out = deepcopy(self)
            out.data *= other.data
            return out
        elif _is_scalar(other):
            out = deepcopy(self)
            out.data *= other[tuple([None] * out.data.ndim)]
            return out
        elif isinstance(other, Tensor):
            out = deepcopy(self)
            out.data *= other
            return out
        else:
            raise NotImplementedError

    def __rmul__(self, other: Self | LinearMap | Tensor | ScalarLike) -> Self:
        if isinstance(other, RawData):
            RawData._same_dims(self, other)
            RawData._same_traj(self, other)
            out = deepcopy(self)
            out.data *= other.data
            return out
        elif isinstance(other, LinearMap):
            return other.apply(self.data)
        elif _is_scalar(other):
            out = deepcopy(self)
            out.data *= other[tuple([None] * out.data.ndim)]
            return out
        elif isinstance(other, Tensor):
            out = deepcopy(self)
            out.data *= other
            return out
        else:
            raise NotImplementedError

    def __sub__(self, other) -> Self:
        return self.__add__(-other)

    def __neg__(self) -> Self:
        return -1 * self.data



    #####################################################################################
    #   RawData SLICING                                                                 #
    #####################################################################################
    def _get_slice(self, index) -> tuple[tuple, tuple]:
        index = list(index)

        # if (len(index) >= self.spat_dim) or (len(index) >= self.samp_dim):
        #     raise ValueError("Invalid Index")

        k_index = index.copy()
        t_index = index.copy()

        t_index[self.spat_dim] = slice(None)    # Prexerve the kx, ky, kz tensor dimension for the traj inxed

        return tuple(k_index), tuple(t_index)
    
    def __getitem__(self, index) -> Self:
        k_idx, t_idx = self._get_slice(index)
        y = deepcopy(self)

        y.data = y.data[k_idx]
        y.traj = y.traj[t_idx]

        return y

    def __setitem__(self, index, value) -> Self:
        k_idx, _= self._get_slice(index)

        self.data[k_idx] = value

        return self

    def __len__(self) -> int:
        return self.data.numel


    #####################################################################################
    #   Tensor Manipulation                                                             #
    #####################################################################################

    def swapdims(self, dim0: int, dim1: int) -> Self:
        r"""
            Swap Tensor Dimensions.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.swapdims(dim0, dim1)
        self.traj = self.traj.swapdims(dim0, dim1)
        return self

    def movedim(self, source: int, destination: int) -> Self:
        r"""
            Move a Tensor Dimension.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.movedim(source, destination)
        self.traj = self.traj.movedim(source, destination)
        return self

    def flatten(self, start: int = 0, end: int = -1) -> Self:
        r"""
            Unflatten Tensor Dimension into multiple dimensions.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.flatten(start, end)
        self.traj = self.traj.flatten(start, end)
        return self
    
    def unflatten(self, dim, sizes) -> Self:
        r"""
            Flatten Tensor Dimensions into one dimension.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.unflatten(dim, sizes)
        self.traj = self.traj.unflatten(dim, sizes)
        return self
    
    def squeeze(self, dim: int | list[int] | None = None) -> Self:
        r"""
            Remove singleton dimensions from Tensor, if no dimesnion(s)
            are specified (i.e. dim=None) then all dingleton dimensions are removed.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.squeeze(dim)
        self.traj = self.traj.squeeze(dim)
        return self
    
    def unsqueeze(self, dim: int) -> Self:
        r"""
            Add a singleton dimension at dim to Tensor.
            On both Kspace Data and Kspace Trajectory
        """
        self.data = self.data.unsqueeze(dim)
        self.traj = self.traj.unsqueeze(dim)
        return self

    def initial_clip(self, ns: int = 50) -> Self:
        r"""
            Remove the first ns kspace samples (they get corrupted sometimes)
        """
        self.data = self.data[ns:]
        self.traj = self.traj[ns:]
        return self

    def retain(self, dim: int, ret: Iterable[int] | Iterable[bool]) -> Self:
        r"""
            Method that retains specific indices from the Kspace Data and locations stored in this
            object. It goes along the dimension 'dim' and keeps all indices that are specified in 'ret',
            and discards the rest.

            Inputs:
                dim:        The dimension along which we will be keeping or discarding data and sample locations
                ret:        An iterable representing the values to retain. If the values are...
                    int:        They represent the indices that we keep. For instance if dim=1 and 3 is not in ret,
                                we will discard all kspace data and locations from the fouth frame (assuming dim=1 
                                is the frame dimension), note indexing starts at 0.
                    bool:       They represent a boolean mask, so if dim=1, len(ret) must be the same as the number
                                kspace samples and locations along dimension 1. And if ret[3]=False, we again discard
                                the fourth frame (assuming dim=1 is the frame dimension), note indexing starts at 0.
        """
        # Check dimension
        if not isinstance(dim, int):
            raise TypeError("dim must be an int.")

        if not self._dim_in_range(dim):
            raise ValueError(
                f"dim is out of range. The data tensor has "
                f"{self.data.ndim} dimensions while dim is {dim}."
            )

        ret = list(ret)
        if not ret:
            raise ValueError("ret cannot be empty.")

        # Ensure ret contains exclusively ints or bools
        if all(type(v) is int for v in ret):
            # Integer index case
            if any(i < 0 or i >= self.data.shape[dim] for i in ret):
                raise ValueError(
                    f"An index in ret is out of range for axis {dim}, "
                    f"whose length is {self.data.shape[dim]}."
                )

            mask_1d = torch.zeros(
                self.data.shape[dim],
                dtype=torch.bool,
                device=self.data.device,
            )
            mask_1d[ret] = True

        elif all(type(v) is bool for v in ret):
            # Boolean mask case
            if len(ret) != self.data.shape[dim]:
                raise ValueError(
                    f"ret has length {len(ret)}, incompatible with "
                    f"axis {dim}, whose length is {self.data.shape[dim]}."
                )

            mask_1d = torch.as_tensor(
                ret,
                dtype=torch.bool,
                device=self.data.device,
            )

        else:
            raise TypeError(
                "ret must be an iterable of exclusively ints "
                "or exclusively bools."
            )

        # Apply mask
        slice_mask = [slice(None)] * self.data.ndim
        slice_mask[dim] = mask_1d
        slice_mask = tuple(slice_mask)

        self.data = self.data[slice_mask]
        self.traj = self.traj[slice_mask]

        return self

    def set_matsize(self, matsize: Sequence[int]) -> Self:
        r"""
            Method that matnually sets the Image Matrix Size and rescales the Kspace Trajectory to match.
        """
        matsize = torch.as_tensor(matsize).to(self._matsize_type)
        if torch.any(matsize <= 0):
            raise ValueError("Image Matrix Dimensions 'matsize' must only contain positive values.")

        # Rescale Kspace Trajectory
        self.traj *= self.matsize[None, None, None, :]
        self.traj /= matsize[None, None, None, :]

        # Set New Matrix Size
        self.matsize = matsize

        return self

    def resfac(self, resfac: Number) -> Self:
        r"""
            Modify the final image matrix size, effectively changing the image resolution.
        """
        if resfac <= 0:
            raise ValueError("Resolution Factor 'resfac' must be strictly positive.")

        return self.set_matsize(torch.ceil(self.matsize * resfac).to(self._matsize_type).tolist())

    def set_mirt_shape(self) -> Self:
        r"""
            Method that sets the kspace trajectory in the format that MIRTorch expects for its NuSense operator

            Data:           from  [ns, ne, nf, nc]  --to-->  [nf, nc, ns * ne]
            Trajectory:     from  [ns, ne, nf, 3]   --to-->  [nf, 3, ns * ne]
        """
        self.mirt_shape = True
        self.spat_dim = 1
        self.samp_dim = 2
        return self.movedim(-1, 0).flatten(1, 2).movedim(-1, 0)

    def coilwise(self) -> Self:
        r"""
            Method that restructures data. This is the setup step for reconstructing coil images.
            It essentially shuffles the data so that the coil dimension is treated like the frame dimension.

            Data:           from  [ns, ne, nf, nc]  --to-->  [ns, ne * nf, nc, 1]
            Trajectory:     from  [ns, ne, nf, 3]   --to-->  [ns, ne * nf, 1, 3]
        """
        self.data = self.data.unsqueeze(4)
        self.traj = self.traj.unsqueeze(3)

        return self.flatten(1, 2)

    def split_cl(self, dim: int = 2, mode=1) -> tuple[Self | None, Self | None]:
        r"""
            Method that splits this RawData Object into two RawData objects. It is meant to separate control and label
            frames for arterial spin labeling data.

            Inputs:
                dim:        The dimension of the kspace data and trajectory tensors that we will be splitting.
                mode:       The way the data will be split:
                    - 1:    (LCLC)  Even indices (and 0) along this dimension correspond to label data, and odd
                                    indices correspond to control data.
                    - 2:    (CLCL)  Even indices (and 0) along this dimension correspond to control data, and odd
                                    indices correspond to label data.
                    - 3:    (LLLL)  All indices correspond to label data (only added for completeness, to match umflexasl)
                    - 4:    (CCCC)  All indices correspond to control data (only added for completeness, to match umflexasl)
        """
        n = self.data.shape[dim]

        if mode == 1:
            # LCLC Mode
            mask_l = [i % 2 == 0 for i in range(n)]     # Label frames are even
            mask_c = [i % 2 != 0 for i in range(n)]     # Control framess are odd
        elif mode == 2:
            # CLCL Mode
            mask_l = [i % 2 != 0 for i in range(n)]     # Label frames are even
            mask_c = [i % 2 == 0 for i in range(n)]     # Control framess are odd
        elif mode == 3:
            # LLLL Mode
            #   Nothing needs to be done essentially
            #   Return a copy of this RawData object 
            #   for the label case, and None for the control.
            return (deepcopy(self), None)
        elif mode == 4:
            # LLLL Mode
            #   Nothing needs to be done essentially
            #   Return a copy of this RawData object 
            #   for the label case, and None for the control.
            return (None, deepcopy(self))
        else:
            raise ValueError("mode must be 1 (LCLC), 2 (CLCL), 3 (LLLL), or 4 (CCCC).")

        rd_l = deepcopy(self).to(self.data.device).retain(dim=dim, ret=mask_l)
        rd_c = deepcopy(self).to(self.data.device).retain(dim=dim, ret=mask_c)

        if not torch.allclose(rd_l.traj, rd_c.traj, rtol=1e-05, atol=1e-08):
            raise ValueError("You're Control and Label will not have the same kspace trajectories after split")

        return (rd_l, rd_c)

    @staticmethod
    def _window_tensor(t: Tensor, wsize: int, wdim: int, cdim: int):
        r"""
            Internal static method that windows any given torch.Tensor.

            Inputs:
                t:      torch.Tensor to be windowed
                wsize:  Window size
                wdim:   The dimension along which we will window t.
                cdim:   TODO: I dont remember what this variable is for
        """
        return t.unfold(wdim, wsize, 1).movedim(-1, cdim).flatten(start_dim=cdim, end_dim=cdim+1)

    def window(self, wsize: int, wdim: int, cdim: int):
        self.data = RawData._window_tensor(self.data ,wsize, wdim, cdim)
        self.traj = RawData._window_tensor(self.traj ,wsize, wdim, cdim)
        return self



    #####################################################################################
    #   Coil Sensetivity Related Methods                                                #
    #####################################################################################

    def estimate_smaps(self, device) -> Smaps:
        y = deepcopy(self).to(device).coilwise()

    def coil_compress(self, smap: Smaps) -> Self:
        r"""
            Do coil compression on a sensetivity map and our raw data
        """
        if smap.device != self.data.device:
            smap = smap.to(self.data.device)

        if smap.is_compressed(): self.data = torch.matmul(smap.U.mH, self.data.unsqueeze(-1)).squeeze(-1)
        
        return self



    #####################################################################################
    #   Density Compensation Factor Estimation                                          #
    #####################################################################################

    def get_dcf(self, max_iter: int = 15, device=None) -> Tensor:
        r"""
            Estimate a density compensation tensowr w from the current kspace trajectory
        """
        device = self.device if device is None else device
        self.printv(blu("DCF CALCULATION") + "...", end="")
        w, t = timed(lambda: self._get_dcf(max_iter, device), device)
        self.printv(grn(" DONE!") + f"  :  [ took {t:.2f} s , VRAM aloc {(torch.cuda.memory_allocated()/(1024 ** 3)):.4f} GiB ]")
        return w

    def _get_dcf(self, max_iter: int = 15, device=None) -> Tensor:
        r"""
            Estimate a density compensation tensowr w from the current kspace trajectory
        """
        # TODO: check if MIRT shape
        # TODO: Run Power Iter on AhA not AAh (if possible... idk)

        # Check Device
        if device is None: device = self.device

        # Object Instantiation
        w = torch.full_like(self.traj[:, 0, :], 1.0, dtype=self._data_type, device=device).unsqueeze(1)     # Weight Tensor (w_1 = vec(1))
        s = torch.full((1, 1) + self.img_dims(), 1.0, dtype=self._data_type, device=device)                 # Fake Sensetivity map
        Af = NuSense(s, self.traj).to(device)                                                               # NuFFT

        # Run Pipe Mennone DCF Algorithm
        for _ in range(max_iter):
            # TODO: Termination Criteria
            w = torch.real(w / (Af * Af.H * w)).to(self._data_type)

        del Af, s
        clr_mem()

        # Normalize and return
        return w / torch.linalg.norm(w)

    def _get_scaled_init(self, A: LinearMap, AhA: LinearMap | None = None, AHy: Tensor | None = None, w: Tensor | None = None, device: Device | None = None) -> Tensor:
        r"""
            Compute (FRAMEWISE):
                s = argmin_s |y - A(s * x_0)|_2^2

            Where:
                - s                     [Complex Scalar]:   Scalar that gives us the best initialization (one scalar per image frame)
                - x_0 := A * (w * y)    [Image Tensor]:     Density Compensated Initialization

            This optimization problem has an analytical solution:
                s = (w * y)'AA'y / (w * y)'AA'(w * y)   Where (.)' is the Conjugate Transpose and (y * w) is computed elementwise.

            Inputs:
                A   [LinearMap]:    System Operator (usually mirtorch.linear.NuSense).
                AhA [LinearMap]:    Gram Matrix of the System Operator (usually mirtorch.linear.NuSenseGram).
                                    If None is passed, we use AhA = A.H * A (much slower so pass it in).
                AHy [Tensor]:       Precomputed A.H * y. If None is passed, it gets computed here. This Tensor is likely to be
                                    used outside of the scope of this function. No need to compute it twice!
                w   [Tensor]:       Density Compensation Factor. If None is passed, no density compensation is done (i.e. w = 1).

            Returns:
                x_0 = s * A'(w * y)

        """
        # TODO: Set 'device' when None is given

        # Compute AH * y if it wasnt given
        if AHy is None:
            AHy = A.H * self.data.to(device)

        # Define the Gram Operator if none was given.
        if AhA is None:
            AhA = A.H * A

        # Are we considering multiple frames
        sumdim = tuple(range(1, AHy.ndim)) if A.batchmode else tuple(range(AHy.ndim))

        # Compute AH * (w * y)
        # NOTE: I tried to keep the resulting tensors in image space as much as possible.
        # Image space tensors tend to be far far smaller.
        if w is not None:
            # We do have a density compensation tensor!
            AHyw = A.H * (self.data.to(device) * w.to(device))
        else:
            # No Density Compensation 
            # w = 1 (tensor)  --->  A'(w * y) = A'y
            AHyw = AHy

        s = torch.real(torch.sum(AHyw * AHy.conj(), dim=sumdim))                    # s = (w * y)'AA'y
        s = s / torch.real(torch.sum(AHyw.conj() * (AhA * AHyw), dim=sumdim))       # s = (w * y)'AA'y / (w * y)'AA'(w * y)

        x0 = AHyw
        x0 *= s[(..., *([None] * (AHy.ndim - 1)))] if A.batchmode else s

        return x0                                                                   # return x_0 = s * A'(y * w)



    #####################################################################################
    #   Conjugate Gradient Descent Image Reconstruction                                 #
    #####################################################################################
    # TODO: Fix Print statements in verbose mode

    def _recon_CG_batch(self, smap: Smaps, reg: Number, w: Tensor | None = None, device: Device | None = None, max_iter: int = 500, soln_pref = "") -> Tensor:
        r"""
            Do a batched CG Reconstruction of the data using Conjugate Gradient Descent

                x = argmin_x 1/2 |Ax - y|_2^2 + reg/2 |x|_2^2

            by finding x s.t.:

                A'(Ax - y) + reg x = 0
        """
        self.printv(blu("BATCH RECON") + "... ", end="")

        # Set Device
        device = self.device if device is None else device

        # Set Tensor Shapes that MIRTorch expects
        if self.mirt_shape:
            y = deepcopy(self)
        else:
            y = deepcopy(self.set_mirt_shape())

        # Make LinearMaps for the forward model and gradient
        A = NuSense(smap[None].get_tensor().to("cpu"), y.traj.to("cpu"), sequential=True).to(device)
        AhA = PBNuSenseGram(smap[None].get_tensor().to("cpu"), y.traj.to("cpu"), batchsize=1).to(device)
        R = Identity(A.size_in).to(device)

        AHy = A.H * y.data.to(device)

        x0 = y._get_scaled_init(A, AhA, AHy, w, device)

        x, t = timed(
            lambda: CG(
                AhA + reg * R,
                alert=True,
                max_iter=max_iter,
                eval_func= lambda res: (torch.linalg.norm(res)).item()
            ).run(x0, AHy, return_info=True),
            device
        )

        self.print_solnv(x, t, res_scl=torch.linalg.norm(y.data), pref=soln_pref)

        del A, AhA, R, x0, AHy
        clr_mem()

        return x.solution.squeeze(1)

    def _recon_CG_framewise(self, smap: Smaps, reg: Number, w: Tensor | None = None, device: Device | None = None, max_iter: int = 500) -> Tensor:
        r"""
            Do a framewise CG Reconstruction of the data
        """
        self.printv(blu("FRAME RECON") + "... ")

        # Set Device
        device = self.device if device is None else device

        # Set Tensor Shapes that MIRTorch expects
        if self.mirt_shape:
            y = deepcopy(self)
        else:
            y = deepcopy(self.set_mirt_shape())

        nf = y.data.shape[0]
        x_img = torch.zeros((nf,) + tuple(self.matsize), dtype=self._data_type, device=device)

        # Loop over all frames
        for i in range(nf):
            self.printv(maj("FRAME: ") + blu(f"{i} "), end="")
            y_i = y[i, :, :]

            # Instantiate Operators
            A = NuSense(smap.get_tensor().to("cpu"), y_i.traj.to("cpu"), batchmode=False).to(device)
            AhA = NuSenseGram(smap.get_tensor().to("cpu"), y_i.traj.to("cpu"), batchmode=False).to(device)
            R = Identity(A.size_in).to(device)

            # Precompute A'y
            AHy = A.H * y_i.data.to(device)

            # Get Initialization
            x0 = y_i._get_scaled_init(A, AhA, AHy, w[i, :, :], device)

            # Perform L2 regularized CG
            x, t = timed(
                lambda: CG(
                    AhA + reg * R,
                    alert=True,
                    max_iter=max_iter,
                    eval_func= lambda res: (torch.linalg.norm(res)).item()
                ).run(x0, AHy, return_info=True),
                device
            )

            # Store solution and log metrics
            self.print_solnv(x, t, res_scl=torch.linalg.norm(y_i.data))
            x_img[i, :, :, :] = x.solution

            # Clear some old stuff
            del A, AhA, R, x0, AHy, x
            clr_mem()

        return x_img

    def _recon_CG_trybatch(self, smap: Smaps, reg: Number, w: Tensor | None = None, device: Device | None = None, max_iter: int = 500) -> tuple[Tensor, bool]:
        try:
            self.printv(blu("TRYING "), end="")
            return self._recon_CG_batch(smap, reg, w, device, max_iter), True
        except:
            # TODO: Better Error Handling for this (Catch OOM, not other stuff)
            self.printv(red("FAIL! "), end="")

        x_img, t = timed(lambda: self._recon_CG_framewise(smap, reg, w, device, max_iter), device)
        self.printv(maj(f"Total Time: ") + "[ " + blu(f"{t:.2f} s") + " ]")
        return x_img, False


    def recon_CG(self, smap: Smaps, reg: Number, w: Tensor | None = None, device: Device | None = None, max_iter: int = 500, batchmode: bool | None = None) -> tuple[Tensor, bool]:
        # How are we handling batching?
        if batchmode is None:
            # If batchmode = None (DEFAULT CASE):
            # We try to do batchmode (much faster), and if
            # it fails (probably OOM) we retry the reconstruction framewise
            return self._recon_CG_trybatch(smap, reg, w, device, max_iter)
        if batchmode:
            # If batchmode = True:
            # We have specified that we want batching (even if it runs out of memory)
            return self._recon_CG_batch(smap, reg, w, device, max_iter), True
        if not batchmode:
            # If batchmode = False:
            # We have specified that we want to reconstruct frame by frame (slower but more memory efficient)
            x_img, t = timed(lambda: self._recon_CG_framewise(smap, reg, w, device, max_iter), device)
            self.printv(maj(f"Total Time: ") + "[ " + blu(f"{t:.2f} s") + " ]")
            return x_img, False

        # If we get here, betchmode was invalid
        raise ValueError("Invalid batchmode for RawData.recon_CG()")



    #####################################################################################
    #   Validation Stuff                                                                #
    #####################################################################################
    @staticmethod
    def _same_traj(d1: Self, d2: Self, atol: float = 1e-05, rtol: float = 1e-08) -> bool:
        r"""
            Checks if two RawData objects have the same Kspace Trajectory
        """
        if not torch.allclose(d1.traj, d2.traj, rtol=rtol, atol=atol):
            raise ValueError(f"The kspace trajectories don't seem to match.")
        
        return True

    @staticmethod
    def _same_dims(d1: Self, d2: Self):
        r"""
            Checks if two RawData objects have compatible Tensor Dimensions
        """
        ddim_same = (d1.data.ndim == d2.data.ndim) and (not torch.any(d1.data.shape != d2.data.shape))
        tdim_same = (d1.traj.ndim == d2.traj.ndim) and (not torch.any(d1.traj.shape != d2.traj.shape))

        if (ddim_same) and (not tdim_same): raise ValueError(f"RawData Objets had different trajectory dimensions of {d1.traj.shape} and {d2.traj.shape}. The data dimesnions were the same.")
        if (not ddim_same) and (tdim_same): raise ValueError(f"RawData Objets had different data dimensions of {d1.data.shape} and {d2.data.shape}. The trajectory dimesnions were the same.")
        if (not ddim_same) and (not tdim_same): raise ValueError(f"RawData Objets had different data and trajectory dimensions. Data: {d1.data.shape} and {d2.data.shape}. Trajectory: {d1.traj.shape} and {d2.traj.shape}.")

        return True

    def _dim_in_range(self, dim: int) -> bool:
        return 0 <= dim < self.data.ndim

    def _idx_in_range(self, dim: int, idx: int) -> bool:
        return idx < self.data.shape[dim]