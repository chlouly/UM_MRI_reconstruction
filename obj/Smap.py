import torch
from torch import Tensor
from pathlib import Path
from typing import Self
from scipy.io import loadmat
from numbers import Number
import numpy as np

class Smaps(Tensor):
    r"""
        class Smaps(torch.Tensor)

        This is a Tensor wrapper class for MRI Coil Sensitivity Maps. It handles loading and coil compression
        The tensor that it wraps must be of the following shape:
            [nc, nx, ny, (nz)]
    """

    dtype = torch.complex64

    def __new__(cls, t: Tensor, cdim: int = 0) -> Self:
        r"""
            Creates a new instance of Smaps from an existing torch.Tensor instance

            Inputs:
                t:      Tensor object to wrap
                cdim:   The current location of the coil dimension
                            - if cdim=None, Nothing special happens
                            - if cdim is a valid dimension of t, that dimension
                              is moved to the 0th place (while the order of the others is preserved)
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        # Move Coil Dimesnion
        t = t.movedim(cdim, 0).to(cls.dtype)

        obj = torch.Tensor._make_subclass(
            cls,
            t,
            require_grad=t.requires_grad,
        )

        obj.U = None                # Coil Compression Matrix
        obj.nc = obj.shape[0]       # Number of Coils
        obj.ns = obj.shape[1:]      # Spatial Dimensions
        obj.nsdim = len(obj.ns)     # Number of Spatial Dimesnions

        return obj
    
    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return f"<Smaps Object: On {str(self.device)} with shape {tuple(self.shape)}. Currently {"COMPRESSED" if self.U is not None else "NOT COMPRESSED"}>"

    # @classmethod
    # def load(cls, path: str | Path, dtype: torch.dtype | None = None, device: torch.device | int | str | None = None, cdim: int = 0) -> Self:
    #     r"""
    #         Load in a tensor from a .mat file
    #     """
    #     return cls(
    #         torch.from_numpy(
    #             loadmat(path)['smap_mirt']
    #             ).to(device=device, dtype=dtype),
    #         cdim=cdim
    #     )

    @classmethod
    def load(cls, path: str | Path, dtype: torch.dtype | None = None, device: torch.device | int | str | None = None, cdim: int = 0) -> Self:
        """
        Load sensitivity maps from a .mat, .nii/.nii.gz, or saved PyTorch tensor file.
        """
        path = Path(path)
        suffix = ''.join(path.suffixes).lower()

        if path.suffix.lower() == '.mat':
            data = cls._load_mat(path)
        elif suffix.endswith(('.nii', '.nii.gz')):
            data = cls._load_nii(path)
        elif path.suffix.lower() in ('.pt', '.pth', '.torch'):
            data = cls._load_tensor(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)

        dtype = cls.dtype if dtype is None else dtype
        data = data.to(device=device, dtype=dtype)

        return cls(data, cdim=cdim)

    @staticmethod
    def _load_mat(path: Path) -> Tensor:
        """Load sensitivity maps from a MATLAB .mat file."""
        data = loadmat(path)['smap_mirt']
        return torch.from_numpy(np.asarray(data))


    @staticmethod
    def _load_nii(path: Path) -> Tensor:
        """Load sensitivity maps from a NIfTI .nii or .nii.gz file."""
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asarray(img.dataobj)

        # Handle complex-valued NIfTI data.
        if np.iscomplexobj(data):
            data = data.astype(np.complex64 if data.dtype == np.complex64 else np.complex128)
        else:
            data = data.astype(np.float32 if data.dtype.itemsize <= 4 else np.float64)

        return torch.from_numpy(data)

    @staticmethod
    def _load_tensor(path: Path) -> Tensor:
        """Load a saved PyTorch tensor from a .pt, .pth, or .torch file."""
        data = torch.load(path, map_location='cpu', weights_only=True)

        if not isinstance(data, Tensor):
            raise TypeError(
                f"Expected a saved torch.Tensor in '{path}', "
                f"got {type(data).__name__} instead."
            )

        return data
    
    @classmethod
    def _from_tensors(cls, t: Tensor, U: Tensor | None) -> Self:
        r"""
            Internal constructor (just for when V already exists)
        """
        obj = Tensor._make_subclass(
            cls,
            t,
            require_grad=t.requires_grad,
        )

        obj.U = U                   # Coil Compression Matrix
        obj.nc = obj.shape[0]       # Number of Coils
        obj.ns = obj.shape[1:]      # Spatial Dimensions
        obj.nsdim = len(obj.ns)     # Number of Spatial Dimesnions
        return obj

    def to(self, *args, **kwargs) -> Self:
        r"""
            Overloading torch.Tensor.to()
        """
        smap = super().to(*args, **kwargs)
        U = self.U.to(*args, **kwargs) if self.U is not None else None

        return Smaps._from_tensors(smap, U)

    def compress(self, ccfac: Number) -> Self:
        r"""
            Creates Coil Compression Matrix
        """
        if self.is_compressed():
            return self
        
        nc_new = int(np.ceil(self.nc / ccfac))

        if nc_new == self.nc:
            # We're not actually compressing
            return self
        
        smap_flat = self.flatten(1)
        U, _, _ = torch.svd_lowrank(smap_flat, q=nc_new)

        return Smaps._from_tensors(
            torch.matmul(U.mH, self.movedim(0, -1).unsqueeze(-1)).squeeze(-1).movedim(-1, 0).as_subclass(Tensor),
            U.as_subclass(Tensor),
        )

    def save(self, path: str | Path) -> None:
        """Save sensitivity maps based on the file extension."""
        path = Path(path)
        suffix = ''.join(path.suffixes).lower()

        if suffix == '.mat':
            self._save_mat(path)
        elif suffix in ('.nii', '.nii.gz'):
            self._save_nii(path)
        elif suffix in ('.pt', '.pth', '.torch'):
            self._save_tensor(path)
        elif suffix == '':
            self._save_tensor(str(path) + '.pt')
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


    def _save_mat(self, path: Path) -> None:
        """Save sensitivity maps to a MATLAB .mat file."""
        from scipy.io import savemat

        data = self.detach().cpu().numpy()
        savemat(path, {'smap_mirt': data})


    def _save_tensor(self, path: Path) -> None:
        """Save sensitivity maps as a PyTorch tensor."""
        torch.save(self.detach().cpu(), path)


    def _save_nii(self, path: Path) -> None:
        """Save sensitivity maps to a NIfTI .nii or .nii.gz file."""
        import nibabel as nib
        import numpy as np

        data = self.detach().cpu().numpy()

        # NIfTI supports complex data, but normalize unsupported dtypes.
        if np.iscomplexobj(data):
            data = data.astype(np.complex64 if data.dtype == np.complex64 else np.complex128)
        elif data.dtype not in (np.dtype('uint8'), np.dtype('int16'),
                                np.dtype('int32'), np.dtype('float32'),
                                np.dtype('float64')):
            data = data.astype(np.float32)

        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, str(path))

    def get_tensor(self) -> Tensor:
        return self.as_subclass(Tensor)

    def is_compressed(self) -> bool:
        return self.U is not None