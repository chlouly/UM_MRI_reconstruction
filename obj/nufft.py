import torch
from torch import Tensor
from collections.abc import Sequence
from mirtorch.linear import LinearMap, NuSense, NuSenseGram
import torchkbnufft as tkbn
from mirtorch.linear.mri import _grid_shape, _kb_type1, _kb_type2

# class NuFFT(LinearMap):
#     r"""
#         A workaround to create a non-SENSE NuFFT LinearMap (by passing in an I as the SENSE encoding matrix)
#     """

#     def __init__(
#         self, 
#         size_in: Sequence[int], 
#         traj: Tensor, 
#         norm="ortho", 
#         batchmode=True,
#         numpoints: int | list[int] = 6,
#         grid_size: float = 2,
#         sequential: bool = False,
#         backend: str = "auto",
#         eps: float = 1e-6
#     ):
#         pass

#     def _apply(self, x):
#         pass

#     def _apply_adjoint(self, x):
#         pass


class NuFFT(LinearMap):
    """
    Non-Cartesian Fourier operator without sensitivity encoding.

    Forward:
        x: [batch, 1, nx, ny, (nz)]
        y: [batch, 1, npoints]

    Adjoint:
        y: [batch, 1, npoints]
        x: [batch, 1, nx, ny, (nz)]

    Without batchmode:
        x: [nx, ny, (nz)]
        y: [1, npoints]

    Trajectory coordinates are in radians per voxel.
    """

    def __init__(
        self,
        im_size,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
        sequential: bool = False,
    ):
        self.im_size = tuple(im_size)
        self.traj = traj
        self.norm = norm
        self.batchmode = batchmode
        self.sequential = sequential

        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")

        if norm not in ("ortho", None):
            raise ValueError("norm must be 'ortho' or None")

        if traj.ndim != (3 if batchmode else 2):
            raise ValueError("Invalid trajectory dimensions")

        if traj.shape[-2] != len(self.im_size):
            raise ValueError("Trajectory dimension mismatch")

        self.grid_size = _grid_shape(self.im_size, grid_size)

        if batchmode:
            nbatch = traj.shape[0]
            size_in = (nbatch, 1) + self.im_size
            size_out = (nbatch, 1, traj.shape[-1])
        else:
            size_in = self.im_size
            size_out = (1, traj.shape[-1])
            self.traj = traj.unsqueeze(0)

        self.A = tkbn.KbNufft(
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=numpoints,
        ).to(device=traj.device)

        self.AT = tkbn.KbNufftAdjoint(
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=numpoints,
        ).to(device=traj.device)

        super().__init__(size_in, size_out)

    def to(self, device):
        super().to(device)
        self.traj = self.traj.to(device)
        return self

    def _apply(self, x: Tensor) -> Tensor:

        image = x if self.batchmode else x[None, None]

        if self.sequential and self.batchmode:
            return torch.cat([
                _kb_type2(
                    image[i:i+1],
                    self.traj[i:i+1],
                    self.A,
                    self.AT,
                    self.norm,
                )
                for i in range(image.shape[0])
            ], dim=0)

        trajectory = self.traj

        result = _kb_type2(
            image,
            trajectory,
            self.A,
            self.AT,
            self.norm,
        )

        return result if self.batchmode else result.squeeze(0)

    def _apply_adjoint(self, y: Tensor) -> Tensor:

        samples = y if self.batchmode else y.unsqueeze(0)

        if self.sequential and self.batchmode:
            return torch.cat([
                _kb_type1(
                    samples[i:i+1],
                    self.traj[i:i+1],
                    self.A,
                    self.AT,
                    self.norm,
                )
                for i in range(samples.shape[0])
            ], dim=0)

        result = _kb_type1(
            samples,
            self.traj,
            self.A,
            self.AT,
            self.norm,
        )

        return result if self.batchmode else result.squeeze(0).squeeze(0)


    
class NuFFTGram(LinearMap):
    """
    Toeplitz approximation to the NUFFT Gram operator.

        G = A.H @ A

    Optionally supports a weighted Gram:

        G = A.H @ W @ A

    Shapes:
        batchmode=True:
            traj:    [nbatch, ndim, npoints]
            weights: [nbatch, npoints] or [npoints]
            x:       [nbatch, 1, *im_size]

        batchmode=False:
            traj:    [ndim, npoints]
            weights: [npoints]
            x:       [*im_size]
    """

    def __init__(
        self,
        im_size,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        weights: Tensor | None = None,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
    ):
        self.im_size = tuple(im_size)
        self.traj = traj
        self.norm = norm
        self.batchmode = batchmode
        self.weights = weights

        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")

        if norm not in ("ortho", None):
            raise ValueError("norm must be 'ortho' or None")

        expected_rank = 3 if batchmode else 2

        if traj.ndim != expected_rank:
            raise ValueError(
                f"traj must have rank {expected_rank}"
            )

        if traj.shape[-2] != len(self.im_size):
            raise ValueError("Trajectory dimension mismatch")

        if weights is not None:
            if weights.shape[-1] != traj.shape[-1]:
                raise ValueError("Weight and trajectory lengths differ")

            if batchmode and weights.ndim == 2:
                if weights.shape[0] != traj.shape[0]:
                    raise ValueError("Weight and trajectory batch sizes differ")
            elif weights.ndim != 1:
                raise ValueError("Invalid weight dimensions")

        self.grid_size = _grid_shape(self.im_size, grid_size)

        if batchmode:
            size_in = (traj.shape[0], 1) + self.im_size
        else:
            size_in = self.im_size

        # Trainable/inference trajectories cannot safely use
        # a precomputed, fixed Toeplitz kernel.
        self._direct = (
            traj.requires_grad
            or torch.is_inference(traj)
            or (
                weights is not None
                and (
                    weights.requires_grad
                    or torch.is_inference(weights)
                )
            )
        )

        if self._direct:
            self._operator = NuFFT(
                im_size=self.im_size,
                traj=traj,
                norm=norm,
                batchmode=batchmode,
                numpoints=numpoints,
                grid_size=grid_size,
            )

        else:
            self.toep_op = tkbn.ToepNufft()

            self.kernel = tkbn.calc_toeplitz_kernel(
                traj,
                im_size=self.im_size,
                weights=weights,
                norm=norm,
                grid_size=self.grid_size,
                numpoints=numpoints,
            )

        self._traj_version = (
            None if self._direct else traj._version
        )

        self._weights_version = (
            None
            if self._direct or weights is None
            else weights._version
        )

        super().__init__(size_in, size_in)

    def to(self, device):
        if self._direct:
            self._operator.to(device)
            self.traj = self._operator.traj
        else:
            super().to(device)

            self._traj_version = self.traj._version
            self._weights_version = (
                None if self.weights is None
                else self.weights._version
            )

        return self

    def _apply(self, x: Tensor) -> Tensor:

        # Exact NUFFT composition when parameters are trainable.
        if self._direct:
            samples = self._operator(x)

            if self.weights is not None:
                weights = self.weights

                if self.batchmode and weights.ndim == 2:
                    weights = weights.unsqueeze(1)

                samples = samples * weights

            return self._operator.adjoint(samples)

        if self.traj._version != self._traj_version:
            raise RuntimeError(
                "Trajectory changed after Toeplitz kernel "
                "construction. Recreate NuFFTGram."
            )

        if (
            self.weights is not None
            and self.weights._version != self._weights_version
        ):
            raise RuntimeError(
                "Weights changed after Toeplitz kernel "
                "construction. Recreate NuFFTGram."
            )

        image = x if self.batchmode else x[None, None]

        result = self.toep_op(
            image,
            self.kernel,
            norm=self.norm,
        )

        return result if self.batchmode else result.squeeze(0).squeeze(0)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return self._apply(x)
