import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from obj.Smap import Smaps


class FSLViewer:
    """Manage one FSLeyes window and add PyTorch tensors as NIfTI overlays."""

    def __init__(
        self,
        fsleyes: str = "fsleyes",
        halted: bool = True,
        save_dir: str | Path | None = None,
    ):
        self.fsleyes = fsleyes
        self.halted = halted
        self.tmpdir = tempfile.TemporaryDirectory()
        self.files: list[Path] = []
        self.process: subprocess.Popen | None = None
        self.set_save_dir(save_dir)

    def set_save_dir(self, save_dir: str | Path | None):
        """Set the permanent save directory; None disables permanent saving."""
        if save_dir is None:
            self.save_dir = None
        else:
            self.save_dir = Path(save_dir).expanduser()
            self.save_dir.mkdir(parents=True, exist_ok=True)
        return self

    @staticmethod
    def _filename(name: str) -> str:
        filename = Path(name).name
        if not filename.endswith((".nii", ".nii.gz")):
            filename += ".nii.gz"
        return filename

    @staticmethod
    def _make_image(
        tensor: torch.Tensor,
        affine: np.ndarray | None = None,
        tdim: int = -1,
    ) -> nib.Nifti1Image:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        data = tensor.movedim(tdim, -1).detach().cpu().numpy()

        if data.ndim not in (2, 3, 4):
            raise ValueError(
                f"Expected a 2D, 3D, or 4D tensor, got {data.ndim}D"
            )

        if np.iscomplexobj(data):
            data = np.abs(data)

        if data.dtype == np.bool_:
            data = data.astype(np.uint8)
        elif data.dtype.kind not in "iuf":
            data = data.astype(np.float32)

        if affine is None:
            affine = np.eye(4)

        return nib.Nifti1Image(data, affine)

    def _view_many(self, paths: list[Path]) -> None:
        """Send one or more overlays in a single FSLeyes invocation."""
        if not paths:
            return

        command = [
            self.fsleyes,
            "--cliserver",
            *(str(path) for path in paths),
        ]

        if self.process is None or self.process.poll() is not None:
            # The first invocation starts the server if one is not running.
            # Passing all paths together avoids a second startup-time command.
            self.process = subprocess.Popen(command)
        else:
            # Add overlays to the already-running server.
            subprocess.run(command, check=True)

    def _add_many(
        self,
        tensors: list[tuple[torch.Tensor, str]],
        affine: np.ndarray | None,
        tdim: int,
        save: bool,
    ):
        # Permanent saving does not depend on the viewing state.
        if save and self.save_dir is not None:
            for tensor, name in tensors:
                img = self._make_image(tensor, affine, tdim)
                nib.save(img, self.save_dir / self._filename(name))

        # Halted means no temporary files and no FSLeyes interaction.
        if self.halted:
            return self

        paths = []

        for tensor, name in tensors:
            img = self._make_image(tensor, affine, tdim)
            path = Path(self.tmpdir.name) / self._filename(name)
            nib.save(img, path)
            self.files.append(path)
            paths.append(path)

        # Crucially, magnitude and phase are sent in one invocation.
        self._view_many(paths)
        return self

    def add(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        affine: np.ndarray | None = None,
        tdim: int = -1,
        save: bool = True,
    ):
        """Save and/or display a tensor."""
        return self._add_many(
            [(tensor, name)],
            affine=affine,
            tdim=tdim,
            save=save,
        )

    def add_complex(
        self,
        tensor: torch.Tensor,
        name: str = "img",
        affine: np.ndarray | None = None,
        tdim: int = -1,
        save: bool = True,
    ):
        """Add magnitude and phase together as two overlays."""
        return self._add_many(
            [
                (torch.abs(tensor), f"{name}_mag"),
                (torch.angle(tensor), f"{name}_ang"),
            ],
            affine=affine,
            tdim=tdim,
            save=save,
        )

    def add_smap(
        self,
        smap: Smaps,
        name: str = "smap",
        affine: np.ndarray | None = None,
        save: bool = True,
    ):
        """Add sensitivity-map magnitude and phase together."""
        return self.add_complex(
            smap.get_tensor(),
            name=name,
            affine=affine,
            tdim=0,
            save=save,
        )

    def halt(self):
        """Disable viewing without disabling permanent saving."""
        self.halted = True
        return self

    def unhalt(self):
        """Enable viewing for subsequent additions."""
        self.halted = False
        return self

    def close(self):
        """Close the managed FSLeyes process and clean up temporary files."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

            self.process = None

        self.tmpdir.cleanup()
        return self

    def hang(self):
        input("Press Enter to close FSL and Terminate Script...")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()