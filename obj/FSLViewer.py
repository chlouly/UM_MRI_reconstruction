# # import subprocess
# # import tempfile
# # import time
# # from pathlib import Path

# # import nibabel as nib
# # import numpy as np
# # import torch
# # from obj.Smap import Smaps


# # class FSLViewer:
# #     """
# #     Manage a single FSLeyes window and dynamically add PyTorch tensors
# #     as NIfTI overlays.

# #     Parameters
# #     ----------
# #     fsleyes : str
# #         FSLeyes executable.
# #     save_dir : str or Path, optional
# #         Directory where tensors are permanently saved as NIfTI files.
# #         Can be changed later with set_save_dir(), or set to None to
# #         disable permanent saving.
# #     """

# #     def __init__(self, fsleyes="fsleyes", halted=False, save_dir=None):
# #         self.halted = halted
# #         self.fsleyes = fsleyes

# #         self.tmpdir = tempfile.TemporaryDirectory()
# #         self.files = []
# #         self.process = None

# #         self.save_dir = None
# #         self.set_save_dir(save_dir)

# #     def set_save_dir(self, save_dir):
# #         """
# #         Change the directory used for permanently saved NIfTI files.

# #         Parameters
# #         ----------
# #         save_dir : str, Path, or None
# #             Directory to save NIfTI files to. Set to None to disable
# #             permanent saving.
# #         """
# #         if save_dir is None:
# #             self.save_dir = None
# #         else:
# #             self.save_dir = Path(save_dir)
# #             self.save_dir.mkdir(parents=True, exist_ok=True)

# #         return self

# #     def _launch(self):
# #         """Launch FSLeyes if it is not already running."""
# #         if self.process is not None and self.process.poll() is None:
# #             return

# #         self.process = subprocess.Popen(
# #             [self.fsleyes, "--cliserver"]
# #         )

# #         # Give FSLeyes a moment to initialise its CLI server.
# #         time.sleep(1.0)

# #         if self.process.poll() is not None:
# #             self.process = None
# #             raise RuntimeError("FSLeyes exited during startup.")

# #     def add(
# #         self,
# #         tensor: torch.Tensor,
# #         name: str = "tensor",
# #         affine: np.ndarray | None = None,
# #         tdim: int = -1,
# #         save: bool = True,
# #     ):
# #         """
# #         Optionally save a tensor permanently and/or display it in FSLeyes.

# #         Parameters
# #         ----------
# #         tensor : torch.Tensor
# #             Tensor to save and/or display.
# #         name : str
# #             Base filename or filename ending in .nii or .nii.gz.
# #         affine : np.ndarray, optional
# #             4x4 NIfTI affine. Defaults to the identity matrix.
# #         tdim : int
# #             Tensor dimension moved to the final position before conversion.
# #         save : bool
# #             If True, save permanently when save_dir is configured.
# #             Independent of whether viewing is halted.
# #         """
# #         if not isinstance(tensor, torch.Tensor):
# #             raise TypeError("tensor must be a torch.Tensor")

# #         # Nothing to do if neither saving nor viewing is requested.
# #         if (not save or self.save_dir is None) and self.halted:
# #             return self

# #         data = tensor.movedim(tdim, -1).detach().cpu().numpy()

# #         if data.ndim not in (2, 3, 4):
# #             raise ValueError(
# #                 f"Expected 2D, 3D, or 4D tensor, got {data.ndim}D"
# #             )

# #         if affine is None:
# #             affine = np.eye(4)

# #         # Convert unsupported dtypes.
# #         if data.dtype == np.bool_:
# #             data = data.astype(np.uint8)
# #         elif data.dtype.kind not in "iufc":
# #             data = data.astype(np.float32)

# #         # Convert complex data to magnitude.
# #         if np.iscomplexobj(data):
# #             data = np.abs(data)

# #         img = nib.Nifti1Image(data, affine)

# #         # Avoid duplicate extensions.
# #         filename = Path(name).name
# #         if not filename.endswith((".nii", ".nii.gz")):
# #             filename += ".nii.gz"

# #         # Permanent saving is independent of viewing.
# #         if save and self.save_dir is not None:
# #             save_path = self.save_dir / filename
# #             nib.save(img, save_path)

# #         # Only create a temporary file and launch FSLeyes when viewing.
# #         if not self.halted:
# #             tmp_path = Path(self.tmpdir.name) / filename
# #             nib.save(img, tmp_path)
# #             self.files.append(tmp_path)

# #             self._launch()
# #             self._send(str(tmp_path))

# #         return self

# #     def add_smap(
# #         self,
# #         smap: Smaps,
# #         name: str = "smap",
# #         affine: np.ndarray | None = None,
# #         save: bool = True,
# #     ):
# #         """Add sensitivity-map magnitude and phase."""
# #         return self.add_complex(
# #             smap.get_tensor(),
# #             name=name,
# #             affine=affine,
# #             tdim=0,
# #             save=save,
# #         )

# #     def add_complex(
# #         self,
# #         t: torch.Tensor,
# #         name: str = "img",
# #         affine: np.ndarray | None = None,
# #         tdim: int = -1,
# #         save: bool = True,
# #     ):
# #         """Add magnitude and phase as separate overlays."""
# #         self.add(
# #             torch.abs(t),
# #             name + "_mag",
# #             affine,
# #             tdim=tdim,
# #             save=save,
# #         )
# #         self.add(
# #             torch.angle(t),
# #             name + "_ang",
# #             affine,
# #             tdim=tdim,
# #             save=save,
# #         )
# #         return self

# #     def _send(self, path: str):
# #         """Send an overlay to the existing FSLeyes CLI server."""
# #         subprocess.run(
# #             [self.fsleyes, "--cliserver", path],
# #             check=True,
# #         )

# #     def halt(self):
# #         """Stop displaying new images; permanent saving remains enabled."""
# #         self.halted = True
# #         return self

# #     def unhalt(self):
# #         """Enable viewing; FSLeyes launches when the next image is added."""
# #         self.halted = False
# #         return self

# #     def close(self):
# #         """Close the FSLeyes process and clean up temporary files."""
# #         if self.process is not None:
# #             if self.process.poll() is None:
# #                 self.process.terminate()
# #                 self.process.wait()

# #             self.process = None

# #         self.tmpdir.cleanup()
# #         return self

# #     def __enter__(self):
# #         return self

# #     def __exit__(self, exc_type, exc_value, traceback):
# #         self.close()


# import subprocess
# import tempfile
# import time
# from pathlib import Path

# import nibabel as nib
# import numpy as np
# import torch

# from obj.Smap import Smaps


# class FSLViewer:
#     """
#     Manage a single FSLeyes window and dynamically add PyTorch tensors
#     as NIfTI overlays.

#     Parameters
#     ----------
#     fsleyes : str
#         FSLeyes executable.
#     halted : bool
#         If True, viewing is initially disabled. Saving remains independent.
#     save_dir : str or Path, optional
#         Directory where tensors are permanently saved as NIfTI files.
#         Can be changed later with set_save_dir(), or set to None to
#         disable permanent saving.
#     """

#     def __init__(
#         self,
#         fsleyes="fsleyes",
#         halted=True,
#         save_dir=None,
#     ):
#         self.halted = halted
#         self.fsleyes = fsleyes

#         self.tmpdir = tempfile.TemporaryDirectory()
#         self.files = []
#         self.process = None

#         self.save_dir = None
#         self.set_save_dir(save_dir)

#     def set_save_dir(self, save_dir):
#         """
#         Change the directory used for permanently saved NIfTI files.

#         Parameters
#         ----------
#         save_dir : str, Path, or None
#             Directory to save NIfTI files to. Set to None to disable
#             permanent saving.
#         """
#         if save_dir is None:
#             self.save_dir = None
#         else:
#             self.save_dir = Path(save_dir)
#             self.save_dir.mkdir(parents=True, exist_ok=True)

#         return self

#     def _launch(self, path: str | Path) -> bool:
#         """
#         Launch the initial FSLeyes instance with its first overlay.

#         Returns
#         -------
#         bool
#             True if a new process was launched; False if the process
#             managed by this object is already running.
#         """
#         if self.process is not None and self.process.poll() is None:
#             return False

#         self.process = subprocess.Popen(
#             [self.fsleyes, "--cliserver", str(path)]
#         )

#         # Allow FSLeyes time to initialise its CLI server.
#         time.sleep(1.0)

#         if self.process.poll() is not None:
#             self.process = None
#             raise RuntimeError("FSLeyes exited during startup.")

#         return True

#     def _send(self, path: str | Path):
#         """Send an overlay to an existing FSLeyes CLI server."""
#         subprocess.run(
#             [self.fsleyes, "--cliserver", str(path)],
#             check=True,
#         )

#     def add(
#         self,
#         tensor: torch.Tensor,
#         name: str = "tensor",
#         affine: np.ndarray | None = None,
#         tdim: int = -1,
#         save: bool = True,
#     ):
#         """
#         Optionally save a tensor permanently and/or display it in FSLeyes.

#         Parameters
#         ----------
#         tensor : torch.Tensor
#             Tensor to save and/or display.
#         name : str
#             Base filename or filename ending in .nii or .nii.gz.
#         affine : np.ndarray, optional
#             4x4 NIfTI affine. Defaults to the identity matrix.
#         tdim : int
#             Tensor dimension moved to the final position before conversion.
#         save : bool
#             If True, save permanently when save_dir is configured.
#             Independent of whether viewing is halted.
#         """
#         if not isinstance(tensor, torch.Tensor):
#             raise TypeError("tensor must be a torch.Tensor")

#         # If neither saving nor viewing is requested, do nothing.
#         if (not save or self.save_dir is None) and self.halted:
#             return self

#         data = tensor.movedim(tdim, -1).detach().cpu().numpy()

#         if data.ndim not in (2, 3, 4):
#             raise ValueError(
#                 f"Expected 2D, 3D, or 4D tensor, got {data.ndim}D"
#             )

#         if affine is None:
#             affine = np.eye(4)

#         # Convert unsupported dtypes.
#         if data.dtype == np.bool_:
#             data = data.astype(np.uint8)
#         elif data.dtype.kind not in "iufc":
#             data = data.astype(np.float32)

#         # Display complex tensors as magnitude.
#         if np.iscomplexobj(data):
#             data = np.abs(data)

#         img = nib.Nifti1Image(data, affine)

#         # Avoid duplicate extensions.
#         filename = Path(name).name
#         if not filename.endswith((".nii", ".nii.gz")):
#             filename += ".nii.gz"

#         # Permanent saving is independent of viewing.
#         if save and self.save_dir is not None:
#             save_path = self.save_dir / filename
#             nib.save(img, save_path)

#         # Only create temporary files and launch/send to FSLeyes if unhalted.
#         if not self.halted:
#             tmp_path = Path(self.tmpdir.name) / filename
#             nib.save(img, tmp_path)
#             self.files.append(tmp_path)

#             # The first image launches FSLeyes with that image.
#             # Subsequent images are sent to the existing CLI server.
#             started = self._launch(tmp_path)
#             if not started:
#                 self._send(tmp_path)

#         return self

#     def add_smap(
#         self,
#         smap: Smaps,
#         name: str = "smap",
#         affine: np.ndarray | None = None,
#         save: bool = True,
#     ):
#         """Add sensitivity-map magnitude and phase."""
#         return self.add_complex(
#             smap.get_tensor(),
#             name=name,
#             affine=affine,
#             tdim=0,
#             save=save,
#         )

#     def add_complex(
#         self,
#         t: torch.Tensor,
#         name: str = "img",
#         affine: np.ndarray | None = None,
#         tdim: int = -1,
#         save: bool = True,
#     ):
#         """Add magnitude and phase as separate overlays."""
#         self.add(
#             torch.abs(t),
#             name + "_mag",
#             affine,
#             tdim=tdim,
#             save=save,
#         )

#         self.add(
#             torch.angle(t),
#             name + "_ang",
#             affine,
#             tdim=tdim,
#             save=save,
#         )

#         return self

#     def halt(self):
#         """Stop displaying new images; permanent saving remains enabled."""
#         self.halted = True
#         return self

#     def unhalt(self):
#         """Enable viewing; launch FSLeyes when the next image is added."""
#         self.halted = False
#         return self

#     def close(self):
#         """Close the managed FSLeyes process and clean up temporary files."""
#         if self.process is not None:
#             if self.process.poll() is None:
#                 self.process.terminate()
#                 self.process.wait()

#             self.process = None

#         self.tmpdir.cleanup()
#         return self

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.close()

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