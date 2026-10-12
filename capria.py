from obj.RawData import RawData
from obj.FSLViewer import FSLViewer
from obj.Smap import Smaps
from obj.helpers import *
from copy import deepcopy
import torch

# Flags
disp = False     # Display Images?
save = True     # Save Images?
verbose = True  # Are we printing

# Check for the best CUDA device
device = best_available_device(verbose=True)
torch.manual_seed(0)
RawData.set_verbose(verbose)

# Reconstruction constnts
ccfac = 4
reg = 1
max_iter = 500
wsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#wsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#wsizes = [20]
#wsizes = [10, 11, 12, 13, 14, 15]
atol = 0.0
rtol = 10**-6

# Load in the Sensetivity Maps and setup the FSL viewer
smap = Smaps.load("/home/clouly/DEV/asldata/20260731/smaps/grey_64_smap.mat", cdim=-1, dtype=torch.complex64, device="cpu").compress(ccfac)
viewer = FSLViewer(halted=(not disp))
viewer.add_smap(smap)

# Start loading data
for y_raw in RawData.from_glob("/home/clouly/DEV/asldata/20260731/asldata_look*"):
    print(y_raw)
    viewer.set_save_dir(str(y_raw.dir))

    # Make directories
    if save:
        l_path = y_raw.dir / "label"
        c_path = y_raw.dir / "control"
        s_path = y_raw.dir / "subtraction"
        l_path.mkdir(exist_ok=True)
        c_path.mkdir(exist_ok=True)
        s_path.mkdir(exist_ok=True)

    # We:
    #   - Clip off the first 50 kspace samples from each echo
    #   - Coil Compress
    #   - Swap echos for frames
    #   - Separate out control and label 
    yl_raw, yc_raw = y_raw.initial_clip().coil_compress(smap).swapdims(1, 2).split_cl(dim=1)

    # The first time the batch recon fails, we stop trying.
    # This is because as window size increases, we use more memory, 
    # so if a previous one fails (for a given directory, all the following recons will have >= window size),
    # we dont try another batch recon. (This flag resets to None every time we load in a new Pfile)
    # Note 
    #   - None means we try batch and do framewise recon if it fails
    #   - True means we just do batch recon (script crashes if it fails)
    #   - False means we just do framewise recon (slower)
    batch_flag = None

    # Loop through different window sizes and reconstruct
    for ws in wsizes:
        print(maj(f"Processing data with Window Size = ") + grn(f"{ws}") + "...")
        # Window the data and reshape
        ylw = deepcopy(yl_raw).window(ws, 2, 0).set_mirt_shape()
        ycw = deepcopy(yc_raw).window(ws, 2, 0).set_mirt_shape()

        # Estimate Density Compensation Tensor
        w = ylw.get_dcf(device=device)

        # Label Recon
        print(cya("LABEL: "), end="")
        xl_img, bf = ylw.recon_CG(smap, reg, w, device, max_iter, batchmode=batch_flag)
        if bf == False: batch_flag = False  # If batchmode failed, we dont try again (for this data directory)
        del ylw
        clr_mem()

        # Save Label Img
        if save: viewer.set_save_dir(str(l_path))
        viewer.add_complex(xl_img, f"w{ws}_r{reg}_lbl", tdim=0, save=save)

        # Control Recon
        print(cya("CNTRL: "), end="")
        xc_img, _ = ycw.recon_CG(smap, reg, w, device, max_iter, batchmode=batch_flag)
        if bf == False: batch_flag = False  # If batchmode failed, we dont try again (for this data directory)
        del ycw, w
        clr_mem()

        # Save Control Img
        if save: viewer.set_save_dir(str(c_path))
        viewer.add_complex(xl_img, f"w{ws}_r{reg}_ctr", tdim=0, save=save)

        # Save Subtraction Img
        if save: viewer.set_save_dir(str(s_path))
        viewer.add_complex(xl_img - xc_img, f"w{ws}_r{reg}_sub", tdim=0, save=save)

        # Display current Memory Usage
        print(maj(f"(W{ws})") + f" Finished. Current Mem Usage: {(torch.cuda.memory_allocated()/(1024 ** 3)):.4f} GiB")


# Wait to finish the script
if disp: viewer.hang()
