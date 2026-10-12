import torch
from torch import Tensor
from scipy.io import loadmat
import time
import gc
from mirtorch.alg import CG

def clr_mem():
    gc.collect()
    torch.cuda.empty_cache()

def best_available_device(verbose=True) -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    if verbose: print(maj("Using ") + grn(f"{device}"))
    return device

def load_smap(path: str) -> Tensor:
    mf = loadmat(path)
    smap = torch.from_numpy(mf['smap_mirt']).movedim(-1, 0).to(torch.complex64)
    
    return smap

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
        
def timed(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start

def print_soln(x, t, res_scl=1.0, pref=""):
    print(pref, end="")
    if x.converged:
        print(grn("SUCC!") + "  :  ", end="")
    else:
        print(red("FAIL!") + "  :  ", end="")

    print(f"[ {x.iterations} its , {t:.2f} s , |res| = {(x.residual_norm.item() / res_scl):.2e} , VRAM aloc {(torch.cuda.memory_allocated()/(1024 ** 3)):.4f} GiB ]")

def recon_cg(y, w, A, AhA, R, reg, device, max_iter):
    AHy = A.H * y.to(device)
    AHyw = A.H * (y.to(device) * w.to(device))

    clr_mem()

    # s = torch.real(torch.sum(AHyw * AHy.conj(), dim=tuple(range(1, AHy.ndim))))
    # s = s / torch.real(torch.sum(torch.abs(A * AHyw) ** 2, dim=tuple(range(1, len(A.size_out)))))
    # x0 = AHyw * s[:, None, None, None, None]

    s = torch.real(torch.sum(AHyw * AHy.conj(), dim=tuple(range(1, AHy.ndim))))
    s = s / torch.real(torch.sum(AHyw.conj() * (AhA * AHyw), dim=tuple(range(1, AHyw.ndim))))
    x0 = AHyw * s[:, None, None, None, None]

    del AHyw
    clr_mem()

    x, recon_time = timed(
        lambda: CG(
            AhA + reg * R,
            alert=True,
            max_iter=max_iter,
            eval_func= lambda res: (torch.linalg.norm(res)).item()
        ).run(x0, AHy, return_info=True),
        device
    )

    return x, recon_time

def maj(s):
    return "\033[35m" + s + "\033[0m"

def cya(s):
    return "\033[36m" + s + "\033[0m"

def red(s):
    return "\033[31m" + s + "\033[0m"

def grn(s):
    return "\033[32m" + s + "\033[0m"

def blu(s):
    return "\033[34m" + s + "\033[0m"