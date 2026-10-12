import torch
from mirtorch.linear import LinearMap, NuSense, NuSenseGram

class PBNuSense(LinearMap):
    """
    A partial batching NuSense
    """

    def __init__(self, smaps, traj, batchsize=1):
        self.nf = traj.shape[0]

        if batchsize > 0:
            self.bs = batchsize
        elif batchsize == 0:
            raise ValueError("Cannot use batchsize = 0")
        else:
            self.bs = self.nf


        self.ops = []
        for start in range(0, self.nf, self.bs):
            end = min(start + self.bs, self.nf)
            self.ops.append(NuSense(smaps, traj[start:end, :, :]))

        size_in = self.ops[0].size_in.copy()
        size_in[0] = self.nf
        size_out = self.ops[0].size_out.copy()
        size_out[0] = self.nf

        super().__init__(size_in, size_out)

    def autobatchsize(self, smaps, traj, initial=None):
        nf = traj.shape[0]

        if initial is None:
            free, _ = torch.cuda.mem_get_info()
            # Conservative initial guess
            initial = min(nf, 8)

        bs = initial

        while bs >= 1:
            try:
                # torch.cuda.empty_cache()

                # op = NuSense(smaps, traj[:bs])

                # x = torch.zeros(
                #     op.size_in,
                #     dtype=smaps.dtype,
                #     device=smaps.device,
                # )

                # _ = op.apply(x)
                # torch.cuda.synchronize()

                # del op, x
                # torch.cuda.empty_cache()

                # return bs
                op = NuSense(smaps, traj[:bs])

                y = op.apply(torch.zeros(op.size_in, dtype=smaps.dtype, device=smaps.device))
                y = op.adjoint(torch.zeros(op.size_out, dtype=smaps.dtype, device=smaps.device))

                torch.cuda.synchronize()  # force any async OOM to surface here

                return bs
            except torch.cuda.OutOfMemoryError:
                print("HALVED")
                torch.cuda.empty_cache()
                bs //= 2

        raise RuntimeError("Batch size 1 does not fit.")

    def _apply(self, x):
        y = []
        for i, start in enumerate(range(0, self.nf, self.bs)):
            end = min(start + self.bs, self.nf)
            y.append(self.ops[i].apply(x[start:end, :, :, :, :]))

        return torch.cat(y, dim=0)

    def _apply_adjoint(self, x):
        y = []
        for i, start in enumerate(range(0, self.nf, self.bs)):
            end = min(start + self.bs, self.nf)
            y.append(self.ops[i].adjoint(x[start:end, :, :]))

        return torch.cat(y, dim=0)


class PBNuSenseGram(LinearMap):
    """
    A partial batching NuSenseGram
    """

    def __init__(self, smaps, traj, batchsize=1):
        self.nf = traj.shape[0]

        if batchsize > 0:
            self.bs = batchsize
        elif batchsize == 0:
            raise ValueError("Cannot use batchsize = 0")
        else:
            self.bs = self.nf


        self.ops = []
        for start in range(0, self.nf, self.bs):
            end = min(start + self.bs, self.nf)
            self.ops.append(NuSenseGram(smaps, traj[start:end, :, :]))

        size_in = self.ops[0].size_in.copy()
        size_in[0] = self.nf
        size_out = self.ops[0].size_out.copy()
        size_out[0] = self.nf

        super().__init__(size_in, size_out)

    def autobatchsize(self, smaps, traj, initial=None):
        nf = traj.shape[0]

        if initial is None:
            free, _ = torch.cuda.mem_get_info()
            # Conservative initial guess
            initial = min(nf, 8)

        bs = initial

        while bs >= 1:
            try:
                # torch.cuda.empty_cache()

                # op = NuSense(smaps, traj[:bs])

                # x = torch.zeros(
                #     op.size_in,
                #     dtype=smaps.dtype,
                #     device=smaps.device,
                # )

                # _ = op.apply(x)
                # torch.cuda.synchronize()

                # del op, x
                # torch.cuda.empty_cache()

                # return bs
                op = NuSense(smaps, traj[:bs])

                y = op.apply(torch.zeros(op.size_in, dtype=smaps.dtype, device=smaps.device))
                y = op.adjoint(torch.zeros(op.size_out, dtype=smaps.dtype, device=smaps.device))

                torch.cuda.synchronize()  # force any async OOM to surface here

                return bs
            except torch.cuda.OutOfMemoryError:
                print("HALVED")
                torch.cuda.empty_cache()
                bs //= 2

        raise RuntimeError("Batch size 1 does not fit.")

    def _apply(self, x):
        y = []
        for i, start in enumerate(range(0, self.nf, self.bs)):
            end = min(start + self.bs, self.nf)
            y.append(self.ops[i].apply(x[start:end, :, :]))
        return torch.cat(y, dim=0)

    def _apply_adjoint(self, x):
        return self._apply(x)