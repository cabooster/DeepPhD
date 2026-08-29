"""Physical noise model as a normalizing-flow composition (FPN, RN, MPGN)."""
import torch
from torch import nn
import numpy as np

from model.noise_model.noise_components import MPGNNormalization, FixPattern, RN

class PhysicalModel(nn.Module):
    """Physical modeling module: normalizing flow over noise residuals (FPN, RN, MPGN)."""

    def __init__(self, arch, mpgn_scale, FPN):
        """
        Args:
            arch: Pipe-separated layer names controlling which noise modules are active.
            mpgn_scale: Learnable MPGNScale module.
            FPN: Learnable fixed-pattern noise module.
        """
        super(PhysicalModel, self).__init__()
        self.arch = arch
        self.mpgn_scale = mpgn_scale
        self.FPN = FPN
        self.normalizing_flow_arch()

    def normalizing_flow_arch(self):
        """Instantiate flow layers according to ``self.arch`` tokens."""
        arch_lyrs = [lyr.strip() for lyr in self.arch.lower().split('|') if lyr.strip()]
        self.use_FPN = False
        self.use_RN = False
        self.use_MPGN = False
        if 'fpn' in arch_lyrs:
            print('|fpn')
            self.use_FPN = True
            self.module_FPN = FixPattern(fp=self.FPN)
        if 'rn' in arch_lyrs:
            print('|rn')
            self.use_RN = True
            self.module_rn = RN()
        if 'mpgn' in arch_lyrs:
            print('|mpgn')
            self.use_MPGN = True
            self.module_mpgn = MPGNNormalization(scale=self.mpgn_scale)


    def forward(self, n, x, patch_info, **kwargs):
        """Map residual ``n`` through the flow; return latent ``z`` and log-det objective."""
        z = n
        objective = torch.zeros(n.shape[0], dtype=torch.float32, device=n.device)
        if self.use_FPN and self.use_RN:
            fpn = self.FPN(patch_info).detach()
            z_temp = (z - fpn).detach()
            _, rn_final = self.module_rn.cal_row_noise_batch(z_temp, x, self.mpgn_scale, patch_info["augmentation_transform"])
            z = z - rn_final
            z, log_abs_det_J_inv = self.module_FPN(z, patch_info, **kwargs)
            objective += log_abs_det_J_inv
        elif self.use_FPN:
            z, log_abs_det_J_inv = self.module_FPN(z, patch_info, **kwargs)
            objective += log_abs_det_J_inv
        elif self.use_RN:
            z, rn_final = self.module_rn.cal_row_noise_batch(z, x, self.mpgn_scale, patch_info["augmentation_transform"])
        if self.use_MPGN:
            z, log_abs_det_J_inv = self.module_mpgn(z, x, **kwargs)
            objective += log_abs_det_J_inv

        return z, objective

    def loss(self, n, x, patch_info, **kwargs):
        """Negative log-likelihood of residual ``n`` under the physical model (per dimension)."""
        z, objective = self.forward(n, x, patch_info, **kwargs)

        # base measure
        if self.use_MPGN:
            logp = self.prior(n)
            log_z = logp(z)
            objective += log_z
            nll = - objective
        else:
            nll = torch.zeros(1, device=n.device)
        nll_dim = torch.mean(nll) / np.prod(x.shape[1:])

        return nll_dim


    def prior(self, x):
        """Standard diagonal Gaussian prior log-density on the latent."""
        h = torch.zeros(list(x.shape[:]), device=x.device)
        pz = gaussian_diag(h, h)

        def logp(z1):
            return pz.logp(z1)

        return logp

def gaussian_diag(mean, logsd):
    """Diagonal Gaussian with given mean and log standard deviation."""
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.logps = lambda x: -0.5 * (np.log(2 * np.pi) + 2. * o.logsd + (x - o.mean) ** 2 / torch.exp(2. * o.logsd))
    o.logp = lambda x: torch.sum(o.logps(x), dim=[1, 2, 3])
    return o
