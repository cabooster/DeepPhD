"""DeepPhD model: physics-informed denoising with FPN / RN / MPGN and a 3D U-Net."""
import torch
from torch import nn

from model.noise_model.physical_model import PhysicalModel
from model.network.network import Network_3D_Unet
from model.noise_model.noise_components import RN
from model.noise_model.utils import MPGNScale, FPN

class DeepPhD(nn.Module):
    """Deep physics-informed denoising: physical modeling + image restoration.

    Combines a 3D U-Net denoiser with optional fixed-pattern noise (FPN),
    row noise (RN), and mixed Poisson-Gaussian noise (MPGN) modeling.
    """

    def __init__(self, x_shape, noise_model, param_inits, RN_loop, original_shape=None):
        """
        Args:
            x_shape: Spatial-temporal patch shape used by the network.
            noise_model: Flow layers to enable, e.g. ``fpn|rn|mpgn``.
            param_inits: Dict with ``init_log_alpha`` and ``init_beta_raw``.
            RN_loop: Number of row-noise estimation / denoise iterations.
            original_shape: Full volume shape for the learnable FPN pattern.
        """
        super(DeepPhD, self).__init__()
        self.network = Network_3D_Unet(in_channels=1, out_channels=1, f_maps=16, final_sigmoid=True)

        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()

        self.FPN = FPN(original_shape)
        self.mpgn_scale = MPGNScale(param_inits)
        self.RN = RN()
        self.physical_model = PhysicalModel(noise_model, self.mpgn_scale, self.FPN)

        self.loop = RN_loop
        flow_layers = set(t.strip() for t in noise_model.lower().split('|') if t.strip())
        self.use_RN = 'rn' in flow_layers
        self.use_FPN = 'fpn' in flow_layers

    def denoise_loss(self, pred, target):
        """L1 + L2 reconstruction loss between prediction and target."""
        return self.L1_loss(pred, target) + self.L2_loss(pred, target)

    def denoise(self, y, clip=False):
        """Run the 3D U-Net on a noisy patch ``y`` (adds/removes channel dim)."""
        y = y.unsqueeze(1)
        x = self.network(y)
        x = x.squeeze(1)
        if clip:
            x = torch.clamp(x, min=0.)
        return x

    def forward(self, **kwargs):
        """Training forward pass; delegates to ``train_combined_model``."""
        return self.train_combined_model(**kwargs)
    
    def train_combined_model(self, epoch, y_a, y_b, init_h, end_h, init_w, end_w, patch_start_w, patch_end_w, patch_start_h, patch_end_h, augmentation_transform, **kwargs):
        """Bidirectional self-supervised training with physical NLL.

        Removes FPN (and optionally RN) from interlaced branches ``y_a`` / ``y_b``,
        denoises with the network, then fits the physical noise model on residuals.

        Returns:
            Tuple of (hybrid_loss, nll, denoise_loss), each shaped ``[1]``.
        """
        patch_info = {
            'init_h':init_h,
            'end_h':end_h, 
            'init_w':init_w, 
            'end_w':end_w, 
            'patch_start_w':patch_start_w, 
            'patch_end_w':patch_end_w, 
            'patch_start_h':patch_start_h, 
            'patch_end_h':patch_end_h, 
            "augmentation_transform" : augmentation_transform,
        }

        if self.use_FPN:
            fpn = self.FPN(patch_info).detach()
            y_a_fpn = (y_a - fpn).float()
            y_b_fpn = (y_b - fpn).float()
        else:
            y_a_fpn = y_a.float()
            y_b_fpn = y_b.float()

        if not self.use_RN:
            mpgn_a = y_a_fpn
            mpgn_b = y_b_fpn

            x_a = self.denoise(mpgn_a, clip=True)
            x_b = self.denoise(mpgn_b, clip=True)
            
        else:
            for param in self.network.parameters():
                param.requires_grad = False
            x_a = self.denoise(y_a_fpn, clip=True)
            x_b = self.denoise(y_b_fpn, clip=True)
            for param in self.network.parameters():
                param.requires_grad = True


            for i in range(self.loop):
                res_a = y_a_fpn - x_b
                res_b = y_b_fpn - x_a

                _, rn_a = self.RN.cal_row_noise_batch(res_a, x_b, self.mpgn_scale, augmentation_transform)
                _, rn_b = self.RN.cal_row_noise_batch(res_b, x_a, self.mpgn_scale, augmentation_transform)

                mpgn_a = y_a_fpn - rn_a
                mpgn_b = y_b_fpn - rn_b

                if i != self.loop - 1:
                    for param in self.network.parameters():
                        param.requires_grad = False
                else: 
                    for param in self.network.parameters():
                        param.requires_grad = True

                x_a = self.denoise(mpgn_a, clip=True)
                x_b = self.denoise(mpgn_b, clip=True)
        
        loss_a = self.denoise_loss(x_a, mpgn_b)
        loss_b = self.denoise_loss(x_b, mpgn_a)

        x_a_d = x_a.detach()
        x_b_d = x_b.detach()

        n_a = y_a - x_b_d
        n_b = y_b - x_a_d

        nll_a = self.physical_model.loss(n_a, x_b_d, patch_info, **kwargs)
        nll_b = self.physical_model.loss(n_b, x_a_d, patch_info, **kwargs)

        nll = (nll_a + nll_b)/2
        FPN_distloss = self.FPN.FPN_distloss()
        AC_loss = self.FPN.AC_loss()

        denoise_loss = (loss_a + loss_b)/2

        lambda_phys = 1000
        k4 = 0.01
        fpn_learned = self.get_FPN()
        if epoch > 6 and torch.mean(torch.abs(fpn_learned)) > 10:
            k2 = 0.00000001
        else:
            k2 = 0.00001
            k4 = 0.1

        hybrid_loss = lambda_phys * nll + denoise_loss 
        
        if self.use_FPN:
            hybrid_loss += (k2 * AC_loss + k4 * FPN_distloss)

        return hybrid_loss.unsqueeze(0), nll.unsqueeze(0), denoise_loss.unsqueeze(0)
      
    
    def inference(self, y, init_h, end_h, init_w, end_w, patch_start_w, patch_end_w, patch_start_h, patch_end_h, **kwargs):
        """Denoise a test patch; optionally iterate RN estimation.

        Returns:
            Tuple of (denoised ``x``, estimated row noise ``rn``, MPGN-only input ``y_mpgn``).
        """
        rn = torch.zeros(y.shape, device=y.device)
        y_mpgn = torch.zeros(y.shape, device=y.device)

        patch_info = {
            'init_h':init_h,
            'end_h':end_h, 
            'init_w':init_w, 
            'end_w':end_w, 
            'patch_start_w':patch_start_w, 
            'patch_end_w':patch_end_w, 
            'patch_start_h':patch_start_h, 
            'patch_end_h':patch_end_h
        }
        fpn = self.FPN(patch_info).detach().float()
        y_fpn = y - fpn

        x = self.denoise(y_fpn, clip=True)

        if self.use_RN:
            for _ in range(self.loop):
                res = y_fpn - x
                _, rn = self.RN.cal_row_noise_test(res, x, self.mpgn_scale)
                y_mpgn = y_fpn - rn
                x = self.denoise(y_mpgn, clip=True)
                    
        return x, rn, y_mpgn

    def get_FPN(self):
        """Return the full learned fixed-pattern noise map."""
        return self.FPN.get_FPN_whole()

    def get_alpha_beta(self):
        """Return MPGN gain ``alpha`` and softplus-style ``beta``."""
        return torch.exp(self.mpgn_scale.log_alpha), (torch.exp(self.mpgn_scale.beta_raw) - torch.exp(-self.mpgn_scale.beta_raw)) / 2
