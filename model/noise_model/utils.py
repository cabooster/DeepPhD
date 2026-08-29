"""Learnable MPGN scale and fixed-pattern noise (FPN) modules."""
import torch
from torch import nn

class MPGNScale(nn.Module):
    """Signal-dependent MPGN scale from system gain α and Gaussian variance β."""

    def __init__(self, param_inits):
        """
        Args:
            param_inits: Dict with ``init_log_alpha`` and ``init_beta_raw``.
        """
        super(MPGNScale, self).__init__()

        self.log_alpha = nn.Parameter(torch.tensor(param_inits['init_log_alpha']), requires_grad=True)
        self.beta_raw = nn.Parameter(torch.tensor(param_inits['init_beta_raw']), requires_grad=True)

    def forward(self, x):
        """Return ``sqrt(alpha * x + beta)``, clamped for numerical stability."""
        alpha = torch.exp(self.log_alpha)
        beta = (torch.exp(self.beta_raw) - torch.exp(-self.beta_raw)) / 2

        scale = alpha * x + beta

        scale = torch.clamp(scale, min=0.001)
        assert torch.min(scale) >= 0
        return torch.sqrt(scale)
    

class FPN(nn.Module):
    """Learnable fixed-pattern noise map cropped / transformed per patch."""

    def __init__(self, original_shape):
        """
        Args:
            original_shape: Full volume shape ``(T, H, W)``; FPN stores an ``(H, W)`` pattern.
        """
        super(FPN, self).__init__()
        self.fpn_pattern = nn.Parameter(torch.zeros(original_shape[1:3], dtype=torch.float32), requires_grad=True)


    def forward(self, patch_info, **kwargs):
        """Crop the global FPN to each patch, apply augmentation, return stacked patches."""
        patches = []

        for i in range(len(patch_info['init_h'])):

            if 'augmentation_transform' in patch_info:
                augmentation_transform = patch_info['augmentation_transform'][i].item()
            else:
                augmentation_transform = 0 

            patch = self.fpn_pattern[patch_info['init_h'][i]:patch_info['end_h'][i], patch_info['init_w'][i]:patch_info['end_w'][i]]
            mask = torch.zeros(patch_info['end_h'][i]-patch_info['init_h'][i], patch_info['end_w'][i]-patch_info['init_w'][i], dtype=torch.bool)
            
            mask[int(patch_info['patch_start_h'][i].item()) : int(patch_info['patch_end_h'][i].item()), int(patch_info['patch_start_w'][i].item()) : int(patch_info['patch_end_w'][i].item())] = True
            patch_detached = patch.detach()
            final_patch = patch_detached.clone()
            final_patch[mask] = patch[mask]

            final_patch = final_patch - torch.mean(final_patch)

            final_patch = FPN_transform(final_patch, augmentation_transform)
            
            patches.append(final_patch)
        
        fpn = torch.stack(patches, dim=0)
        fpn = fpn.unsqueeze(1)
        
        return fpn

    
    def get_FPN_whole(self):
        """Return the detached full-frame FPN pattern."""
        return self.fpn_pattern.detach()

    def FPN_distloss(self):
        """L1 magnitude regularizer on the FPN pattern."""
        noise = self.fpn_pattern
        flatten_values = noise[:, :].reshape(-1)
        distloss = torch.mean(torch.abs(flatten_values))
        return distloss

    
    def AC_loss(self):
        """Autocorrelation penalty after removing the row-mean component of FPN."""
        h, w = self.fpn_pattern.shape

        fpn_pattern_onlyrow = self.fpn_pattern.mean(dim=1, keepdim=True).expand(h, w)
        fpn_pattern_norow = self.fpn_pattern - fpn_pattern_onlyrow

        f = torch.fft.fft2(fpn_pattern_norow)
        AC = torch.fft.ifft2(torch.abs(f) ** 2)
        AC = torch.fft.fftshift(AC)
        AC_abs = torch.abs(AC)

        center_h, center_w = h // 2, w // 2
        AC_abs[center_h, center_w] = 0.0
        return AC_abs.mean()


def FPN_transform(fpn, augmentation_transform):
    """Apply the same geometric augmentation used on image patches to an FPN crop."""
    if augmentation_transform == 0:  # no transformation
        fpn = fpn
    elif augmentation_transform == 1:  # left rotate 90
        fpn = torch.rot90(fpn, k=1, dims=(-2, -1))
    elif augmentation_transform == 2:  # left rotate 180
        fpn = torch.rot90(fpn, k=2, dims=(-2, -1))
    elif augmentation_transform == 3:  # left rotate 270
        fpn = torch.rot90(fpn, k=3, dims=(-2, -1))
    elif augmentation_transform == 4:  # horizontal flip
        fpn = torch.flip(fpn, dims=(-1,))
    elif augmentation_transform == 5:  # horizontal flip & left rotate 90
        fpn = torch.flip(fpn, dims=(-1,))
        fpn = torch.rot90(fpn, k=1, dims=(-2, -1))
    elif augmentation_transform == 6:  # horizontal flip & left rotate 180
        fpn = torch.flip(fpn, dims=(-1,))
        fpn = torch.rot90(fpn, k=2, dims=(-2, -1))
    elif augmentation_transform == 7:  # horizontal flip & left rotate 270
        fpn = torch.flip(fpn, dims=(-1,))
        fpn = torch.rot90(fpn, k=3, dims=(-2, -1))
    return fpn
