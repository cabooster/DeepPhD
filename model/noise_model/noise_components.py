"""Noise-component layers used inside the physical normalizing flow."""
import torch
from torch import nn


class FixPattern(nn.Module):
    """FPN translation layer in the normalizing flow."""

    def __init__(self, fp):
        super(FixPattern, self).__init__()
        self._fp = fp

    def forward(self, x, patch_info, **kwargs):
        """Subtract the FPN crop for ``patch_info`` from residual ``x``."""
        fpn = self._fp(patch_info)
        z = x - fpn
        log_abs_det_J_inv = 0
        return z, log_abs_det_J_inv


class MPGNNormalization(nn.Module):
    """MPGN normalization in the normalizing flow (signal-dependent scale)."""

    def __init__(self, scale):
        super(MPGNNormalization, self).__init__()
        self.scale = scale

    def forward(self, n, x, **kwargs):
        """Normalize residual ``n`` by ``scale(x)`` and return log-det of the inverse Jacobian."""
        scale = self.scale(x)
        z = n / scale
        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])
        return z, log_abs_det_J_inv


class RN(nn.Module):
    """Row noise (RN) estimation without learnable parameters."""

    def __init__(self):
        super(RN, self).__init__()

    def cal_row_noise_batch(self, n, x, mpgn_scale, augmentation_transform):
        """Estimate row noise for a training batch of patches along one image row.

        Assumes the batch contains all patches of one row (length B), each possibly
        geometrically augmented. Patches are inverse-transformed, concatenated along W,
        RN is computed by MPGN-weighted column aggregation, then re-augmented.

        Args:
            n: Residual patches ``[B, T, H, W]``.
            x: Signal estimate for MPGN scale, same shape as ``n``.
            mpgn_scale: MPGNScale module.
            augmentation_transform: Per-sample augmentation ids ``[B]``.

        Returns:
            Tuple of (residuals without RN, RN patches), both ``[B, T, H, W]``.
        """
        B, T, H, W = n.shape

        # Undo augmentation so patches share the original sensor orientation.
        n_orig = torch.stack([inverse_transform(n[i], augmentation_transform[i].item()) for i in range(B)], dim=0)
        x_orig = torch.stack([inverse_transform(x[i], augmentation_transform[i].item()) for i in range(B)], dim=0)

        # Patch-level MPGN scale (detached).
        scale = mpgn_scale(x_orig).detach()
        
        # Concatenate patches along width into one full row.
        full_row_noise = n_orig.permute(1, 2, 0, 3).reshape(T, H, B * W)
        full_row_scale = scale.permute(1, 2, 0, 3).reshape(T, H, B * W)

        # Row-noise estimate via weighted column aggregation.
        num = full_row_noise / full_row_scale
        den = 1.0 / full_row_scale
        rn = torch.sum(num, dim=-1, keepdim=True) / torch.sum(den, dim=-1, keepdim=True)  # [T, H, 1]
        rn_mean = torch.mean(rn, dim=1, keepdim=True)
        rn = rn - rn_mean

        # Broadcast RN across the full row width.
        rn = rn.expand(-1, -1, full_row_noise.shape[-1])

        # Split back into per-patch widths.
        W = n.shape[-1]
        rn_patches = rn.split(W, dim=-1)

        # Subtract RN and re-apply the original augmentations.
        n_without_rn = []
        rn_final = []
        for i in range(B):
            patch_without_rn = n_orig[i] - rn_patches[i]
            n_without_rn.append(apply_transform(patch_without_rn, augmentation_transform[i].item()))
            rn_final.append(apply_transform(rn_patches[i], augmentation_transform[i].item()))

        n_without_rn = torch.stack(n_without_rn, dim=0)
        rn_final = torch.stack(rn_final, dim=0)

        return n_without_rn, rn_final
    
    def cal_row_noise_test(self, n, x, mpgn_scale):
        """Estimate row noise at test time (no augmentation; batch is one image row)."""
        B, D, H, W = n.shape
        n = n.permute(1, 2, 0, 3).reshape(D, H, B * W)
        x = x.permute(1, 2, 0, 3).reshape(D, H, B * W)
        scale = mpgn_scale(x).detach()
        rows = n.shape[-1]
        num = n / scale
        den = 1 / scale

        rn = torch.sum(num, dim=-1, keepdim=True) / torch.sum(den, dim=-1, keepdim=True)
        rn_mean = torch.mean(rn, dim=1, keepdim=True)
        rn = rn - rn_mean
        rn = rn.repeat(1, 1, rows)

        n_without_rn = n - rn
        n_without_rn = n_without_rn.reshape(D, H, B, W).permute(2, 0, 1, 3)
        rn = rn.reshape(D, H, B, W).permute(2, 0, 1, 3)
        return n_without_rn, rn 


def inverse_transform(x: torch.Tensor, augmentation_transform: int) -> torch.Tensor:
    """Undo geometric augmentation and restore the patch to the original orientation.

    Args:
        x: Tensor of shape ``[T, H, W]``.
        augmentation_transform: Augmentation type in ``0..7``.

    Returns:
        Tensor of the same shape with orientation restored.
    """
    if augmentation_transform == 0:
        return x
    elif augmentation_transform == 1:
        return torch.rot90(x, k=3, dims=[1, 2])  
    elif augmentation_transform == 2:
        return torch.rot90(x, k=2, dims=[1, 2])
    elif augmentation_transform == 3:
        return torch.rot90(x, k=1, dims=[1, 2])
    elif augmentation_transform == 4:
        return torch.flip(x, dims=[2])  
    elif augmentation_transform == 5:
        return torch.flip(torch.rot90(x, k=3, dims=[1, 2]), dims=[2])
    elif augmentation_transform == 6:
        return torch.flip(torch.rot90(x, k=2, dims=[1, 2]), dims=[2])
    elif augmentation_transform == 7:
        return torch.flip(torch.rot90(x, k=1, dims=[1, 2]), dims=[2])
    else:
        raise ValueError(f"Unsupported augmentation_transform: {augmentation_transform}")


def apply_transform(x, augmentation_transform):
    """Apply geometric augmentation ``augmentation_transform`` (ids 0–7) to ``x``."""
    if augmentation_transform == 0:
        return x
    elif augmentation_transform == 1:
        return torch.rot90(x, k=1, dims=[-2, -1])
    elif augmentation_transform == 2:
        return torch.rot90(x, k=2, dims=[-2, -1])
    elif augmentation_transform == 3:
        return torch.rot90(x, k=3, dims=[-2, -1])
    elif augmentation_transform == 4:
        return torch.flip(x, dims=[-1])
    elif augmentation_transform == 5:
        return torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1])
    elif augmentation_transform == 6:
        return torch.rot90(torch.flip(x, dims=[-1]), k=2, dims=[-2, -1])
    elif augmentation_transform == 7:
        return torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=[-2, -1])
    else:
        raise ValueError(f"Unsupported augmentation_transform: {augmentation_transform}")
