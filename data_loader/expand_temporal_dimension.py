#!/usr/bin/env python3
"""
Script to expand temporal dimension of TIF image sequences.
Reads TIF files from 1channel directory (shape: T*H*W) and expands T dimension
by concatenating forward and backward sequences until reaching 400+ frames.
"""

import os
import glob
import tifffile
import numpy as np


def expand_temporal_dimension(img, target_frames=800):
    """
    Expand the temporal axis by forward-backward concatenation until ``target_frames``.

    For example, a 15-frame video [1,2,...,15] becomes
    [1,2,...,15, 14,...,2, 1,2,...,15, ...] (turning frames are not duplicated).

    Args:
        img: Volume of shape ``(T, H, W)``.
        target_frames: Desired number of frames after expansion.

    Returns:
        Expanded array of shape ``(target_frames, H, W)`` (or shorter if trimming is skipped).
    """

            
    T, H, W = img.shape
    print(f"  original frames: {T}")

    # Create forward-backward sequence
    expanded_frames = []
    current_frames = 0
    forward = True

    while current_frames < target_frames:
        if forward:
            # Add forward sequence
            expanded_frames.append(img)
            current_frames += T
            forward = False
        else:
            # Add backward sequence (reverse along time axis, excluding first and last)
            # This avoids duplication of frames at the turning points
            if T > 2:
                # For T>2, reverse and exclude first and last frame
                backward_seq = img[::-1][1:-1]
                expanded_frames.append(backward_seq)
                current_frames += (T - 2)
            elif T == 2:
                # For T=2, just alternate between the two frames
                expanded_frames.append(img[::-1])
                current_frames += T
            else:
                # For T=1, just repeat
                expanded_frames.append(img)
                current_frames += T
            forward = True

    # Concatenate all sequences along time axis
    expanded_img = np.concatenate(expanded_frames, axis=0)

    # Trim to exactly target_frames if exceeded
    if expanded_img.shape[0] > target_frames:
        expanded_img = expanded_img[:target_frames]

    final_frames = expanded_img.shape[0]
    print(f"  expanded frames: {final_frames}")
    print(f"  expanded shape: {expanded_img.shape}")
    return expanded_img
            


