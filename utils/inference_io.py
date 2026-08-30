"""Save denoised results and estimated noise maps from DeepPhD inference."""
import os
import time

import numpy as np
import torch
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.dataloader import FixedSizeGroupBatchSampler, testset
from data_loader.dataloader_utils import multibatch_test_save, singlebatch_test_save, test_preprocess_chooseOne_real


def cast_output_image(output_img, input_data_type):
    """Cast a float denoised volume to the original TIFF dtype with clipping."""
    if input_data_type == 'uint16':
        return np.clip(output_img, 0, 65535).astype('uint16')
    if input_data_type == 'int16':
        return np.clip(output_img, -32767, 32767).astype('int16')
    if input_data_type == 'uint8':
        return np.clip(output_img, 0, 255).astype('uint8')
    if input_data_type == 'int8':
        return np.clip(output_img, -128, 127).astype('int16')
    return output_img.astype('int32')


def save_denoised_result(output_img, logdir, experiment_label, test_im_name, epoch, input_data_type, original_T):
    """Write the denoised volume (trimmed to ``original_T``) under ``logdir/plots``."""
    output_img = cast_output_image(output_img, input_data_type)
    plot_pth = os.path.join(logdir, 'plots')
    os.makedirs(plot_pth, exist_ok=True)
    result_name = (
        f"{plot_pth}//{experiment_label}_{test_im_name.replace('.tif', '')}_"
        f"E_{str(epoch).zfill(2)}.tif"
    )
    io.imsave(result_name, output_img[:original_T, :, :], check_contrast=False)


def save_noise_outputs(logdir, experiment_label, epoch, *, test_im_name=None, rn_img=None, fpn_learned=None, use_rn=False, use_fpn=False):
    """Optionally save estimated RN and learned FPN maps as TIFF files."""
    if use_rn and rn_img is not None:
        rn_pth = os.path.join(logdir, 'RN')
        os.makedirs(rn_pth, exist_ok=True)
        rn_img = np.clip(rn_img.squeeze().astype(np.float32), -32767, 32767).astype('int16')
        im_stem = test_im_name.replace('.tif', '') if test_im_name else 'unknown'
        io.imsave(f'{rn_pth}/RN_{im_stem}_ep_{epoch}.tif', rn_img, check_contrast=False)

    if use_fpn and fpn_learned is not None:
        fpn_pth = os.path.join(logdir, 'FPN')
        os.makedirs(fpn_pth, exist_ok=True)
        fpn_learned = np.clip(fpn_learned, -32767, 32767).astype('int16')
        io.imsave(f'{fpn_pth}/FPN_ep_{epoch}.tif', fpn_learned, check_contrast=False)


def run_inference(
    deepphd,
    settings,
    logdir,
    experiment_label,
    epoch,
    *,
    use_rn=False,
    use_fpn=False,
    save_noise=False,
    log_first_image=True,
    train_stats=None,
):
    """Run patch-wise inference on all TIFFs in ``settings.datasets_path`` and save outputs.

    Stitches overlapping patches back into full volumes, optionally dumps RN/FPN,
    and prints a short summary for the first image (including train stats if provided).
    """
    num_test_img = [f for f in os.listdir(settings.datasets_path) if f.endswith('.tif')]
    for img_id in range(len(num_test_img)):
        name_list, noise_img, coordinate_list, test_im_name, input_data_type, num_w, original_T = (
            test_preprocess_chooseOne_real(settings, img_id=img_id)
        )
        test_data = testset(name_list, coordinate_list, noise_img)
        sampler = FixedSizeGroupBatchSampler(
            dataset_len=len(test_data),
            patches_per_row=num_w,
            group_size=settings.test_batch,
        )
        testloader = DataLoader(test_data, batch_sampler=sampler, num_workers=settings.num_workers)

        denoise_img = np.zeros(noise_img.shape)
        rn_img = np.zeros(noise_img.shape) if save_noise and use_rn else None

        test_start = time.time()

        with torch.no_grad():
            for y_patch, single_coordinate in tqdm(
                testloader, desc="Processing Patches", total=len(testloader), ncols=100
            ):
                y_patch = y_patch.float().cuda()
                patch_kwargs = {
                    'init_h': single_coordinate['init_h'].cuda(),
                    'end_h': single_coordinate['end_h'].cuda(),
                    'init_w': single_coordinate['init_w'].cuda(),
                    'end_w': single_coordinate['end_w'].cuda(),
                    'patch_start_w': single_coordinate['patch_start_w'].cuda(),
                    'patch_end_w': single_coordinate['patch_end_w'].cuda(),
                    'patch_start_h': single_coordinate['patch_start_h'].cuda(),
                    'patch_end_h': single_coordinate['patch_end_h'].cuda(),
                }

                x, rn_patch, _ = deepphd.module.inference(y_patch, **patch_kwargs)
                x = np.squeeze(x.cpu().detach().numpy())

                postprocess_turn = 1 if x.ndim == 3 else x.shape[0]

                if postprocess_turn > 1:
                    for patch_id in range(postprocess_turn):
                        output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = (
                            multibatch_test_save(single_coordinate, patch_id, x)
                        )
                        denoise_img[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = output_patch

                        if save_noise and use_rn:
                            rn_patch_id = np.squeeze(rn_patch.cpu().detach().numpy()[patch_id])
                            rn_img[
                                single_coordinate['init_s'].numpy()[patch_id]:single_coordinate['end_s'].numpy()[patch_id],
                                single_coordinate['init_h'].numpy()[patch_id]:single_coordinate['end_h'].numpy()[patch_id],
                                single_coordinate['init_w'].numpy()[patch_id]:single_coordinate['end_w'].numpy()[patch_id],
                            ] = rn_patch_id
                else:
                    output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = (
                        singlebatch_test_save(single_coordinate, x)
                    )
                    denoise_img[
                        stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w,
                    ] = output_patch

                    if save_noise and use_rn:
                        rn_img[
                            int(single_coordinate['init_s']):int(single_coordinate['end_s']),
                            int(single_coordinate['init_h']):int(single_coordinate['end_h']),
                            int(single_coordinate['init_w']):int(single_coordinate['end_w']),
                        ] = np.squeeze(rn_patch.cpu().detach().numpy())

        output_img = denoise_img.squeeze().astype(np.float32)
        alpha, beta = deepphd.module.get_alpha_beta()
        alpha = alpha.detach().cpu().numpy()
        beta = beta.detach().cpu().numpy()
        test_time = time.time() - test_start

        if save_noise:
            fpn_learned = deepphd.module.get_FPN().cpu().numpy() if use_fpn else None
            save_noise_outputs(
                logdir, experiment_label, epoch,
                test_im_name=test_im_name,
                rn_img=rn_img, fpn_learned=fpn_learned,
                use_rn=use_rn, use_fpn=use_fpn,
            )

        if log_first_image and img_id == 0:
            if train_stats is not None:
                print(
                    "{}, epoch: {}, tr_loss: {:.3f}, tr_nll: {:.3f}, tr_denoise_loss: {:.3f}, "
                    "alpha:{:.2f}, beta:{:.2f}, tr_time: {:d}, ts_time: {:d}, T_time: {:d}".format(
                        experiment_label,
                        epoch,
                        train_stats['loss'],
                        train_stats['nll'],
                        train_stats['denoise_loss'],
                        alpha,
                        beta,
                        int(train_stats['train_time']),
                        int(test_time),
                        int(train_stats['train_time'] + test_time),
                    )
                )
            else:
                print(
                    "{}, epoch: {}, alpha: {:.2f}, beta: {:.2f}, test_time: {:d}".format(
                        experiment_label, epoch, alpha, beta, int(test_time)
                    )
                )

        save_denoised_result(
            output_img, logdir, experiment_label, test_im_name, epoch,
            input_data_type, original_T,
        )
