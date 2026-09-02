"""DeepPhD inference entry: load a checkpoint and denoise fluorescence volumes."""
import os

import numpy as np
import torch
from torch.nn.parallel import DataParallel

from data_loader.dataloader import testset
from data_loader.dataloader_utils import test_preprocess_chooseOne_real
from model.DeepPhD import DeepPhD
from DeepPhD_train import Settings, init_params
from utils.arg_parser import configure_gpus, find_checkpoint, resolve_logdir, test_parser
from utils.inference_io import run_inference


def load_model_checkpoint(model, checkpoint_path):
    """Load model weights from a training checkpoint.

    Args:
        model: DeepPhD model (possibly wrapped in DataParallel).
        checkpoint_path: Path to a ``.pth`` checkpoint.

    Returns:
        Tuple of (model with loaded weights, epoch number stored in the checkpoint).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model, checkpoint['epoch_num']


def main(args):
    """Build the model from ``args``, load the checkpoint, and run inference."""
    device_ids = configure_gpus(args.gpu)

    noise_tokens = set(args.noise_model.split('|'))
    use_rn = 'rn' in noise_tokens
    use_fpn = 'fpn' in noise_tokens

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(0)

    settings = Settings()
    settings.device_ids = device_ids
    settings.num_gpus = len(device_ids)
    if args.datasets_path is not None:
        settings.datasets_path = args.datasets_path

    print(f"Using GPU(s): {args.gpu} (visible as cuda:{device_ids})")

    experiment_label = os.path.basename(args.exp_dir.rstrip('/'))
    logdir = resolve_logdir(args.exp_dir)
    checkpoint_dir = os.path.join(logdir, 'saved_models')
    checkpoint_path, epoch = find_checkpoint(checkpoint_dir, epoch=args.epoch)
    print(f"Loading checkpoint: {checkpoint_path}")

    name_list, noise_im, coordinate_list, _, _, num_w, _ = test_preprocess_chooseOne_real(
        settings, img_id=0
    )
    original_shape = noise_im.shape
    test_data = testset(name_list, coordinate_list, np.zeros(original_shape, dtype=np.float32))
    x_shape = test_data[0][0].shape

    deepphd = DeepPhD(
        x_shape,
        noise_model=args.noise_model,
        param_inits=init_params(),
        RN_loop=settings.RN_loop,
        original_shape=original_shape,
    )
    deepphd = deepphd.cuda()
    deepphd = DataParallel(deepphd, device_ids=settings.device_ids)
    deepphd, epoch = load_model_checkpoint(deepphd, checkpoint_path)
    deepphd.eval()

    run_inference(
        deepphd,
        settings,
        logdir,
        experiment_label,
        epoch,
        use_rn=use_rn,
        use_fpn=use_fpn,
        save_noise=args.save_noise,
    )


if __name__ == "__main__":
    main(test_parser())
