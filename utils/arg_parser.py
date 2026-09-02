"""CLI argument parsers and helpers for DeepPhD train / inference."""
import argparse
import os


def configure_gpus(gpu):
    """Restrict visible GPUs via CUDA_VISIBLE_DEVICES before CUDA init."""
    gpu_ids = [int(part.strip()) for part in gpu.split(',') if part.strip()]
    if not gpu_ids:
        raise ValueError('At least one GPU id is required for --gpu')
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f'Invalid GPU id(s) in --gpu={gpu!r}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in gpu_ids)
    return list(range(len(gpu_ids)))


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_logdir(experiment_name):
    """Map experiment name to results/<experiment_name>/."""
    experiment_name = experiment_name.rstrip('/')
    if os.path.isabs(experiment_name):
        path = experiment_name
    else:
        path = os.path.join(_project_root(), 'results', experiment_name.lstrip('./'))
    if not path.endswith('/'):
        path += '/'
    return path


def find_checkpoint(checkpoint_dir, epoch=None):
    """Return checkpoint path and epoch number (latest if epoch is None)."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    epoch_to_file = {}
    for name in checkpoints:
        stem = name[:-4] if name.endswith('.pth') else name
        parts = stem.split('_')
        if len(parts) >= 2 and parts[0] == 'epoch' and parts[1].isdigit():
            epoch_to_file[int(parts[1])] = name

    if not epoch_to_file:
        raise FileNotFoundError(f"No valid epoch checkpoint in {checkpoint_dir}")

    if epoch is None:
        epoch = max(epoch_to_file)
    elif epoch not in epoch_to_file:
        raise FileNotFoundError(f"Epoch {epoch} not found in {checkpoint_dir}")

    return os.path.join(checkpoint_dir, epoch_to_file[epoch]), epoch


def train_parser():
    """Parse CLI arguments for DeepPhD training."""
    parser = argparse.ArgumentParser(description="Train DeepPhD on real fluorescence data")
    parser.add_argument("--exp_dir", type=str, default='./demo/',
                        help="Experiment name under results/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        '--gpu', type=str, default='0,1',
        help='GPU device id(s), comma-separated, e.g. 0 or 0,1 (default: 0,1)')
    parser.add_argument(
        "--noise_model", type=str, default='fpn|rn|mpgn',
        help="Choose an appropriate noise model that matches how your data were acquired, e.g. fpn|rn|mpgn for light-sheet and widefield microscopy. mpgn for multiphoton microscopy. fpn|mpgn for SMLM.")
    parser.add_argument('--datasets_path', type=str, default=None,
                        help="Path to dataset; defaults to Settings.datasets_path if not set")
    parser.add_argument(
        '--fresh_start', action='store_true',
        help="Delete existing experiment directory and train from scratch")
    parser.add_argument(
        '--save_noise', action='store_true',
        help="Save learned FPN and estimated RN maps during validation")
    return parser.parse_args()


def test_parser():
    """Parse CLI arguments for DeepPhD inference."""
    parser = argparse.ArgumentParser(description="Run DeepPhD inference from a trained checkpoint")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment name or absolute path to the training log directory")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch to load; default is the latest saved epoch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        '--gpu', type=str, default='0,1',
        help='GPU device id(s), comma-separated, e.g. 0 or 0,1 (default: 0,1)')
    parser.add_argument(
        "--noise_model", type=str, default='fpn|rn|mpgn',
        help="Choose an appropriate noise model that matches how your data were acquired, e.g. fpn|rn|mpgn for light-sheet and widefield microscopy. mpgn for multiphoton microscopy. fpn|mpgn for SMLM.")
    parser.add_argument('--datasets_path', type=str, default=None,
                        help="Path to dataset; defaults to Settings.datasets_path if not set")
    parser.add_argument(
        '--save_noise', action='store_true',
        help="Save learned FPN and estimated RN maps")
    return parser.parse_args()
