"""DeepPhD training entry: joint optimization of physics parameters and the denoise network."""
import os
import shutil
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from data_loader.dataloader_utils import prepare_dataset
from data_loader.dataloader import trainset, FixedSizeGroupBatchSampler
from tqdm import tqdm
from utils.arg_parser import configure_gpus, resolve_logdir, train_parser
from utils.inference_io import run_inference
from model.DeepPhD import DeepPhD


def save_checkpoint(model, optimizer, epoch_num, checkpoint_dir):
    """Persist model, optimizer state, and epoch index to ``checkpoint_dir``."""
    checkpoint = {'epoch_num' : epoch_num, 'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_dir)


def load_checkpoint(model, optimizer, checkpoint_dir):
    """Restore model and optimizer from a checkpoint; return them with the saved epoch."""
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch_num']


def init_params():
    """Default initial values for MPGN gain (log_alpha) and variance (beta_raw)."""
    init_log_alpha = 2.
    init_beta_raw = 2.
    init_dict = {'init_log_alpha':init_log_alpha, 'init_beta_raw':init_beta_raw}
    return init_dict


class Settings():
    """Training and data hyperparameters shared by train and inference scripts."""

    def __init__(self):
        self.overlap_factor = 0.6
        self.patch_x = self.patch_y = 128
        self.patch_t = 128
        self.gap_x = self.gap_y = 96

        self.select_img_num = 100000
        self.train_datasets_size = 6000
        self.datasets_path = 'datasets_real/20251218-zebrafish-crop/CZ223-1-10X-zoom05-laser0.4-exposure-5ms'

        self.num_workers = 4
        self.epochs = 51
        self.lr_network = 0.0001
        self.lr_physics = 0.05
        self.RN_loop = 2
        self.test_batch = 16
        self.train_batch = 4
        self.device_ids = [0]
        self.num_gpus = 1

        self.test_datasize = 100000000


def main(args):
    """Train DeepPhD and run a final validation pass on the last epoch."""
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

    device = torch.device('cuda:0')
    print(f"Using GPU(s): {args.gpu} (visible as cuda:{device_ids})")


    exp_dir = args.exp_dir.rstrip('/')
    args.experiment_label = os.path.basename(exp_dir)
    logdir = resolve_logdir(args.exp_dir)

    if args.fresh_start and os.path.exists(logdir):
        shutil.rmtree(logdir)

    os.makedirs(logdir, exist_ok=True)
    checkpoint_dir = os.path.join(logdir, 'saved_models')

    prepared_dataset = prepare_dataset(settings=settings)
    name_list, coordinate_list, noise_img_all, stack_index, original_shape, num_w = prepared_dataset.train_preprocess_lessMemoryMulStacks()
    print(f"batchsize new: {num_w}")
    train_dataset = trainset(name_list, coordinate_list, noise_img_all, stack_index)

    sampler = FixedSizeGroupBatchSampler(
        dataset_len=len(train_dataset),
        patches_per_row=num_w,
        group_size=settings.train_batch,                   
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=8)

    x_shape = next(iter(train_dataloader))['y_a'].shape

    param_inits = init_params()

    deepphd = DeepPhD(x_shape[1:], noise_model=args.noise_model, param_inits=param_inits,
    RN_loop = settings.RN_loop, original_shape=original_shape)

    optimizer = torch.optim.Adam([
    {'params': deepphd.network.parameters(), 'lr': settings.lr_network*1},
    {'params': deepphd.FPN.fpn_pattern, 'lr': settings.lr_physics*1},
    {'params': deepphd.mpgn_scale.log_alpha, 'lr': settings.lr_physics*0.1},
    {'params': deepphd.mpgn_scale.beta_raw, 'lr': settings.lr_physics*0.1},
    ], betas=(0.9, 0.999), eps=1e-08)

    print("physical model num params: {}".format(int(np.sum([np.prod(params.shape) for params in deepphd.physical_model.parameters()]))))
    print("network num params: {}".format(np.sum([np.prod(params.shape) for params in deepphd.network.parameters()])))

    deepphd = deepphd.to(device)
    deepphd = DataParallel[DeepPhD](deepphd, device_ids=settings.device_ids)
    
    start_epoch = 1
    if not os.path.exists(checkpoint_dir) or args.fresh_start:
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        models = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        last_epoch = min(max([int(i[:-4].split("_")[1]) for i in models[:]]), 50)
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{}.pth'.format(last_epoch))
        deepphd, optimizer, start_epoch = load_checkpoint(deepphd, optimizer, checkpoint_path)
        start_epoch += 1
        print('found an existing previous checkpoint, resuming from epoch {}'.format(last_epoch))

    start_time = time.time()

    for epoch in range(start_epoch, settings.epochs):
        train_loss_mean = train_time = test_time = 0
        train_nll_mean = train_denoise_loss_mean = 0
        train_loss, train_nll, train_denoise_loss = [], [], []
        train_curr_time = time.time()
        deepphd.train()

        for image in tqdm(train_dataloader, desc="Processing Patches", total=len(train_dataloader), ncols=100):
            optimizer.zero_grad()
            kwargs = {
                'epoch': epoch,
                'y_a': image['y_a'].float().cuda(),
                'y_b': image['y_b'].float().cuda(),
                'init_h': image['init_h'].cuda(),
                'end_h': image['end_h'].cuda(),
                'init_w': image['init_w'].cuda(),
                'end_w' : image['end_w'].cuda(),
                'patch_start_w' : image['patch_start_w'].cuda(),
                'patch_end_w' : image['patch_end_w'].cuda(),
                'patch_start_h' : image['patch_start_h'].cuda(),
                'patch_end_h' : image['patch_end_h'].cuda(), 
                'augmentation_transform' : image["augmentation_transform"].cuda()
            }

            hybrid_loss, nll, denoise_loss = deepphd(**kwargs)
            if hybrid_loss.requires_grad and hybrid_loss.grad_fn is not None:
                torch.mean(hybrid_loss).backward()
            else:
                print(f"Epoch {epoch}: loss has no gradient, skip backward. (value={hybrid_loss.item():.6f})")
                continue
            train_loss.append(hybrid_loss.mean().item())
            train_nll.append(nll.mean().item())
            train_denoise_loss.append(denoise_loss.mean().item())
            optimizer.step()

            train_loss_mean = np.mean(train_loss)
            train_nll_mean = np.mean(train_nll)
            train_denoise_loss_mean = np.mean(train_denoise_loss)
            train_time = time.time() - train_curr_time
        
        save_checkpoint(
            deepphd,
            optimizer,
            epoch,
            os.path.join(checkpoint_dir, 'epoch_{}.pth'.format(epoch))
        )
        
        if epoch == settings.epochs - 1:
            deepphd.eval()
            run_inference(
                deepphd,
                settings,
                logdir,
                args.experiment_label,
                epoch,
                use_rn=use_rn,
                use_fpn=use_fpn,
                save_noise=args.save_noise,
                train_stats={
                    'loss': train_loss_mean,
                    'nll': train_nll_mean,
                    'denoise_loss': train_denoise_loss_mean,
                    'train_time': train_time,
                },
            )

    total_time = time.time() - start_time
    print('Total time = %f' % total_time)


if __name__ == "__main__":
    main(train_parser())
