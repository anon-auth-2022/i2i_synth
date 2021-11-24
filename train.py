import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

from StyleGAN2_ada.training.augment import AugmentPipe
from StyleGAN2_ada.torch_utils import misc
from fid import fid
from torch_tools.visualization import to_image_grid
from torch_tools.data import UnannotatedDataset


def main():
    opt = TrainOptions().parse()   # get training options
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    augment_pipe = AugmentPipe(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1,
                               xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    # prepare ADA pipeline
    augment_pipe.train().requires_grad_(False).cuda()
    augment_pipe.p.copy_(torch.as_tensor(opt.ada_p))
    mean_real_signs = []

    print(f'Number of training images: {len(dataset)}')

    test_dataset_A, test_dataset_B, train_dataset_A, train_dataset_B = init_data(opt.dataroot)

    model = create_and_load_model(opt, next(iter(dataset)))
    model.save_dir = opt.out_dir
    imgs_dir = f'{opt.out_dir}/images'
    os.makedirs(imgs_dir, exist_ok=True)
    # the total number of samples showed
    total_samples = 0
    epoch_samples = 0

    checkpoint_path, models_dir = f'{opt.out_dir}/checkpoint.pt', opt.out_dir

    if opt.load_state_from_checkpoint:
        model, aug_p, opt.epoch_count, total_samples, epoch_samples = \
            recover_from_checkpoint(model, checkpoint_path)

        if opt.drop_checkpoint_epochs:
            opt.epoch_count, total_samples, epoch_samples = 0, 0, 0
        elif not opt.ada_fixed and aug_p is not None:
            augment_pipe.p.copy_(torch.tensor(aug_p))

    # training loop
    print(f'opt.epoch_count: {opt.epoch_count}')
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        dataset.set_epoch(epoch)
        for _, data in enumerate(dataset.__iter__(start_from=epoch_samples)):  # inner epoch loop
            batch_size = data['A'].size(0)
            total_samples += batch_size
            epoch_samples += batch_size

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            ada_kwargs = {}
            if epoch < opt.ada_stop_epoch:
                ada_kwargs = dict(aug_B=augment_pipe, aug_A=augment_pipe) if opt.aug_before_gen \
                        else dict(aug_B=augment_pipe, aug_fake=augment_pipe)
            model.optimize_parameters(**ada_kwargs)

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            # update ADA
            if epoch < opt.ada_stop_epoch:
                if opt.ada_fixed:
                    if opt.ada_linear_annealing_to is not None:
                        scale = 1.0 - \
                            (epoch / opt.ada_stop_epoch) * (1.0 - opt.ada_linear_annealing_to)
                        augment_pipe.p.copy_(torch.as_tensor(opt.ada_p * scale))

                else:
                    mean_real_signs.append(
                        torch.sign(model.pred_real - opt.ada_real_sign_thr).mean().item())

                    if (total_samples // batch_size) % opt.ada_interval == 0:
                        adjust_ada_params_adaptive(opt, augment_pipe, mean_real_signs, batch_size)
                        mean_real_signs = []

            # display images
            if total_samples % opt.display_freq == 0:
                visualize(model, test_dataset_A, imgs_dir, epoch, total_samples)

            # print training losses and save logging information to the disk
            if total_samples % opt.print_freq == 0:
                log_losses(model.get_current_losses(), epoch, total_samples)

            # cache our latest model every <save_latest_freq> iterations
            if total_samples % opt.save_latest_freq == 0:
                save_suffix = f'iter_{total_samples}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                save_checkpoint(model, augment_pipe.p.data.item(),
                                epoch, total_samples, epoch_samples, checkpoint_path)

        # save model and calculate FID
        if epoch % opt.save_epoch_freq == 0:
            save_and_calculate_metrics(
                model, epoch, total_samples,
                test_dataset_A, test_dataset_B, train_dataset_A, train_dataset_B, models_dir)

            save_checkpoint(model, augment_pipe.p.data.item(),
                             epoch + 1, total_samples, 0, checkpoint_path)
            if opt.backup_all_checkpoints:
                save_checkpoint(model, augment_pipe.p.data.item(),
                                epoch + 1, total_samples, 0,
                                checkpoint_path.replace('.pt', f'_epoch_{epoch + 1}.pt'))

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}')

        # update learning rates at the end of every epoch.
        model.update_learning_rate()
        epoch_samples = 0


def init_data(dataroot):
    test_dataset_A = UnannotatedDataset(f'{dataroot}/testA')
    test_dataset_B = UnannotatedDataset(f'{dataroot}/testB')
    train_dataset_A = UnannotatedDataset(f'{dataroot}/trainA')
    train_dataset_B = UnannotatedDataset(f'{dataroot}/trainB')
    return test_dataset_A, test_dataset_B, train_dataset_A, train_dataset_B


def recover_from_checkpoint(model, checkpoint_path):
    print(f'Recovering state from: {checkpoint_path}')
    if not os.path.isfile(checkpoint_path):
        print('failed to load checkpoint, skipping')
        return model, 0.0, 0, 0, 0

    state = torch.load(checkpoint_path)
    model.load_training_state_dict(state)
    epoch_samples = state['epoch_samples'] if 'epoch_samples' in state else 0
    aug_p = state['aug_p'] if 'aug_p' in state else None

    return model, aug_p, state['epoch'], state['total_samples'], epoch_samples


def save_checkpoint(model, aug_p, epoch, total_samples, epoch_samples,
                    checkpoint_path):
    state = model.training_state_dict()
    state.update(dict(epoch=epoch,
                      total_samples=total_samples,
                      epoch_samples=epoch_samples,
                      aug_p=aug_p))
    torch.save(state, checkpoint_path)


def create_and_load_model(opt, data_sample):
    # create a model given opt.model and other options
    model = create_model(opt)
    model.data_dependent_initialize(data_sample)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    model.parallelize()
    return model


def log_losses(losses_dict, epoch, n_samples):
    log_string = ' | '.join([f'{n}: {v: 0.3f}' for n, v in losses_dict.items()])
    print(f'epoch {epoch} - n_samples {n_samples}: {log_string}')


def visualize(model, test_dataset_A, imgs_dir, epoch, total_samples):
    visual_sample = plot_test_samples(model.get_generator(), test_dataset_A)
    visual_sample.save(f'{imgs_dir}/e{epoch}_{total_samples}.jpg')


@torch.no_grad()
def plot_test_samples(gen, test_data, count=8):
    gen.eval()

    sample = torch.stack([test_data[i] for i in range(count)]).cuda()
    sample_translated = gen(sample)
    visualization = to_image_grid(torch.cat([sample, sample_translated]), nrow=count)
    gen.train()

    return visualization


@torch.no_grad()
def calculate_fid(G, dataset_A, dataset_B, batch_size=16, log_prefix=''):
    G.eval()

    def iter_gen():
        for sample in DataLoader(dataset_A, batch_size):
            yield G(sample.cuda())

    fid_val = fid.calculate_fid_given_iterators(
        DataLoader(dataset_B, batch_size),
        iter_gen(),
        compute_option=fid.FIDBackend.numpy,
        normalize_input=False,
        verbose=False)

    G.train()
    print(f'{log_prefix} FID: {fid_val: 0.3f}')
    return fid_val


def save_and_calculate_metrics(model, epoch, total_samples,
                               test_ds_A, test_ds_B, train_ds_A, train_ds_B,
                               models_dir):
    model.save_networks('latest')
    model.save_networks(epoch)

    fid_log_text = f'Epoch {epoch} - total_samples {total_samples} FID'
    G = model.get_generator()
    fid_test = calculate_fid(G, test_ds_A, test_ds_B, 16, f'{fid_log_text}-test')
    fid_train = calculate_fid(G, train_ds_A, train_ds_B, 16, f'{fid_log_text}-train')

    fid_log_dict = dict(fid_test=fid_test, fid_train=fid_train,
                        epoch=epoch, total_samples=total_samples)
    print_dict(**fid_log_dict)
    save_model(model, fid_log_dict, models_dir, total_samples)


def print_dict(**kwargs):
    for k, v in kwargs.items():
        print(f'{k}: {v}', end='; ')
    print('\n')


def save_model(model, fid_log_dict, root, step):
    state_dict = dict(model=model.state_dict(), **fid_log_dict)
    torch.save(state_dict, f'{root}/model_{step}.pt')


def adjust_ada_params_adaptive(opt, augment_pipe, mean_real_signs, batch_size):
    mean_real_sign = np.mean(mean_real_signs)

    adjust = np.sign(mean_real_sign - opt.ada_target) * \
            (batch_size * opt.ada_interval) / opt.ada_adjust_speed

    augment_pipe.p.copy_(
        (augment_pipe.p + adjust).max(misc.constant(0, device='cuda')))


if __name__ == '__main__':
    main()
