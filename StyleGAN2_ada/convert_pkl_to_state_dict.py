import torch
import pickle
import argparse
import sys

from pathlib import Path

# from .training.networks import Generator
from gans.models.StyleGAN2_ada.training.networks import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to the .pkl stylegan2-ada-pytorch checkpoint')
    args = parser.parse_args()

    path = Path(args.path).resolve()
    out_path = path.parent / f'{path.stem}.pt'

    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
        torch.save({
            'G': ckpt['G'].state_dict(),
            'D': ckpt['D'].state_dict(),
            'G_ema': ckpt['G_ema'].state_dict()
        }, out_path)

    print(f'Saved to "{out_path}".')


if __name__ == '__main__':
    main()
