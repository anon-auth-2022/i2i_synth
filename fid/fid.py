#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import enum
import numpy as np
from scipy import linalg
import torch

from fid.utils import get_activations, np_to_torch, load_inception


class FIDBackend(enum.Enum):
    pytorch = 0
    numpy = 1


@torch.no_grad()
def calculate_activation_statistics(it, model, cuda=True, verbose=True):
    act = get_activations(it, model, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()

        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()

        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))

        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)

        for _ in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    mu1 = np_to_torch(mu1).cuda()
    mu2 = np_to_torch(mu2).cuda()
    sigma1 = np_to_torch(sigma1).cuda()
    sigma2 = np_to_torch(sigma2).cuda()

    assert mu1.shape == mu2.shape,  'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    if len(sigma1.shape) == 2:
        sigma1.unsqueeze_(0)
        sigma2.unsqueeze_(0)
        diff.unsqueeze_(0)

    covmean = sqrt_newton_schulz(sigma1 @ sigma2, 50)
    fids = torch.empty(diff.shape[0])
    for i in range(len(fids)):
        fids[i] = diff[i].norm() ** 2 + \
                  sigma1[i].trace() + sigma2[i].trace() - 2 * covmean[i].trace()
    return fids.squeeze()


def calculate_frechet_distance_numpy(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # !!! WARNING !!! USE THIS ONLY WHEN DATASET IS SUFFICIENTLY SMALL (USUALLY <= 1000)
    # SO THAT NAIVE PYTORCH IMPLEMENTATION (FOR SOME UNKNOWN REASON) RETURNS NAN OR -INF
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return torch.FloatTensor([
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    ])


@torch.no_grad()
def calculate_fid_given_iterators(ref_iterator, gen_iterators,
                                  dims=2048, cuda=True, compute_option=FIDBackend.pytorch,
                                  verbose=True, model=None, **model_kwargs):
    if model is None:
        model = load_inception(dims, **model_kwargs)
    if cuda:
        model.cuda()

    try:
        next(iter(gen_iterators[0]))
    except Exception:
        gen_iterators = [gen_iterators]

    m_ref, s_ref = calculate_activation_statistics(ref_iterator, model, cuda, verbose)

    fid_values = []
    for gen_iterator in gen_iterators:
        m_gen, s_gen = calculate_activation_statistics(gen_iterator, model, cuda, verbose)

        if compute_option is FIDBackend.pytorch:
            compute_fn = calculate_frechet_distance
        elif compute_option is FIDBackend.numpy:
            compute_fn = calculate_frechet_distance_numpy
        fid = compute_fn(m_ref, s_ref, m_gen, s_gen).cpu().numpy()
        fid_values.append(fid.item())

    return fid_values if len(gen_iterators) > 1 else fid_values[0]
