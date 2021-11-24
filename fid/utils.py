from tqdm.auto import tqdm
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

from fid.inception import InceptionV3


def np_to_torch(a):
    if isinstance(a, type(np.array([1]))):
        a = torch.from_numpy(a)
    return a


def get_activations(it, model, cuda=True, verbose=True):
    model.eval()
    stacked_features = []

    for images in tqdm(it) if verbose else it:
        if cuda:
            images = images.cuda()
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        features = model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        stacked_features.append(features.cpu().view(features.shape[0], -1).numpy())

    return np.vstack(stacked_features)


def load_inception(dims, **model_kwargs):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    return InceptionV3([block_idx], **model_kwargs)
