import os
import argparse
from io import BytesIO
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import lmdb

from torch_tools.utils import numerical_order, wrap_with_tqdm


def _filename(path):
    return os.path.basename(path).split('.')[0]


def imagenet_transform(size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop([size, size]),
        transforms.ToTensor(),
        normalize,])


def adaptive_image_resize(x, h, w):
    if x.size[0] > x.size[1]:
        t = transforms.Resize(h)
    else:
        t = transforms.Resize(w)
    return t(x)


class UnannotatedDataset(Dataset):
    def __init__(self, root_dir, numerical_sort=False, force_rgb=True,
                 transform=transforms.Compose(
                     [
                         transforms.ToTensor(),
                         lambda x: 2 * x - 1
                     ])):
        self.img_files = []
        for root, _, files in os.walk(root_dir):
            for file in numerical_order(files) if numerical_sort else sorted(files):
                if UnannotatedDataset.file_is_img(file):
                    self.img_files.append(os.path.join(root, file))
        self.transform = transform
        self.force_rgb = force_rgb

    @staticmethod
    def file_is_img(name):
        extension = os.path.basename(name).split('.')[-1]
        return extension in ['jpg', 'jpeg', 'png', 'webp', 'JPEG']

    def align_names(self, target_names):
        new_img_files = []
        img_files_names_dict = {_filename(f): f for f in self.img_files}
        for name in target_names:
            try:
                new_img_files.append(img_files_names_dict[_filename(name)])
            except KeyError:
                print('names mismatch: absent {}'.format(_filename(name)))
        self.img_files = new_img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.img_files[item])
        if self.force_rgb:
            img = img.convert('RGB')
        if self.transform is not None:
            return self.transform(img)
        else:
            return img


class LabeledDatasetImagesExtractor(Dataset):
    def __init__(self, ds, img_field=0):
        self.source = ds
        self.img_field = img_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item][self.img_field]


class DatasetLabelWrapper(Dataset):
    def __init__(self, ds, label, transform=None):
        self.source = ds
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        img = self.source[item]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.label[item])


class FilteredDataset(Dataset):
    def __init__(self, source, filterer=lambda i, s: s[1], target=[], verbose=True):
        self.source = source
        if not isinstance(target, list):
            target = [target]
        self.indices = [i for i, s in wrap_with_tqdm(enumerate(source), verbose)
                        if filterer(i, s) in target]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


class TransformedDataset(Dataset):
    def __init__(self, source, transform, img_index=0):
        self.source = source
        self.transform = transform
        self.img_index = img_index

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if isinstance(out, tuple):
            return self.transform(out[self.img_index]), out[1 - self.img_index]
        else:
            return self.transform(out)


class TensorsDataset(Dataset):
    def __init__(self, source_dir):
        self.source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)\
            if f.endswith('.pt')]

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, index):
        return torch.load(self.source_files[index])


class TensorDataset(Dataset):
    def __init__(self, source, device='cpu'):
        self.data = torch.load(source)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class LMDBDataset(Dataset):
    def __init__(self, path,
                 transform=transforms.Compose(
                     [transforms.ToTensor(), lambda x: 2 * x - 1])
                ):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = int(index)
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(8)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        if self.transform is not None:
            img = self.transform(img)
        return img


class RGBDataset(Dataset):
    def __init__(self, source_dataset):
        super(RGBDataset, self).__init__()
        self.source = source_dataset

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if out.shape[0] == 1:
            out = out.repeat([3, 1, 1])
        return out


def directory_rgb_iterator(path, batch_size=32, total=None):
    ds = UnannotatedDataset(path, transform=transforms.Compose(
        [
            lambda img: img.convert('RGB'),
            transforms.ToTensor(),
        ]
    ))
    if total is not None:
        ds = Subset(ds, list(range(min(total, len(ds)))))

    return DataLoader(ds, batch_size=batch_size)


def make_lmdb(args):
    dataset = UnannotatedDataset(args.data, transform=None)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        for i, img in enumerate(tqdm(dataset)):
            key = f'{str(i).zfill(8)}'.encode('utf-8')

            with env.begin(write=True) as txn:
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='png')
                txn.put(key, img_byte_arr.getvalue())

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(len(dataset)).encode('utf-8'))


def make_tensor(args):
    dataset = UnannotatedDataset(args.data, transform=None)
    sample_shape = np.array(dataset[0]).shape
    data = np.empty([len(dataset)] + list(sample_shape), dtype=np.uint8)

    for i, img in enumerate(tqdm(dataset)):
        data[i] = np.array(img)
    torch.save(torch.from_numpy(data).permute(0, 3, 1, 2), args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LMDB dataset creation')
    parser.add_argument('--data', type=str, help='path to the image dataset')
    parser.add_argument('--out', type=str, help='filename of the result lmdb dataset')
    parser.add_argument('--n_worker', type=int, default=8,
                        help='number of workers for preparing dataset')
    parser.add_argument('command', choices=['make_lmdb', 'make_tensor'])
    args = parser.parse_args()

    func = locals()[args.command]
    func(args)
