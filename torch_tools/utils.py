import ast
from time import time
from tqdm.auto import tqdm
import torch


class DictWrapper:
    def __init__(self, **entries):
        self.__dict__.update(entries)


__failed_to_copy = []

def try_call(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exception:
            global __failed_to_copy
            if func.__name__ not in __failed_to_copy:
                print(f'failed to call {func.__name__}: {exception}')
                __failed_to_copy.append(func.__name__)
    return wrapper


def numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))


def wrap_with_tqdm(it, verbose=True, **kwargs):
    if not verbose:
        return it
    else:
        return tqdm(it, **kwargs)


class Timer(object):
    def __init__(self, cuda_sync=True):
        self._start = time()
        self._cumulative_time = 0.0
        self._resets_count = 0
        self._ignore_current = False
        self._cuda_sync = cuda_sync

    def reset(self):
        if self._cuda_sync:
            torch.cuda.synchronize()
        current_time = time()
        diff = current_time - self._start
        self._start = current_time
        if not self._ignore_current:
            self._resets_count += 1
            self._cumulative_time += diff
        self._ignore_current = False
        return diff

    def avg(self):
        if self._resets_count > 0:
            return self._cumulative_time / self._resets_count
        else:
            return 0.0

    def ignore_current(self):
        self._ignore_current = True


def txt_to_opts(file_path):
    opts = {}
    with open(file_path) as f:
        for line in f.readlines():
            line = line.replace(' ', '')
            if ':' in line:
                key, val = line.split(':')
                val = val[:-1]
                if val == 'true':
                    val = 'True'
                if val == 'false':
                    val = 'False'
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    pass

                opts[key] = val
    return DictWrapper(**opts)
