from torch import nn


def set_requires_grad(module, requires_grad=True):
    for p in module.parameters():
        p.requires_grad = requires_grad


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class HiddenModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = [module]

    def get_hidden(self):
        return self.module[0]

    def set_hidden(self, new_module):
        self.module[0] = new_module

    def __getattr__(self, name):
        try:
            return getattr(self.module[0], name)
        except AttributeError as e:
            raise e

    def forward(self, *args, **kwargs):
        return self.module[0](*args, **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input[0]


def blend(current, incoming, alpha_1=0.99, alpha_2=None):
    # handle dict-like or module inputs
    try:
        current_dict = current.state_dict()
        incoming_dict = incoming.state_dict()
    except Exception:
        current_dict = current
        incoming_dict = incoming

    if alpha_2 is None:
        alpha_2 = 1.0 - alpha_1

    for name in current_dict.keys():
        current_dict[name] = alpha_1 * current_dict[name] + alpha_2 * incoming_dict[name].data

    try:
        current.load_state_dict(current_dict)
    except Exception:
        current = current_dict

    return current
