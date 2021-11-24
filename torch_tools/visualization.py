import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
try:
    import plotly.graph_objects as go
except:
    print('failed to import plotly')

from .data import UnannotatedDataset


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    else:
        tensor = ((tensor + 1) / 2).clamp(0, 1)

    return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def to_image_grid(tensor, adaptive=False, **kwargs):
    return to_image(make_grid(tensor, **kwargs), adaptive)


def overlayed(img, mask, channel=1, nrow=1):
    imgs_grid = make_grid(torch.clamp(img, -1, 1), nrow=nrow, padding=5)
    masks_grid = make_grid(mask, nrow=nrow, padding=5, pad_value=0.5).cpu()[0]

    overlay = 0.5 * (1 + imgs_grid.clone().cpu())
    overlay -= 0.5 * masks_grid.unsqueeze(0)
    overlay[channel] += masks_grid

    return torch.clamp(overlay, 0, 1)


def draw_with_mask(img, masks, names=None, horizontal=True):
    if isinstance(masks, torch.Tensor) and len(masks.shape) < 5:
        masks = [masks,]
    if horizontal:
        fig, axs = plt.subplots(ncols=len(masks) + 1, figsize=(3 * len(masks) + 3, 3), dpi=250)
    else:
        fig, axs = plt.subplots(nrows=len(masks) + 1, figsize=(len(img), len(masks) + 3), dpi=250)
    nrow = 1 if horizontal else img.shape[0]

    axs[0].axis('off')
    axs[0].set_title('original', fontsize=8)
    axs[0].imshow(to_image(make_grid(torch.clamp(img, -1, 1), nrow=nrow, padding=5)))

    for i, mask in enumerate(masks):
        overlay = overlayed(img, mask, int(i > 0), nrow)
        ax = axs[i + 1]
        ax.axis('off')
        if names is not None:
            ax.set_title(names[i], fontsize=8)
        ax.imshow(to_image(overlay, True))

    return fig


class SamplesGrid(object):
    def __init__(self, dataset_dir, size):
        self.dataset_dir = dataset_dir
        self.set_size(size)

    def __call__(self):
        grid = make_grid(next(iter(self.dataloader)), nrow=self.grid_size[0])
        return to_image(grid)

    def set_size(self, size):
        self.grid_size = size
        self.dataloader = torch.utils.data.DataLoader(
            UnannotatedDataset(self.dataset_dir), size[0] * size[1], shuffle=True)


def plotly_lines(fig, ticks, values, name, color, opacity=1, dash='dash'):
    fig.add_trace(go.Scatter(
        x=ticks,
        y=values,
        name=name,
        # marker=dict(size=10),
        showlegend=(name is not None),
        line=dict(color='rgba({}, {}, {}, {})'.format(*color, opacity), width=3, dash=dash),
    ))


def plotly_prepare_fig(fig):
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linewidth=2,
            linecolor='rgb(204, 204, 204)',
            # ticks='outside',
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            linecolor='rgb(204, 204, 204)',
            showticklabels=True,
        ),

        legend=go.layout.Legend(
            traceorder="normal",
            font=dict(
                size=18,
                family='Times New Roman',
                color='Black',
            ),
            x=0.25, y=0.25,
            bgcolor="rgba(255, 255, 255, 0.0)",
            orientation="v",
        ),
        # autosize=True,
        width=280,
        height=300,
        margin=dict(
            autoexpand=False,
            l=70,
            r=0,
            t=0,
            b=55,
        ),
        showlegend=True,
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        color='Black',
        title_font=dict(family='Times New Roman', size=24),
        tickfont=dict(size=16,
                      family='Times New Roman',
                      color='Black'),
    )
    fig.update_yaxes(
        title_font=dict(family='Times New Roman', size=24),
        color='Black',
        tickfont=dict(size=16,
                      family='Times New Roman',
                      color='Black'),
    )
