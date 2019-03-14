from torch import nn
import numpy as np

class Flatten(nn.Module):
    """
    Flatten layer
    https://discuss.pytorch.org/t/non-legacy-view-module/131/11
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class View(nn.Module):
    """
    Resize layer
    https://github.com/pytorch/pytorch/issues/2486
    """

    def __init__(self, *view_args):
        super(View, self).__init__()
        self._view_args = view_args

    def forward(self, x):
        return x.view(x.size(0), *self._view_args)


class StateUpdateAutoEncoder(nn.Module):

    def __init__(self, latent_layer_size, height, width):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(1600, latent_layer_size),
            nn.ReLU(True),
        )

        self._decoder = nn.Sequential(
            nn.Linear(latent_layer_size, 1600),
            nn.ReLU(True),
            View(16, height, width),
            nn.ConvTranspose2d(16, 2, 3, stride=1, padding=1),
            nn.Softmax(),
        )

    def forward(self, input):
        latent = self._encoder(input)
        return self._decoder(latent)

    def reset_weights(self):
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)