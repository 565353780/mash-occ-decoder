from torch import nn
from typing import Union


class AutoEncoder(nn.Module):
    def __init__(self, hidden_dims: Union[list, None] = None):
        super(AutoEncoder, self).__init__()
        self.anchor_num = 400
        self.anchor_dim = 22

        if hidden_dims is None:
            data_dim = self.anchor_num * self.anchor_dim
            hidden_dims = [
                data_dim,
                data_dim // 2,
                data_dim // 2,
            ]

        self.encoder = nn.Sequential()
        self.encoder.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        for i in range(1, len(hidden_dims) - 1):
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.decoder = nn.Sequential()
        for i in range(len(hidden_dims) - 1, 1, -1):
            self.decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(hidden_dims[1], hidden_dims[0]))
        return

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape(x.shape[0], self.anchor_num, self.anchor_dim)
        return decoded
