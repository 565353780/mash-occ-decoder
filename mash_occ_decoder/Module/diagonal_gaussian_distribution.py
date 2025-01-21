import torch
import numpy as np


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = 0 * self.var
            self.std =  0 * self.std
        return

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.mean.device
        )
        return x

    def kl(self, other=None):
        if other is None:
            kl = 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
            )
        else:
            kl = 0.5 * torch.mean(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

        if self.deterministic:
            return 0 * kl

        return kl

    def nll(self, sample, dims=[1, 2, 3]):
        logtwopi = np.log(2.0 * np.pi)
        nll = 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

        if self.deterministic:
            return 0 * nll

        return nll

    def mode(self):
        return self.mean
