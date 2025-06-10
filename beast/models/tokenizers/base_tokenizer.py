import abc
import torch

class TokenizerBase(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return self.encode(input)

    @abc.abstractmethod
    def encode(self, trajs, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, tokens, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruct_traj(self, tokens, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reconstruction_error(self, raw_traj, **kwargs):
        raise NotImplementedError