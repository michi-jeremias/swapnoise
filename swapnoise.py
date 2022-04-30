import torch
from torch.utils.data import TensorDataset


class SwapnoiseDataset(TensorDataset):
    """
    A TensorDataset that implements the swapnoise noise scheme.
    __getitem__ returns a row for tensor in the dataset (same as
    in TensorDataset), where every cell value is taken from the
    same column as the original cell, but from a random row with
    probability p, (p is set in p_noise).

    Arguments:
        p_noise (float or list of floats): probabilities for each tensor to
            replace the value of a cell with the value the same column but
            from a random row. For each tensor a swap probability must be supplied,
            even if it is 0. If all values in p_noise are 0 the behaviour is the
            same as a TensorDataset.
        tensors (tensor): tensors to load into the dataset.
    """

    def __init__(self, noise, *tensors):

        super(TensorDataset, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        if len(tensors) > 1:
            assert isinstance(noise, list)
            assert len(tensors) == len(noise)
            for n in noise:
                assert n >= 0 and n <= 1
            self.rows = tensors[0].shape[0]
        else:
            assert isinstance(noise, float)
            assert noise >= 0 and noise <= 1
            self.rows = tensors[0].shape[0]

        self.tensors = tensors
        self.noise = noise
        self.length = tensors[0].size(0)

        self.swap_idxs = []
        self.eff_noise = []
        self.idxs = (
            torch.tensor(range(self.tensors[0].shape[0])).unsqueeze(1).to(device)
        )
        self.shuffle(self.noise)

    def __getitem__(self, index):
        return tuple(
            torch.gather(tensor, 0, swap_idx[index].unsqueeze(0).to(device))
            for tensor, swap_idx in zip(self.tensors, self.swap_idxs)
        )

    def __len__(self):
        return self.tensors[0].size(0)

    def __repr__(self):
        return f"Swapnoise Dataset with {self.eff_noise} effective noise levels"

    def shuffle(self, noise=None):
        if noise is None:
            noise = self.noise
        self.swap_idxs = []
        self.eff_noise = []
        for tensor, n in zip(self.tensors, noise):
            self.idxs_rand = torch.randint(0, tensor.shape[0], tensor.shape).to(
                self.device
            )
            select = (
                torch.bernoulli(torch.zeros(tensor.shape) + n)
                .to(torch.long)
                .to(self.device)
            )
            select_inv = select * (-1) + 1
            self.swap_idxs.append((select * self.idxs_rand) + (select_inv * self.idxs))
            self.eff_noise.append(select.to(torch.float).mean())
