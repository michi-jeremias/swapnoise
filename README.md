# swapnoise
Swapnoise dataset for pytorch

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
