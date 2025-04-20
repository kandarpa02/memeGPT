import torch
from torch.nn import DataParallel

class BnbDataParallel(DataParallel):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, *inputs, **kwargs):
        inputs = tuple(input_.to(self.device_ids[0]) for input_ in inputs)
        return super().forward(*inputs, **kwargs)
