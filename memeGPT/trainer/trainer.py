import torch
from torch.amp import autocast 
from torch.amp import GradScaler

class Trainer:
    def __init__(self, model, optimizer, mix_precision=False, scaler=None, device='cuda'):

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.device = device

        self.model = model.to(self.device)

        self.optimizer     = optimizer
        self.mix_precision = mix_precision
        self.scaler        = scaler or GradScaler()
        self._loss         = 0.0

    def process(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.optimizer.zero_grad()

        if self.mix_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(**batch)
                loss   = output.loss.mean()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(**batch)
            loss   = output.loss.mean()
            loss.backward()
            self.optimizer.step()

        self._loss = loss.item()

    @property
    def loss(self):
        return self._loss
