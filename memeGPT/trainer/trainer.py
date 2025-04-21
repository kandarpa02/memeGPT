from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, _optimizer, mix_precision = False, scaler=None, device='cpu'):
        self.model = model
        self._optimizer = _optimizer
        self.mix_precision = mix_precision
        self.device = device
        self.scaler = scaler if scaler else GradScaler()
        self._loss = 0

    def process(self, batch):
        self._optimizer.zero_grad()
        if self.mix_precision == True:
            with autocast(device_type=self.device):
                output = self.model(**batch)
                loss = output.loss
            
            loss = loss.mean()
            self._loss = loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()

        else:
            output = self.model(**batch)
            loss = output.loss.mean()
            self._loss = loss
            loss.backward()
            self._optimizer.step()
    
    @property
    def loss(self):
        return self._loss
    

