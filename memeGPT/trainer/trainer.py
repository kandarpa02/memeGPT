from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, _optimizer, mix_precision = False, device='cpu'):
        self.model = model
        self._optimizer = _optimizer
        self.mix_precision = mix_precision
        self.device = device
        self.scaler = GradScaler(self.device)
        self._loss = 0

    def process(self, batch):
        self._optimizer.zero_grad()
        if self.mix_precision == True:
            with autocast():
                output = self.model(**batch)
                loss = output.loss
            
            loss = loss.mean()
            self._loss = loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()

        else:
            output = self.model(**self.batch)
            loss = output.loss.mean()
            self._loss = loss
            loss.backward()
            self._optimizer.step()
    
    @property
    def loss(self):
        return self._loss
    

