import torch.optim as optim

class BaseOptimizer:
    def __init__(self, model, lr=1e-5, weight_decay=0.01):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self):
        raise NotImplementedError("Subclasses should implement this method")


class AdamWOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        super().__init__(model, lr, weight_decay)
        self.betas = betas
        self.eps = eps

    def __call__(self):
        return optim.AdamW(self.model.params(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)


class SGDOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-5, momentum=0.9, weight_decay=0.01):
        super().__init__(model, lr, weight_decay)
        self.momentum = momentum

    def __call__(self):
        return optim.SGD(self.model.params(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)


class AdagradOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-5, lr_decay=0, weight_decay=0.01):
        super().__init__(model, lr, weight_decay)
        self.lr_decay = lr_decay

    def __call__(self):
        return optim.Adagrad(self.model.params(), lr=self.lr, lr_decay=self.lr_decay, weight_decay=self.weight_decay)


class RMSpropOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0.01):
        super().__init__(model, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps

    def __call__(self):
        return optim.RMSprop(self.model.params(), lr=self.lr, alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay)

