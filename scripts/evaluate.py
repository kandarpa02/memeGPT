import torch
class Validation:
    def __init__(self, model, val_data, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.val_data = val_data

    def val_loss(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                loss = output.loss.mean()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
