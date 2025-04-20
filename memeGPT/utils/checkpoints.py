import torch
import os

class Checkpoints:
    def __init__(self):
        pass

    def save_checkpoints(self, model, optimizer, epoch, loss, path="memeGPT/checkpoints"):
        os.makedirs(path, exist_ok=True)
        rounded_loss = round(loss, 4)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch{epoch}_loss{rounded_loss}.pth")

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Saved checkpoint at: {checkpoint_path}")

    def load_checkpoints(self, model: PeftModel, optimizer, path: str):

        ckpt = torch.load(path, map_location=lambda s, l: s.cuda() if torch.cuda.is_available() else s)

        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        epoch = ckpt['epoch']
        loss = ckpt['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")

        return model, optimizer, epoch, loss

