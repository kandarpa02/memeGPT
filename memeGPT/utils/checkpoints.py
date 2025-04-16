import torch
import os

class Checkpoints:
    def __init__(self):
        pass
    def save_checkpoints(self, model, optimizer, epoch, loss, path="memeGPT/checkpoints"):

        os.makedirs(path, exist_ok=True)
        rounded_loss = round(loss, 4)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch{epoch}_loss{rounded_loss}.pth")

        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }
        torch.save(checkpoint, path+f"/checkpoint_{epoch}_{loss}.pth")
        print(f"Saved checkpoint at: {checkpoint_path}")

    def load_checkpoints(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss