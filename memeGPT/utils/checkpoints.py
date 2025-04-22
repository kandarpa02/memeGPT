import torch
import os
from peft import PeftModel
from torch.amp import GradScaler

class Checkpoints:
    def __init__(self):
        pass

    def save_checkpoint(self, model: PeftModel, optimizer, scaler: GradScaler, epoch: int, loss: float, path: str = "memeGPT/checkpoints"):
        """Save full training state including LoRA adapters, optimizer, and scaler"""
        os.makedirs(path, exist_ok=True)
        rounded_loss = round(loss, 4)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch{epoch}_loss{rounded_loss}")

        model.save_pretrained(checkpoint_path)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, os.path.join(checkpoint_path, "training_state.pth"))

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, base_model, optimizer, scaler: GradScaler, model_path: str, train_state_path, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)

        state = torch.load(
            train_state_path ,
            map_location=device,
            weights_only= False
        )
        
        optimizer.load_state_dict(state['optimizer_state_dict'])
        for opt_state in optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if 'scaler_state_dict' in state: 
            scaler.load_state_dict(state['scaler_state_dict'])

        epoch = state['epoch']
        loss = state['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")

        return model, optimizer, scaler, epoch, loss