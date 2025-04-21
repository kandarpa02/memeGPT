import torch
import os
from peft import PeftModel

class Checkpoints:
    def __init__(self):
        pass

    def save_checkpoints(self, model: PeftModel, optimizer, epoch, loss, path="memeGPT/checkpoints"):
        os.makedirs(path, exist_ok=True)
        rounded_loss = round(loss, 4)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch{epoch}_loss{rounded_loss}")

        model.save_pretrained(checkpoint_path)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_path, "training_state.pth"))

        print(f"Saved LoRA checkpoint at: {checkpoint_path}")

    def load_checkpoints(self, base_model, optimizer, path: str):
        
        model = PeftModel.from_pretrained(base_model, path)

        # Load training state
        state = torch.load(os.path.join(path, "training_state.pth"), map_location='cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.load_state_dict(state['optimizer_state_dict'])

        epoch = state['epoch']
        loss = state['loss']
        print(f"Loaded LoRA checkpoint from epoch {epoch} with loss {loss:.4f}")

        return model, optimizer, epoch, loss