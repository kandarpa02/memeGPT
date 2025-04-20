import torch
import os

class Checkpoints:
    def __init__(self):
        pass
    def save_checkpoints(self, model, optimizer, lora_adapter, epoch, loss, path="memeGPT/checkpoints"):
        os.makedirs(path, exist_ok=True)
        rounded_loss = round(loss, 4)
        checkpoint_path = os.path.join(path, f"checkpoint_epoch{epoch}_loss{rounded_loss}.pth")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lora_adapter_state_dict': lora_adapter.state_dict()  # Save LoRA adapter state
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at: {checkpoint_path}")


    def load_checkpoints(self, model, optimizer, lora_adapter, path):
        checkpoint = torch.load(path)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load LoRA adapter state
        lora_adapter.load_state_dict(checkpoint['lora_adapter_state_dict'])
        
        # Retrieve epoch and loss
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        return model, optimizer, lora_adapter, epoch, loss
