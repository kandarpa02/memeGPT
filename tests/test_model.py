import torch
import sys
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
from memeGPT.data.dataloader import T3nsorLoader
from peft import prepare_model_for_kbit_training

class TestEvaluator:
    def __init__(self, model, test_data, device='cpu'):
        self.model = model
        self.test_data = test_data
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.inference_mode():  # Better for quantized models
            for batch in self.test_data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                loss = output.loss.mean()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

if __name__ == "__main__":
    model_name = sys.argv[1]
    data_path = sys.argv[2]
    peft_path = sys.argv[3]

    # Initialize with original quantized config
    wrapper = Model(model_name)
    tokenizer = text_tokenizer(model_name)()
    
    # Load LoRA adapters
    wrapper.load_weights(peft_path, model_name)
    model = wrapper.model
    
    dataset = T3nsorLoader(data_path)
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False,
        pin_memory=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T_e = TestEvaluator(model, test_loader, device)

    avg_loss = T_e.evaluate()
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")