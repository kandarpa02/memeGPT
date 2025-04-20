import torch
import sys
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
from memeGPT.data.dataloader import T3nsorLoader
import yaml
from peft import LoraConfig, TaskType

with open(sys.argv[4], 'r') as f:
    config = yaml.safe_load(f)

task_type_str = config['peft']['task_type']
task_type_enum = TaskType[task_type_str] 
peft_config = LoraConfig(
    task_type=task_type_enum,
    r=int(config['peft']['r']),
    lora_alpha=int(config['peft']['lora_alpha']),
    lora_dropout=float(config['peft']['lora_dropout']),
    target_modules=config['peft']['target_modules'],
    bias=config['peft']['bias']
)


class TestEvaluator:
    def __init__(self, model, test_data, device='cpu'):
        self.model = model.to(device)
        self.test_data = test_data
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
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
    tokenizer = text_tokenizer(model_name)()
    model = Model(model_name)
    path = sys.argv[2]
    weights_path = sys.argv[3]
    
    dataset = T3nsorLoader(path)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.load_weights(weights_path, model_name, peft_config, map_location='cuda')
    model = model()
    
    T_e = TestEvaluator(model, test_loader, device)

    avg_loss = T_e.evaluate()
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


    # How to use this: 
    #     in terminal >> python test_model.py model data_path weights_path peft_config
    