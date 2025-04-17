
# run in this order: >>> python train.py --config --data --batch_size --epochs

import torch
from torch.nn import DataParallel
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
import memeGPT.trainer.optimizer as _opt
from memeGPT.trainer.trainer import Trainer
from scripts.evaluate import Validation
import time, sys, warnings
import yaml
from memeGPT.data.dataloader import Load_data
from memeGPT.utils.checkpoints import Checkpoints
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")

with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = int(config["training"]["epochs"])
model_name = "distilgpt2"
model = Model(model_name)
tokenizer = text_tokenizer(model_name)()
DATA_PATH = sys.argv[2]

# Logging parameters using MLflow
mlflow.start_run()
mlflow.log_param("model_name", model_name)
mlflow.log_param("epochs", epochs)
mlflow.log_param("learning_rate", config["training"]["lr"])
mlflow.log_param("batch_size", int(sys.argv[3]))
mlflow.log_param("optimizer", config["training"]["optimizer"])

lr = float(config["training"]["lr"])
alpha = float(config["training"]["alpha"])
betas = tuple(config["training"]["betas"])
weight_decay = float(config["training"]["weight_decay"])
momentum = float(config["training"]["momentum"])

C = Checkpoints()
_data = Load_data(DATA_PATH, tokenizer)

train_loader, _, val_loader = _data.dataloader(batch_size=int(sys.argv[3]))

model.freeze(
    num=int(config["training"]["num"]),
    ln_=int(config["training"]["ln"]), 
    wte = int(config["training"]["wte"]),
    wpe = int(config["training"]["wpe"])
)

total_params, trainable_params = model.num_params()
print(f"Total params: {total_params}")

model.lora(
    r = int(config['lora']['r']),
    alpha_l= int(config['lora']['alpha_l']),
    dropout= int(config['lora']['dropout']),
)

optimzers = {
    'AdamW': _opt.AdamWOptimizer(model, lr, betas, weight_decay),
    'SGD': _opt.SGDOptimizer(model, lr, momentum, weight_decay),
    'RMSprop': _opt.RMSpropOptimizer(model, lr, alpha, weight_decay)
}

optimizer = optimzers[config["training"]["optimizer"]]
optimizer = optimizer()

if torch.cuda.device_count()>1:
    model = DataParallel(model())
else:
    model = model()

model = model.to(device)

validation = Validation(model, val_data=val_loader, tokenizer=tokenizer, device=device)

def trainer(
        model_inp,
        mix_precision=config["training"]["precision"],
        epochs=epochs,
        load_checkpoint_=config["training"]["checkpoint_load"],
        save_checkpoint_=config["training"]["checkpoint_save"]
    ):

    if load_checkpoint_ == None:
        optimizer_ = optimizer
        model_inp = model
        epochs_ = epochs
        best_val_loss = float('inf')
    else:
        model_inp, optimizer_, epochs_, best_val_loss = C.load_checkpoints(model, optimizer, load_checkpoint_)
    
    Train = Trainer(model_inp, optimizer_, mix_precision, device=device)

    model_inp.train()
    for epoch in range(epochs_):
        t0 = time.time()
        total_loss = 0
        val_loss = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            Train.process(batch)
            total_loss += Train.loss.item()

            current_val_loss = validation.val_loss()
            val_loss += current_val_loss

            # Logging metrics for each step using MLflow
            mlflow.log_metric("train_loss", Train.loss.item(), step=epoch * len(train_loader) + step)
            mlflow.log_metric("val_loss", current_val_loss, step=epoch * len(train_loader) + step)

            sys.stdout.write(
                f"\rEpoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {Train.loss.item():.4f}"
            )
            sys.stdout.flush()

            avg_val_loss = val_loss / len(train_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                C.save_checkpoints(
                    model=model_inp,
                    optimizer=optimizer_,
                    epoch=epoch,
                    loss=best_val_loss,
                    path=save_checkpoint_
                )

        t1 = time.time()
        print(f"\nEpoch {epoch+1}/{epochs} - Training Loss: {total_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {t1 - t0:.2f}s")
        print(f"Umm training is done senpai >_< !!!!!!!")
    mlflow.pytorch.log_model(model_inp, "model")
    mlflow.end_run()

if __name__ == "__main__":
    trainer(
        model,
        epochs=int(sys.argv[4])
    )
