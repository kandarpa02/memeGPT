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

epochs = int(config["training"].get("epochs", 5))
model_name = "distilgpt2"
model = Model(model_name)
tokenizer = text_tokenizer(model_name)()
DATA_PATH = sys.argv[2]

# Logging parameters using MLflow
mlflow.start_run()
mlflow.log_param("model_name", model_name)
mlflow.log_param("epochs", epochs)
mlflow.log_param("learning_rate", config["training"].get("lr", 1e-4))
mlflow.log_param("batch_size", int(sys.argv[3]))
mlflow.log_param("optimizer", config["training"].get("optimizer", "AdamW"))

lr = float(config["training"].get("lr", 1e-4))
alpha = float(config["training"].get("alpha", 0.99))
betas = tuple(config["training"].get("betas", [0.9, 0.999]))
weight_decay = float(config["training"].get("weight_decay", 0.01))
momentum = float(config["training"].get("momentum", 0.9))

C = Checkpoints()
_data = Load_data(DATA_PATH, tokenizer)

train_loader, _, val_loader = _data.dataloader(batch_size=int(sys.argv[3]))

model.freeze(
    num=int(config["training"].get("num", 0)),
    ln_=int(config["training"].get("ln", 0)), 
    wte=int(config["training"].get("wte", 0)),
    wpe=int(config["training"].get("wpe", 0))
)

total_params, trainable_params = model.num_params()
print(f"Total params: {total_params}")

model.lora(
    r=int(config['lora'].get('r', 8)),
    alpha_l=int(config['lora'].get('alpha_l', 16)),
    dropout=float(config['lora'].get('dropout', 0.5)),
)

optimizers = {
    'AdamW': _opt.AdamWOptimizer(model, lr, betas, weight_decay),
    'SGD': _opt.SGDOptimizer(model, lr, momentum, weight_decay),
    'RMSprop': _opt.RMSpropOptimizer(model, lr, alpha, weight_decay)
}

optimizer_name = config["training"].get("optimizer", "AdamW")
optimizer = optimizers.get(optimizer_name)
if optimizer is None:
    raise ValueError(f"Invalid optimizer: {optimizer_name}")
optimizer = optimizer()

if torch.cuda.device_count() > 1:
    model = DataParallel(model())
else:
    model = model()

model = model.to(device)

validation = Validation(model, val_data=val_loader, tokenizer=tokenizer, device=device)

def trainer(
        model_inp,
        mix_precision=config["training"].get("precision", True),
        epochs=epochs,
        load_checkpoint_=config["training"].get("checkpoint_load"),
        save_checkpoint_=config["training"].get("checkpoint_save")
    ):

    override_lr = config["training"].get("override_lr")
    override_opt = config["training"].get("override_optimizer")

    if load_checkpoint_ is None:
        optimizer_ = optimizer
        model_inp = model
        _val_loss = 0
        epochs_cp = 0
    else:
        model_inp, optimizer_, epochs_cp, _val_loss = C.load_checkpoints(model, optimizer, load_checkpoint_)
        print(f" Resuming training from epoch {epochs_cp} with val_loss = {_val_loss}")

        if override_opt:
            print(f" Overriding optimizer with {override_opt}")
            opt_fn = optimizers.get(override_opt)
            if opt_fn is None:
                raise ValueError(f"Unknown optimizer override: {override_opt}")
            optimizer_ = opt_fn()

        if override_lr is not None:
            print(f" Overriding learning rate with {override_lr}")
            for param_group in optimizer_.param_groups:
                param_group['lr'] = float(override_lr)

    Train = Trainer(model_inp, optimizer_, mix_precision, device=device)

    model_inp.train()
    for epoch in range(epochs_cp, epochs_cp + epochs):
        t0 = time.time()
        total_loss = 0
        val_loss = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            Train.process(batch)
            total_loss += Train.loss.item()

            current_val_loss = validation.val_loss()
            val_loss += current_val_loss

            mlflow.log_metric("train_loss", Train.loss.item(), step=epoch * len(train_loader) + step)
            mlflow.log_metric("val_loss", current_val_loss, step=epoch * len(train_loader) + step)

            sys.stdout.write(
                f"\rEpoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {Train.loss.item():.4f}"
            )
            sys.stdout.flush()

        avg_val_loss = val_loss / len(train_loader)

        C.save_checkpoints(
            model=model_inp,
            optimizer=optimizer_,
            epoch=epoch + 1,
            loss=avg_val_loss,
            path=save_checkpoint_
        )

        t1 = time.time()
        print(f"\nEpoch {epoch+1}/{epochs_cp + epochs} - Training Loss: {total_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {t1 - t0:.2f}s")

    print(f"Umm training is done senpai >_< !!!!!!!")

    mlflow.pytorch.log_model(model_inp, "model")
    mlflow.end_run()

if __name__ == "__main__":
    trainer(
        model,
        epochs=int(sys.argv[4])
    )
