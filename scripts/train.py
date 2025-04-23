# run in this order: >>> python train.py --config --batch_size --epochs
import torch
from torch.amp import GradScaler
from memeGPT.utils.bnb_dataparallel import BnbDataParallel
from torch.utils.data import DataLoader
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
import memeGPT.trainer.optimizer as _opt
from memeGPT.trainer.trainer import Trainer
from memeGPT.data.dataloader import T3nsorLoader
from scripts.evaluate import Validation
import time, sys, warnings
import yaml
from memeGPT.utils.checkpoints import Checkpoints
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")

with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = int(sys.argv[3])
model_name = config["training"]["model"]
model = Model(model_name)
tokenizer = text_tokenizer(model_name)()

mlflow.start_run()
mlflow.log_param("model_name", model_name)
mlflow.log_param("epochs", epochs)
mlflow.log_param("learning_rate", config["training"].get("lr", 1e-4))
mlflow.log_param("batch_size", int(sys.argv[2]))
mlflow.log_param("optimizer", config["training"].get("optimizer", "AdamW"))

lr = float(config["training"].get("lr", 1e-4))
alpha = float(config["training"].get("alpha", 0.99))
betas = tuple(config["training"].get("betas", [0.9, 0.999]))
weight_decay = float(config["training"].get("weight_decay", 0.01))
momentum = float(config["training"].get("momentum", 0.9))


batch_size= int(sys.argv[2])

train_paths = config["data"]["train"]
val_paths   = config["data"]["val"] if config["data"].get("train_on_full", False) else config["data"]["train"]

train_ds = T3nsorLoader(train_paths)
train_mem = [train_ds[i] for i in range(len(train_ds))]
val_ds = T3nsorLoader(val_paths)
val_mem = [val_ds[i] for i in range(len(val_ds))]

train_loader = DataLoader(train_mem, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
val_loader   = DataLoader(val_mem,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

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
    target_modules=config['lora'].get('modules', ["c_attn", "c_proj"])
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

model = model()

if torch.cuda.device_count() > 1:
    model = BnbDataParallel(model)
else:
    model = model

primary = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(primary)

validation = Validation(model, val_data=val_loader, tokenizer=tokenizer, device=device)
C = Checkpoints()

def trainer(
        model_inp,
        mix_precision=config["training"].get("precision", True),
        epochs=epochs,
        load_model_checkpoint_=config["training"].get("checkpoint_load"),
        load_train_state = config["training"]['train_state'],
        save_checkpoint_=config["training"].get("checkpoint_save")
    ):

    override_lr = config["training"].get("override_lr")
    override_opt = config["training"].get("override_optimizer")

    if load_model_checkpoint_ is None:
        model_inp = model
    else:
        temp_scaler = GradScaler(enabled=mix_precision)
        model_inp, _, _, _, _ = C.load_checkpoint(
            base_model=model,
            optimizer=optimizer,
            scaler = temp_scaler,
            model_path=load_model_checkpoint_,
            train_state_path= load_train_state
        )

    if load_train_state is None:
        optimizer_ = optimizer
        _val_loss = 0
        epochs_cp = 0
        loaded_scaler = None
    else:
        temp_scaler = GradScaler(enabled=mix_precision)
        _, optimizer_, loaded_scaler, epochs_cp, _val_loss = C.load_checkpoint(
            base_model=model,
            optimizer=optimizer,
            scaler = temp_scaler,
            model_path=load_model_checkpoint_,
            train_state_path= load_train_state
        )

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

    Train = Trainer(
        model_inp, 
        optimizer_, 
        mix_precision = config['training'].get('precision', 'False'), 
        device='cuda',
        scaler= loaded_scaler if loaded_scaler else GradScaler(enabled=mix_precision)
    )

    for epoch in range(epochs_cp, epochs_cp + epochs):
        t0 = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            Train.process(batch)
            epoch_loss += Train.loss

            sys.stdout.write(
                f"\rEpoch {epoch+1}/{epochs_cp + epochs} | Step {step}/{len(train_loader)}"
            )
            sys.stdout.flush()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validation.val_loss()

        C.save_checkpoint(
        model=model_inp,
        optimizer=optimizer_,
        scaler=Train.scaler, 
        epoch=epoch + 1,
        loss=val_loss,
        path=save_checkpoint_
    )

        t1 = time.time()
        print(f"\nEpoch {epoch+1}/{epochs_cp + epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {t1 - t0:.2f}s")

    print(f"training is done senpai >_< !!!!!!!")

    mlflow.pytorch.log_model(model_inp, "model")
    mlflow.end_run()

if __name__ == "__main__":
    trainer(model)