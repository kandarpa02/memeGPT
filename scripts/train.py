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

# Load config
with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparams
batch_size = int(sys.argv[2])
epochs     = int(sys.argv[3])
model_name = config["training"]["model"]

# Tokenizer setup
tokenizer = text_tokenizer(model_name)()

# MLflow logging
mlflow.start_run()
mlflow.log_param("model_name", model_name)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)
mlflow.log_param("learning_rate", config["training"].get("lr", 1e-4))
mlflow.log_param("optimizer", config["training"].get("optimizer", "AdamW"))

# DataLoader setup
train_ds = T3nsorLoader(config["data"]["train"])
val_ds   = T3nsorLoader(config["data"].get("val", config["data"]["train"]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def trainer():
    # Precision & checkpoint paths
    mix_precision    = bool(config["training"].get("precision", True))
    model_ckpt_path  = config["training"].get("checkpoint_load")
    train_state_path = config["training"].get("train_state")
    save_checkpoint  = config["training"].get("checkpoint_save")

    C = Checkpoints()

    # ----------------------------------------------------------------------------
    # 1) Build & prepare the model
    model = Model(model_name)
    model.freeze(
        num=int(config["training"].get("num", 0)),
        ln_=int(config["training"].get("ln", 0)),
        wte=int(config["training"].get("wte", 0)),
        wpe=int(config["training"].get("wpe", 0))
    )
    model.lora(
        r=int(config['lora'].get('r', 8)),
        alpha_l=int(config['lora'].get('alpha_l', 16)),
        dropout=float(config['lora'].get('dropout', 0.5)),
        target_modules=config['lora'].get('modules', ["c_attn", "c_proj"])
    )
    model = model()  # instantiate the nn.Module

    # ----------------------------------------------------------------------------
    # Prepare fresh optimizer & scaler (for possible resume)
    lr           = float(config["training"].get("lr", 1e-4))
    alpha        = float(config["training"].get("alpha", 0.99))
    betas        = tuple(config["training"].get("betas", [0.9, 0.999]))
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    momentum     = float(config["training"].get("momentum", 0.9))
    optim_name   = config["training"].get("optimizer", 'AdamW')

    OPTIMIZERS = {
        'AdamW':  _opt.AdamWOptimizer(model, lr, betas, weight_decay),
        'SGD':    _opt.SGDOptimizer(model, lr, momentum, weight_decay),
        'RMSprop':_opt.RMSpropOptimizer(model, lr, alpha, weight_decay)
    }
    Optim = OPTIMIZERS[config["training"]["optimizer"]]
    if Optim is None:
        raise ValueError(f"Invalid optimizer: {optim_name}")

    # # instantiate fresh optimizer & scaler
    # if optim_name == 'AdamW':
    #     optimizer = OptimCls(model, lr, betas, weight_decay)
    # elif optim_name == 'SGD':
    #     optimizer = OptimCls(model, lr, momentum, weight_decay)
    # else:
    #     optimizer = OptimCls(model, lr, alpha, weight_decay)
    scaler = GradScaler(enabled=mix_precision)

    # ----------------------------------------------------------------------------
    # 2) Load checkpoints: full train state or model-only
    start_epoch = 0
    last_val    = None

    if model_ckpt_path and train_state_path:
        # resume everything
        model, optimizer, scaler, start_epoch, last_val = C.load_checkpoint(
            base_model=model,
            optimizer=optimizer(),
            scaler=scaler,
            model_path=model_ckpt_path,
            train_state_path=train_state_path
        )
    elif model_ckpt_path:
        # load only model weights + LoRA adapters
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_ckpt_path)
        # optimizer & scaler remain fresh
    # else: fresh train

    # ----------------------------------------------------------------------------
    # 3) Wrap & move model
    if torch.cuda.device_count() > 1:
        model = BnbDataParallel(model)
    model = model.to(device)

    # ----------------------------------------------------------------------------
    # 4) Prepare Trainer & Validation
    validation  = Validation(model, val_data=val_loader, tokenizer=tokenizer, device=device)
    trainer_obj = Trainer(
        model,
        optimizer,
        mix_precision=mix_precision,
        device=device,
        scaler=scaler
    )

    # ----------------------------------------------------------------------------
    # 5) Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        t0, total_loss = time.time(), 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer_obj.process(batch)
            total_loss += trainer_obj.loss

        avg_train_loss = total_loss / len(train_loader)
        val_loss       = validation.val_loss()

        # ----------------------------------------------------------------------------
        # 6) Save checkpoint
        if save_checkpoint:
            C.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch+1,
                loss=val_loss,
                path=save_checkpoint
            )

        print(f"Epoch {epoch+1}/{start_epoch+epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Time: {time.time()-t0:.2f}s")

    # ----------------------------------------------------------------------------
    # 7) Finalize MLflow
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()


if __name__ == "__main__":
    trainer()
