# memeGPT

A GPT-style language model for meme/joke generation, built from scratch with PyTorch and trained using a fully modular and production-ready pipeline. Supports cloud training, MLflow logging, Docker, and CI/CD with GitHub Actions.

---

## Features

- GPT-2 architecture (`memeGPT`)
- Modular design (Tokenizer, Model, Optimizer, Trainer)
- Cloud training-ready with Docker & GPU
- MLflow for experiment tracking
- CI/CD via GitHub Actions
- Hugging Face `datasets` integration
- Checkpointing and mixed precision training

---

Key libraries:
- `torch`
- `transformers`
- `datasets`
- `mlflow`
- `pyyaml`
- `tqdm`

---

## Training on the Cloud (Docker + GPU)

### Requirements

- GPU VM (GCP, AWS, LambdaLabs, etc.)
- Docker with NVIDIA runtime (`nvidia-docker2` installed)
- Dataset mounted or accessible
- Port open for MLflow UI (optional)

---

### 1. Build Docker Image

```bash
docker build -t memegpt -f dockerfile.train .
```
---

### 2. Run Docker Container with GPU

```bash
docker run --gpus all -it -v $(pwd):/workspace memegpt
```

---

### 3. Start Training

```bash
python train.py config/train_config.yaml data/your_data.json 512 10
```

Arguments:

- `train_config.yaml`: Your config file
- `your_data.json`: Dataset path
- `512`: Batch size
- `10`: Epochs

---

## MLflow Integration

To start the MLflow UI locally:

```bash
mlflow ui --port 5000
```

Open in browser: [http://localhost:5000](http://localhost:5000)

For cloud VMs, forward the port:

```bash
ssh -L 5000:localhost:5000 user@your-cloud-ip
```

Training script is integrated with MLflow and logs:

- Training loss
- Validation loss
- Parameters
- Artifacts (optional)

---

## Testing

Organize your unit tests under `/tests` and run:

```bash
pytest
```

---

## CI/CD with GitHub Actions

On every push:

- Install dependencies
- Run unit tests (I haven't added tests yet, will be uploaded soon)
- Lint and check formatting

Workflow file: `.github/workflows/ci.yaml`

---

## Checkpoint Handling

Configurable through `train_config.yaml`:

```yaml
training:
  checkpoint_save: "checkpoints/best_model.pt"
  checkpoint_load: null
```

Set `checkpoint_load` if resuming training from a previous checkpoint.

---

## Dockerfile Example

Here’s a minimal `dockerfile.train`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["bash"]
```

---

## 🧠 Arguments Recap

```bash
python train.py <config_path> <data_path> <batch_size> <epochs>
```

Example:

```bash
python train.py config/train_config.yaml data/memes.json 512 10
```

---

