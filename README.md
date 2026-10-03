# PyTensorForge

PyTensorForge is a lightweight Python machine learning framework designed for learning, experimentation, and rapid prototyping. It provides a modular core for tensor operations, neural network layers, activation functions, optimizers, loss metrics, and model definitions.

## Key Features

- Core tensor and scalar abstractions for numerical computation
- Layer-based neural network support with dense and sequential model building
- Common activation functions such as ReLU, Sigmoid, Tanh, ELU, and SELU
- Standard optimization algorithms with SGD and batch optimization
- Loss functions including mean squared error, mean absolute error, and binary cross-entropy
- Weight initialization utilities with Glorot, He, LeCun, and zeros initializers
- Model implementations for regression and logistic classification

## Project Structure

- `src/core/` – Tensor, scalar, matrix, and vector primitives
- `src/neural/` – Neural layer abstractions, parameters, and dense layers
- `src/activations/` – Activation function implementations
- `src/optimizer/` – Optimizer base classes and SGD implementation
- `src/loss/` – Loss function abstractions and implementations
- `src/initializers/` – Weight initializer utilities
- `src/models/` – High-level model APIs for regression, classification, and sequential models
- `src/scaling/` – Data scaling utilities
- `src/models/gpt/` – decoder-only GPT Transformer (see `PHASE1_GPT_TRAINING.md`)
- `src/data/`, `src/training/` – streaming corpora, token shards, resumable trainer (`PHASE1_GPT_TRAINING.md`, `PHASE2_DATA_PIPELINE.md`)
- `src/tokenization/` – tokenizer abstraction, word-level and byte-level BPE (`TOKENIZER.md`)
- `src/inference/` – KV-cache inference runtime, continuous batching, chat templates (`PHASE3_INFERENCE.md`)
- `src/serving/` – OpenAI-compatible HTTP/SSE model server (`PHASE4_SERVING.md`)
- RoPE, activation checkpointing, bf16/fp16 numerics, parallel tokenization (`PHASE5_TRAINING_EFFICIENCY.md`)
- Context extension for RoPE models: extrapolate, linear, NTK (`PHASE6_CONTEXT_EXTENSION.md`)
- Fine-tuning (`init_from`) and chat training with assistant-only loss masking (`PHASE7_FINETUNING.md`)
- `web/` – TypeScript chat web client (prebuilt in `web/dist`)
- `predict/` – Prediction utilities and example datasets
- `test/` – Unit tests and example usage scripts

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/philipsz-davido/PyTensorForge.git
   cd PyTensorForge
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

   This installs the `pytensorforge` command (`train`, `resume`, `evaluate`, `inspect`,
   `tokenize`, `prepare-dataset`, `generate`, `export`, `serve`). Runtime dependencies are
   `numpy` and `pyyaml` only.

> Note: This repository is intended for educational use and may require additional dependencies depending on the development environment.

## Usage

Import components from the package and build models using the provided modules. Example:

```python
from src.neural.Dense import Dense
from src.neural.Layer import Layer
from src.optimizers.SGD import SGD
from src.loss.MSE import MSE
```

Refer to the `test/` directory for example scripts and usage patterns.

Serving a trained model:

```bash
pytensorforge export checkpoints/latest --output exports/my-gpt --tokenizer tokenizer/ --chat-template ptf-chat
pytensorforge serve exports/my-gpt --name my-gpt --api-key-file api_keys.txt
```

Then open `http://127.0.0.1:8000/` or point any OpenAI-compatible client at
`http://127.0.0.1:8000/v1`. See `serve.example.yaml` and `PHASE4_SERVING.md`.

## Contribution

Contributions are welcome. Please open an issue or submit a pull request for feature requests, bug fixes, or documentation improvements.

## License

This project is provided under the terms of the repository license. Check the project root for license details.


