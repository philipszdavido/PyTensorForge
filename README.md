# PyTensorForge

PyTensorForge is a Python-based deep learning and model experimentation framework focused on neural networks, transformer architectures, and efficient inference workflows. The project combines foundational tensor operations with modern training, tokenization, export, and serving capabilities for research-oriented AI development.

## Overview

The framework is designed for developers and researchers who want to explore machine learning concepts in a modular environment while also supporting practical LM workloads such as training, checkpointing, tokenization, generation, and deployment of OpenAI-compatible model servers.

PyTensorForge includes:

- Core tensor and numerical primitives
- Neural network layers and parameter management
- Activation functions and optimizer logic
- Model training and validation workflows
- GPT-style transformer implementations
- Tokenization, dataset preparation, and context extension
- Inference runtime and HTTP model serving

## Core Capabilities

### Neural Network Foundations

- Tensor, scalar, vector, and matrix abstractions
- Dense network layers and parameter-driven computation
- Basic activation functions including ReLU, Sigmoid, Tanh, ELU, and SELU
- Optimizers such as SGD
- Loss functions for regression and classification tasks

### Transformer and LLM Workflows

- Decoder-only GPT model architecture
- Tokenizer training and BPE byte-level tokenization
- Streaming dataset preparation and sharded corpora
- Checkpoint-based training and resumption
- Model export for inference use cases
- Generation runtime with KV-cache support and context length extension
- OpenAI-compatible server interface for deployment

### Training and Serving Infrastructure

- CLI tooling for training, evaluation, export, generation, and serving
- Model inspection and validation utilities
- Config-driven execution for reproducible setups
- Support for chat templates and assistant-style supervised finetuning patterns

## Project Structure

- `src/core/` – tensor and numeric primitives
- `src/neural/` – neural layers and parameters
- `src/activations/` – activation implementations
- `src/optimizers/` – optimization routines
- `src/loss/` – loss functions
- `src/initializers/` – initialization strategies
- `src/models/` – model definitions, including transformer and regression code
- `src/tokenization/` – tokenizer implementations and training utilities
- `src/data/` – corpus, sharding, streaming, validation, and dataset utilities
- `src/training/` – training pipeline and checkpoint management
- `src/inference/` – generation runtime, cache handling, and model execution
- `src/serving/` – HTTP serving infrastructure and API configuration
- `src/scaling/` – scaling utilities
- `configs/` – model and training configuration files
- `predict/` – prediction utilities and sample data
- `test/` – project-level examples and validation scripts
- `web/` – frontend assets for model interaction

## Requirements

- Python 3.10+
- NumPy
- PyYAML

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/philipszdavido/PyTensorForge.git
cd PyTensorForge
pip install -e .
```

This installs the `pytensorforge` CLI, which provides commands for training, evaluation, dataset preparation, generation, export, and serving.

## Quick Start

### 1. Train a model

```bash
pytensorforge train configs/train.yaml
```

### 2. Resume training from a checkpoint

```bash
pytensorforge resume latest --config configs/train.yaml
```

### 3. Evaluate a saved checkpoint

```bash
pytensorforge evaluate checkpoints/latest --config configs/train.yaml
```

### 4. Prepare a dataset for training

```bash
pytensorforge prepare-dataset data/corpus --tokenizer tokenizer.json --output data/shards
```

### 5. Export a trained model for inference

```bash
pytensorforge export checkpoints/latest --output exports/my-gpt --tokenizer tokenizer.json
```

### 6. Generate text from a model

```bash
pytensorforge generate exports/my-gpt --prompt "Once upon a time" --max-new-tokens 128
```

### 7. Serve the model through an OpenAI-compatible API

```bash
pytensorforge serve exports/my-gpt --name my-gpt --host 127.0.0.1 --port 8000
```

The server exposes an OpenAI-compatible interface at the configured host and port, allowing local or remote clients to interact with the model in a familiar API format.

## Documentation and Design Notes

The repository includes a set of project documents covering the major stages of development, including:

- `PHASE1_GPT_TRAINING.md`
- `PHASE2_DATA_PIPELINE.md`
- `PHASE3_INFERENCE.md`
- `PHASE4_SERVING.md`
- `PHASE5_TRAINING_EFFICIENCY.md`
- `PHASE6_CONTEXT_EXTENSION.md`
- `PHASE7_FINETUNING.md`
- `TOKENIZER.md`

These documents provide implementation context for the framework’s research and engineering goals, especially around GPT training, data preparation, and inference optimization.

## Example Workflow

A typical PyTensorForge workflow consists of:

1. Training or preparing a tokenizer
2. Building a dataset or sharded corpus
3. Training a model with checkpoints enabled
4. Exporting optimized checkpoints for inference
5. Serving the model via the built-in API or generating responses directly

## Contributing

Contributions are welcome. Developers are encouraged to open issues, propose enhancements, or submit pull requests for bug fixes, feature additions, performance improvements, and documentation updates.

## License

This repository does not currently include a dedicated LICENSE file in the root directory. Please consult the repository owner or project documentation for the applicable licensing terms before commercial or redistribution use.

## Project Status

PyTensorForge is a research-driven and experimentation-focused framework with support for both foundational ML primitives and modern transformer-based model workflows. It is best suited for learning, iterative development, and custom AI experimentation in a compact Python codebase.


