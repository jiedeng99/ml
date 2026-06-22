# Convolutional Neural Networks

## Learned

- `Conv2d`

## Questions and Answers

### What are the different optimizers in PyTorch, and what does an optimizer do?

An optimizer updates a neural network's parameters, such as weights and biases, to reduce the loss function.

All optimizers try to answer the same question: given the gradients, what is the smartest way to update the parameters so the loss decreases as quickly and reliably as possible?

Common choices:

| Architecture or task | Common optimizers |
| --- | --- |
| Feedforward networks | `Adam` |
| Reinforcement learning | `Adam` |
| CNNs | `SGD + Momentum`, `Adam` |
| Transformers and LLMs | `AdamW` |

### What does `tqdm` do?

`tqdm` displays a progress bar for loops, which makes it easier to track training, validation, or data processing progress.

### What do dropout and batch normalization do?

Dropout and batch normalization are used to stabilize training and improve model performance.

- Dropout reduces overfitting by randomly zeroing out some activations during training.
- Batch normalization helps gradients behave more consistently, which can make the model converge faster and reach better accuracy.

Dropout is widely used in all architectures.

Common patterns:

| Architecture | Common pattern |
| --- | --- |
| Feedforward network | `Linear -> BatchNorm -> ReLU -> Dropout` |
| CNN | `Conv -> BatchNorm -> ReLU` |
| Transformer | `Attention -> LayerNorm` |

In CNNs, dropout is used less frequently than in feedforward networks. In transformers, `BatchNorm` is usually replaced by `LayerNorm`.

### What are the differences between `BatchNorm1d` and `BatchNorm2d`?

The difference is whether the features are 1-dimensional or 2-dimensional across the batch.

| Model type | Layer | Batch normalization |
| --- | --- | --- |
| MLP | `nn.Linear` | `nn.BatchNorm1d` |
| CNN | `nn.Conv2d` | `nn.BatchNorm2d` |
| 3D CNN, such as video models | `nn.Conv3d` | `nn.BatchNorm3d` |

### When doing validation, why are datasets still divided into batches?

The main reasons are GPU memory efficiency and better GPU utilization. GPUs are optimized for parallel computation, so batching validation data is usually faster and more memory-efficient than processing examples one at a time.

You can often use a larger validation batch size than training batch size because validation does not need to store gradients for backpropagation.
