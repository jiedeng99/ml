# Convolutionary Neural Networks

## 2026-06-09

Learned:
- optimizers including SGD + momentum, ADAM and different variants of ADAM
- nn.Linear
- Hessians
- Eigenvalues

Questions and Answers:
- What are the different optimizers in PyTorch and what does the optimizer do?
Optimizer is the component to update a neural network' parameters (weights and biases) to reduce loss function.
All optimizers are trying to answer the same question: Given the gradients, what is the smartest way to update the parameters so the loss decreases as quickly and reliably as possible.

The common optimizers to be used are as follows.

FeedForward networks, Reinforcement Learning            ADAM
CNN                                                     SGD + Momentum, ADAM
Transformer, LLM                                        AdamW

- What does tqdm do?
It gives a graphical bar showing the progress.

- What does dropout and batch norm do?
These are used in neural networks to stablize model training and improve the model. The purpose of dropout is to reduce overfitting by randomly zeroes some weights and biases. The purpose of BatchNorm is to make gradients behave more consistenly so that the model converges faster with good accuracy.

Dropout is widely used in all architectures.

In a feedforward network, a common pattern is Linear -> BatchNorm -> ReLU -> Dropout.
In CNN, a common pattern is Conv -> BatchNorm -> ReLU. Dropout is used less frequently.
In transformer, BatchNorm is replaced by LayerNorm, so it is usually Attention -> LayerNorm. 

- What are the differences between BatchNorm1d and BatchNorm2d?
The difference is whether the feature involves 1-dimensional data or 2-dimensional data across the batch.
In MLP, after nn.Linear, it is nn.BatchNorm1d.
In CNN, after nn.Conv2d, it is nn.BatchNorm2d.
In 3D CNN, such as videos, after nn.Conv3d, it is nn.BatchNorm3d.

- When doing validation, why are the datasets still divided into batches?
The main reasons are GPU's memory efficiency and better GPU utilization due to the fact that GPU is optimized for parallel computation. You can use a larger validation batch size than training batch size to get faster validation execution.

