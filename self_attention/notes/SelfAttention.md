# Self Attention

## Learned

- `Transformer encoder`
- `Multi-head attention`

## Questions and Answers

### Why shuffle is set to be True while loading training dataset train_loader set, but set to False while loading validation dataset?
During training, it is better to introduce randomness in the training data to avoid model overfitting. Validation doesn't update the model. It simple goes through all the validation data to calculate the loss to test how well the model performs. So there is no need to shuffle the validation data.

### Why is it called feedforward networks? What makes it different than fully connected network and MLP?
A neural network is called "feedforward network" because the network only flows forward. It doesn involve recurrence or feedback, in other words, cycles in the network. 
A fully connected network describes the connectivity patter of a layer: every neuron connects to every neuron in the next layer. A layer in a MLP network is a fully connected layer.  
A MLP (Multi-Layer Perceptron) network is a type of deep network that is both feedword and fully connected.

### What are the d_model, dim_feedforward in the transformer encoder?
The d_model is the number of features that will be used as input into the model.
The dim_feedforward is the width of the MLP layers inside the transformer encoder. Inside, the encoder layers will first map d_model -> dim_feedforward, and then map dim_feedforward -> d_model. 

### What is the difference in using the following:
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

The first only defines an encoder layer which consists of input -> attention -> ffn -> output.
The latter uses the defined layer, stack two (as defined by num_layers) of them together, run both of them independently. This way, the model learns more abstract relationships and provides more representational power. The wellknown models all stack many layers together. 

### What isn't optimizer.zero_grad() called before loss.backward(), but instead after scheduler.step()?
Mathematically it is the same. A lot of times optimizer.zero_grad() is up front before updating the weights. But a lot of the production code also have it after scheduler.step(). 

### What does torch.stack() do?
It stacks the tensors together by adding a new dimension.

### What does pin_memory do, why when loading inference data, it is not used?
pin_momory is used to speed up data copy from cpu to gpu. During training, each batch of data is transferred from CPU to GPU. Since training runs many epochs, even a small speedup in the cpu to gpu throughput will help with the training speed. 

During inference, however, the inference data is transferred just once, the speed improvement in data copy from cpu to gpu is negligible.

