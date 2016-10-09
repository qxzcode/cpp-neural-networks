# Neural network stuff

This contains some experiments with different types of neural networks in C++. There are three "tests," which use the `NeuralNet`, `DeepNeuralNet`, and `RecurrentNeuralNet` classes, all implemented by me from scratch.

### `NN_test.cpp`

This uses the `NeuralNet` class, which implements a simple two-layer NN (one level of connections).

### `DNN_test.cpp`

This uses the `DeepNeuralNet` class, which implements a deep neural network that can be dynamically initialized with an arbitrary number of arbitrarily-sized layers. Backpropagation is naturally used for training.

### `RNN_test.cpp`

This uses the `RecurrentNeuralNet` class, which implements a recurrent neural network with one hidden layer in each iteration. (There are three set of connections: input to hidden, previous hidden to current hidden, and hidden to output.)

I played around with training, but didn't get it to work very well.
