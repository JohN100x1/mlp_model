from time import time

import numpy as np


def softmax(a):
    """
    Softmax function.

    Parameters
    ----------
    a : K x h_out-dimensional pre-activations

    Returns
    -------
    y : K x h_out-dimensional output array
    """
    y = (np.exp(a).T / np.sum(np.exp(a), axis=1)).T
    return y


def cross_entropy(y, y_pred):
    """
    cross-entropy loss function.

    Parameters
    ----------
    y : K x h_out-dimensional output array
    y_pred : K x h_out-dimensional predicted output array

    Returns
    -------
    L : Int, cross-entropy loss
    """
    log_y = np.log(y_pred)
    L = -np.sum(y * log_y) / y.shape[0]
    return L


def accuracy(y, y_pred):
    """
    Calculates accuracy via comparison with the classfication index.

    Parameters
    ----------
    y : K x h_out-dimensional output array
    y_pred : K x h_out-dimensional predicted output array

    Returns
    -------
    acc : Int, Accuracy
    """
    logical_arr = np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)
    acc = np.mean(logical_arr)
    return acc


def get_batch(x, y, batch_size=128):
    """
    Get x and y batch from dataset.

    Parameters
    ----------
    x : K x h_in-dimensional array of inputs
    y : K x h_out-dimensional array of outputs
    batch_size : Int, Size of the batches

    Returns
    -------
    x_batch : K x h_in-dimensional array of inputs
    y_batch : K x h_out-dimensional array of outputs
    """
    indices = np.random.choice(x.shape[0], batch_size, replace=False)
    x_batch = x[indices, :]
    y_batch = y[indices, :]
    return x_batch, y_batch


class DenseLayer:
    """
    Dense layer object class.
    """

    def __init__(self, h_in, h_out, activation):
        """
        Dense layer initialisation.

        Parameters
        ----------
        h_in : Int, number of inputs
        h_out : Int, number of neurons
        activation : function, activation function
        """
        self.h_in = h_in
        self.h_out = h_out
        self.activation = activation
        # Initialise W and b
        self.W = np.random.uniform(-0.04, 0.04, size=(h_in, h_out))
        self.b = np.random.uniform(-0.04, 0.04, size=(h_out,))


class MlpModel:
    """
    Multi-layer perceptron model object class.
    """

    def __init__(self, input_size):
        """
        Multi-layer perceptron model initialisation.

        Parameters
        ----------
        input_size : int, Input size of Mlp model
        """
        self.input_size = input_size
        self.model = [
            DenseLayer(input_size, 400, np.tanh),
            DenseLayer(400, 400, np.tanh),
            DenseLayer(400, 400, np.tanh),
            DenseLayer(400, 400, np.tanh),
            DenseLayer(400, 400, np.tanh),
            DenseLayer(400, 10, softmax),
        ]
        self.num_layers = len(self.model)

    def predict(self, x):
        """
        Predict using Multi-layer perceptron model.

        Parameters
        ----------
        x : K x h_in-dimensional array of inputs

        Returns
        -------
        h : K x h_out-dimensional output array
        """
        h = x
        for layer in self.model:
            h = layer.activation(layer.b + h @ layer.W)
        return h

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=128,
        epochs=40,
        learning_rate=0.01,
    ):
        """
        Train MLP model using SGD.

        Parameters
        ----------
        X_train : K x h_in-dimensional array of training inputs
        y_train : K x h_out-dimensional array of training outputs
        X_val : K x h_in-dimensional array of validation inputs
        y_val : K x h_out-dimensional array of validation outputs
        batch_size : Int, Size of the batches used in SGD
        epochs : Int, Number of epochs
        learning_rate : Int, rate of learning for SGD

        Returns
        -------
        losses : 2 x L-dimensional array of training/validation losses
        accuracies : 2 x L-dimensional array of training/validation accuracies
        """
        num_batches = X_train.shape[0] // batch_size
        losses = np.zeros((2, epochs))
        accuracies = np.zeros((2, epochs))
        for epoch in range(epochs):
            t0 = time()
            for batch in range(num_batches):
                x_batch, y_batch = get_batch(X_train, y_train)
                h = [x_batch]
                a = [x_batch]
                # Forward pass
                for layer in self.model:
                    ak = layer.b + h[-1] @ layer.W
                    hk = layer.activation(ak)
                    a.append(ak)
                    h.append(hk)
                # Final layer Delta
                delta = h[-1] - y_batch
                for k in range(self.num_layers - 1, -1, -1):
                    layer = self.model[k]
                    # Calculate gradients
                    grad_w = (h[k].T @ delta) / batch_size
                    grad_b = np.mean(delta, axis=0)
                    # Calculate the next delta
                    if k == 0:
                        pass
                    else:
                        delta = (1 - np.tanh(a[k]) ** 2) * (delta @ layer.W.T)
                    # Apply gradients
                    layer.W -= learning_rate * grad_w
                    layer.b -= learning_rate * grad_b
            t1 = time()
            time_taken = t1 - t0
            # Training loss and accuracy
            y_pred_train = self.predict(X_train)
            losses[0, epoch] = cross_entropy(y_train, y_pred_train)
            accuracies[0, epoch] = accuracy(y_train, y_pred_train)
            # Validation loss and accuracy
            y_pred_val = self.predict(X_val)
            losses[1, epoch] = cross_entropy(y_val, y_pred_val)
            accuracies[1, epoch] = accuracy(y_val, y_pred_val)
            print(
                f"Epoch: {epoch}",
                f"Time taken: {time_taken}",
                f"train/val loss: {losses[:, epoch]}",
                f"train/val acc: {accuracies[:, epoch]}",
            )
        return losses, accuracies
