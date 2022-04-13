from time import time
from typing import Callable

import numpy as np


def softmax(a: np.array) -> np.array:
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


def cross_entropy(y: np.array, y_predicted: np.array) -> float:
    """
    cross-entropy loss function.

    Parameters
    ----------
    y : K x h_out-dimensional output array
    y_predicted : K x h_out-dimensional predicted output array

    Returns
    -------
    entropy : float, cross-entropy loss
    """
    log_y = np.log(y_predicted)
    entropy = -np.sum(y * log_y) / y.shape[0]
    return entropy


def accuracy(y: np.array, y_predicted: np.array) -> np.array:
    """
    Calculates accuracy via comparison with the classification index.

    Parameters
    ----------
    y : K x h_out-dimensional output array
    y_predicted : K x h_out-dimensional predicted output array

    Returns
    -------
    acc : h_out-dimensional array, Accuracy
    """
    logical_arr = np.argmax(y, axis=1) == np.argmax(y_predicted, axis=1)
    acc = np.mean(logical_arr)
    return acc


def get_batch(
    x: np.array, y: np.array, batch_size: int = 128
) -> tuple[np.array, np.array]:
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
    """Dense layer object class."""

    def __init__(self, h_in: int, h_out: int, activation: Callable):
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
        # Initialize W and b
        self.W = np.random.uniform(-0.04, 0.04, size=(h_in, h_out))
        self.b = np.random.uniform(-0.04, 0.04, size=(h_out,))


class MlpModel:
    """Multi-layer perceptron model object class."""

    def __init__(self, input_size: int):
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

    def predict(self, x: np.array) -> np.array:
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
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        batch_size: int = 128,
        epochs: int = 40,
        learning_rate: float = 0.01,
    ):
        """
        Train MLP model using SGD.

        Parameters
        ----------
        x_train : K x h_in-dimensional array of training inputs
        y_train : K x h_out-dimensional array of training outputs
        x_val : K x h_in-dimensional array of validation inputs
        y_val : K x h_out-dimensional array of validation outputs
        batch_size : Int, Size of the batches used in SGD
        epochs : Int, Number of epochs
        learning_rate : float, rate of learning for SGD

        Returns
        -------
        losses : 2 x L-dimensional array of training/validation losses
        accuracies : 2 x L-dimensional array of training/validation accuracies
        """
        num_batches = x_train.shape[0] // batch_size
        losses = np.zeros((2, epochs))
        accuracies = np.zeros((2, epochs))
        for epoch in range(epochs):
            t0 = time()
            for batch in range(num_batches):
                x_batch, y_batch = get_batch(x_train, y_train)
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
                    if k != 0:
                        delta = (1 - np.tanh(a[k]) ** 2) * (delta @ layer.W.T)
                    # Apply gradients
                    layer.W -= learning_rate * grad_w
                    layer.b -= learning_rate * grad_b
            t1 = time()
            time_taken = t1 - t0
            # Training loss and accuracy
            y_predicted_train = self.predict(x_train)
            losses[0, epoch] = cross_entropy(y_train, y_predicted_train)
            accuracies[0, epoch] = accuracy(y_train, y_predicted_train)
            # Validation loss and accuracy
            y_predicted_val = self.predict(x_val)
            losses[1, epoch] = cross_entropy(y_val, y_predicted_val)
            accuracies[1, epoch] = accuracy(y_val, y_predicted_val)
            print(
                f"Epoch: {epoch}",
                f"Time taken: {time_taken}",
                f"train/val loss: {losses[:, epoch]}",
                f"train/val acc: {accuracies[:, epoch]}",
            )
        return losses, accuracies
