import torch
import pennylane as qml
from time import time
import numpy as np
from Interfaces.iModel import Model
from Interfaces.iAnsatz import Ansatz
from Interfaces.iCircuit import Circuit
from Circuits.fCircuits import *

class QNNTorch(Model):
    def __init__(self, nqubits, ansatz, embedding, n_class, layers, epochs, lr=0.01, batch_size=50) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = qml.device("lightning.gpu", wires=nqubits)
        self.params_per_layer = None
        self.circuit_factory = CircuitFactory(nqubits)
        self.qnn = None
        self.params = None
        self._build_circuit()

        # Initialize PyTorch optimizer and loss function
        self.opt = None  # Will initialize in fit method with model parameters
        self.criterion = torch.nn.CrossEntropyLoss()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz).getCircuit()
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        # Retrieve parameters per layer from the ansatz
        self.params_per_layer = ansatz.getParameters()

        # Define the quantum circuit as a PennyLane qnode
        @qml.qnode(self.device, interface='torch', diff_method='adjoint')
        def circuit(x, theta):
            embedding.getCircuit()(x, wires=range(self.nqubits))
            for i in range(self.layers):
                ansatz(theta[i * self.params_per_layer: (i + 1) * self.params_per_layer], wires=range(self.nqubits))
            observable = [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]
            return torch.tensor(observable)

        self.qnn = circuit

    def forward(self, x, theta):
        return self.qnn(x, theta)

    def fit(self, X, y):
        # Convert training data to torch tensors and transfer to device (CPU or GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")

        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        y_train = torch.tensor(y, dtype=torch.long).to(device)

        # Initialize parameters as torch tensors
        params = torch.randn(self.layers * self.params_per_layer, requires_grad=True, device=device)

        # Define optimizer
        self.opt = torch.optim.Adam([params], lr=self.lr)

        # Create data loader for batching
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        start_time = time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad()
                # Forward pass
                predictions = torch.stack([self.forward(x, params) for x in batch_X])
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                # Backward pass and optimization step
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()

            # Print the average loss for the epoch
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

        print(f"Training completed in {time() - start_time:.2f} seconds")
        self.params = params.detach().cpu()  # Save trained parameters

    def predict(self, X):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert test data to torch tensors
        X_test = torch.tensor(X, dtype=torch.float32).to(device)
        # Forward pass for prediction
        y_pred = torch.stack([self.forward(x, self.params) for x in X_test])
        # Apply softmax to get probabilities
        y_pred = torch.softmax(y_pred, dim=1)
        # Return the class with the highest probability
        return torch.argmax(y_pred, dim=1).cpu().numpy()