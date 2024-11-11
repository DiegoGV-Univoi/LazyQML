import torch
import pennylane as qml
from time import time
import numpy as np
from lazyqml.Interfaces.iModel import Model
from lazyqml.Interfaces.iAnsatz import Ansatz
from lazyqml.Interfaces.iCircuit import Circuit
from lazyqml.Factories.Circuits.fCircuits import *
from lazyqml.Global.globalEnums import Backend
from lazyqml.Utils.Utils import printer
import warnings

class QNNBag(Model):

    def __init__(self, nqubits, backend, ansatz, embedding, n_class, layers, epochs, max_samples, max_features, n_features, n_estimators, shots, lr=0.01, batch_size=50, seed=1234) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.shots = shots
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.backend = backend
        self.deviceQ = qml.device(backend.value, wires=self.nqubits, seed=seed)
        self.device = None
        self.params_per_layer = None
        self.circuit_factory = CircuitFactory(self.nqubits, nlayers=layers)
        self.qnn = None
        self.params = None
        self._build_circuit()
        warnings.filterwarnings("ignore")
        # Initialize loss function based on the number of classes
        if self.n_class == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        # Retrieve parameters per layer from the ansatz
        self.params_per_layer = ansatz.getParameters()

        # Define the quantum circuit as a PennyLane qnode
        @qml.qnode(self.deviceQ, interface='torch', diff_method='adjoint')
        def circuit(x, theta):
            # Apply embedding and ansatz circuits
            embedding.getCircuit()(x, wires=range(self.nqubits))
            ansatz.getCircuit()(theta, wires=range(self.nqubits))

            if self.n_class==2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]
            
        self.qnn = circuit

    def forward(self, x, theta):
        qnn_output = self.qnn(x, theta)
        if self.n_class == 2:
            return qnn_output.squeeze()
        else:
            return torch.stack([output for output in qnn_output]).T

    def fit(self, X, y):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.backend == Backend.lightningGPU else "cpu")

        # Convert training data to torch tensors and transfer to device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.device)

        num_params = int(self.layers * self.params_per_layer)

        self.random_estimator_features = []

        for j in range(self.n_estimators):
            # Re-initialize parameters
            self.params = torch.randn((num_params,), device=self.device, requires_grad=True)
            self.opt = torch.optim.Adam([self.params], lr=self.lr)

            # Select random samples and features for each estimator
            random_estimator_samples = np.random.choice(a=X.shape[0], size=(int(self.max_samples * X.shape[0]),), replace=False)
            X_train_est = X[random_estimator_samples, :]
            y_train_est = y[random_estimator_samples]

            random_estimator_features = np.random.choice(a=X_train_est.shape[1], size=max(1, int(self.max_features * X_train_est.shape[1])), replace=False)
            self.random_estimator_features.append(random_estimator_features)

            # Filter data by selected features
            X_train_est = X_train_est[:, random_estimator_features]

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                list(zip(X_train_est, y_train_est)), batch_size=self.batch_size, shuffle=True, drop_last=True
            )

            start_time = time()

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in data_loader:
                    self.opt.zero_grad()
                    predictions = torch.stack([self.forward(x, self.params) for x in batch_X])
                    loss = self.criterion(predictions, batch_y)
                    loss.backward()
                    self.opt.step()
                    epoch_loss += loss.item()

                printer.print(f"\t\tEpoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

            printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

    def predict(self, X):
        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Initialize y_predictions with the correct shape
        if self.n_class == 2:
            y_predictions = torch.zeros(X_test.shape[0], 1, device=self.device)  # shape (batch_size, 1)
        else:
            y_predictions = torch.zeros(X_test.shape[0], self.n_class, device=self.device)  # shape (batch_size, n_class)

        for j in range(self.n_estimators):
            X_test_features = X_test[:, self.random_estimator_features[j]]
            y_pred = torch.stack([self.forward(x, self.params) for x in X_test_features])
            
            # Ensure the shape of y_pred matches the expectations
            if self.n_class == 2:
                y_pred = y_pred.view(-1, 1)  # shape (batch_size, 1) for binary classification
            else:
                # For multi-class, ensure it has the shape (batch_size, n_class)
                y_pred = y_pred.view(-1, self.n_class)  # shape (batch_size, n_class)

            y_predictions += y_pred  # Now should work without error

        # Average predictions over all estimators
        y_predictions /= self.n_estimators

        # For binary classification, use sigmoid to get probabilities
        if self.n_class == 2:
            return (torch.sigmoid(y_predictions.detach()).cpu().numpy() > 0.5).astype(int)  # Convert to binary predictions
        else:
            return torch.argmax(y_predictions.detach(), dim=1).cpu().numpy()  # For multi-class predictions