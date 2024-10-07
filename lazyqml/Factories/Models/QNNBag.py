import torch
import pennylane as qml
from time import time
import numpy as np
from Interfaces.iModel import Model
from Interfaces.iAnsatz import Ansatz
from Interfaces.iCircuit import Circuit
from Circuits.fCircuits import *

class QNNBag(Model):

<<<<<<< Updated upstream
    def __init__(self, nqubits, backend, ansatz, embedding, n_class, layers, epochs, max_samples, n_samples, max_features, n_features, n_estimators, lr=0.01, batch_size=50) -> None:
=======
    def __init__(self, nqubits, ansatz, embedding, n_class, layers, epochs, max_samples, n_samples, max_features, n_features, n_estimators, shots, lr=0.01, batch_size=50) -> None:
>>>>>>> Stashed changes
        super().__init__()
        self.nqubits = int(n_features * max_features)
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
        self.device = qml.device("lightning.gpu", wires=nqubits, shots=self.shots)
        self.params_per_layer = None
        self.circuit_factory = CircuitFactory(nqubits)
        self.qnn = None
        self.params = None
        self._build_circuit()

        # Initialize PyTorch optimizer and loss function
        self.opt = None  # Will initialize in fit method with model parameters
        if self.n_class==2:
            self.criterion = torch.nn.BCELoss()
        else:
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
            
            ansatz(theta, wires=range(self.nqubits), nlayers = self.layers)
            if self.n_class==2:
                observable = qml.expval(qml.PauliZ(0))
            else:
                observable = [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]
            return torch.tensor(observable)

        self.qnn = circuit

    def forward(self, x, theta):
        return (self.qnn(x, theta) + 1)/2

    def fit(self, X, y):
        # Convert training data to torch tensors and transfer to device (CPU or GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        # Define optimizer
        self.opt = torch.optim.Adam([params], lr=self.lr)
        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        y_train = torch.tensor(y, dtype=torch.long).to(device)
        self.params = []
        self.random_estimator_features = []

        for j in range(self.n_estimators):
            print(f"bagging - estimator {j}\n")
            # Initialize parameters as torch tensors
            params = torch.randn(self.layers * self.params_per_layer, requires_grad=True, device=device)

            random_estimator_samples = np.random.choice(a=X.shape[0], 
                                                 size=(int(self.max_samples*X.shape[0]),), 
                                                 #p=max_samples*np.ones(X_train.shape[0])
                                                 )

            X_train_est = X[random_estimator_samples,:]
            y_train_est = y[random_estimator_samples]

            
            random_estimator_features = np.random.choice(a=X_train_est.shape[1], 
                                                        size=(max(1,int(self.max_features*X_train_est.shape[1])),), 
                                                        replace=False, 
                                                        #p=max_features*np.ones(X_train_est.shape[1])
                                                        )
            
            X_train_est = X_train_est[:,random_estimator_features]
            
            
            X_train_est = torch.tensor(X_train_est, device=device).float()
            
            y_train_est = torch.tensor(y_train_est, device=device).float()

            # Create data loader for batching
            data_loader = torch.utils.data.DataLoader(
                list(zip(X_train_est, y_train_est)), batch_size=self.batch_size, shuffle=True, drop_last=True
            )
            start_time = time()
            
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in data_loader:
                    self.opt.zero_grad()
                    # Forward pass
                    predictions = torch.stack([self.forward(X_train_est, params) for x in batch_X])
                    # Compute loss
                    loss = self.criterion(predictions, batch_y)
                    # Backward pass and optimization step
                    loss.backward()
                    self.opt.step()
                    epoch_loss += loss.item()

                # Print the average loss for the epoch
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

            print(f"Training completed in {time() - start_time:.2f} seconds")
            self.params.append(params.detach().cpu())  # Save trained parameters
            self.random_estimator_features.append(random_estimator_features)

    def predict(self, X):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert test data to torch tensors
        X_test = torch.tensor(X, dtype=torch.float32).to(device)

        if self.n_class==2:
            y_predictions = torch.tensor(np.zeros(X_test.shape[0]))
        else:
            y_predictions = torch.tensor(np.zeros(X_test.shape[0], self.n_class))

        for j in range (self.n_estimators):

            X_test = X_test[:, self.random_estimator_features[j]]

            # Forward pass for prediction
            y_pred = torch.stack([self.forward(x, self.params[j]) for x in X_test])
            # Apply softmax to get probabilities
            y_pred = torch.softmax(y_pred, dim=1)
            y_predictions += y_pred
            # Return the class with the highest probability
        y_predictions = y_predictions/self.n_estimators

        return torch.argmax(y_predictions, dim=1).cpu().numpy()