import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from time import time

import sys
sys.path.append('/home/diego/LazyQML/lazyqml/')
from Global.globalEnums import *
from Factories.Circuits.fCircuits import CircuitFactory
from Utils.Utils import printer


# class HybridQuantumClassicalModel(nn.Module):
#     def __init__(self, 
#                 classical_model, 
#                 nqubits, 
#                 backend=Backend.lightningQubit, 
#                 ansatz='default', 
#                 embedding='default', 
#                 n_class=2, 
#                 layers=1, 
#                 shots=1000, 
#                 lr=0.01, 
#                 seed=1234,
#                 fine_tune_classical=False):
#         super().__init__()
        
#         # Classical model configuration
#         self.classical_model = classical_model
#         self.fine_tune_classical = fine_tune_classical
        
#         # Freeze or unfreeze classical model parameters
#         for param in self.classical_model.parameters():
#             param.requires_grad = fine_tune_classical
        
#         # Quantum Neural Network configuration
#         self.nqubits = nqubits
#         self.backend = backend
#         self.ansatz = ansatz
#         self.embedding = embedding
#         self.n_class = n_class
#         self.layers = layers
#         self.shots = shots
#         self.lr = lr
        
#         # Initialize quantum device
#         self.deviceQ = qml.device(backend.value, wires=nqubits, seed=seed) if backend != Backend.lightningGPU else qml.device(backend.value, wires=nqubits)
        
#         # Circuit factory
#         self.circuit_factory = CircuitFactory(nqubits, nlayers=layers)
        
#         # Build quantum circuit
#         self._build_quantum_circuit()
        
#         # Determine classical output dimension
#         classical_output_dim = self._get_classical_output_dim()
        
#         # Bridge layer
#         self.bridge = nn.Linear(classical_output_dim, nqubits)
        
#         # Final classification layer
#         if n_class == 2:
#             self.classifier = nn.Linear(1, 1)
#             self.criterion = nn.BCEWithLogitsLoss()
#         else:
#             self.classifier = nn.Linear(1, n_class)
#             self.criterion = nn.CrossEntropyLoss()

#     def _get_classical_output_dim(self):
#         with torch.no_grad():
#             dummy_input = torch.zeros((1,) + self.classical_model.input_shape)
#             classical_output = self.classical_model(dummy_input)
#             return classical_output.numel()

#     def _build_quantum_circuit(self):
#         ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
#         embedding = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        
#         @qml.qnode(self.deviceQ, interface='torch', diff_method='adjoint')
#         def quantum_circuit(x, theta):
#             embedding.getCircuit()(x, wires=range(self.nqubits))
#             ansatz.getCircuit()(theta, wires=range(self.nqubits))
            
#             return qml.expval(qml.PauliZ(0))
        
#         self.quantum_circuit = quantum_circuit
        
#         # Initialize quantum parameters
#         self.quantum_params = nn.Parameter(torch.randn(self.layers * ansatz.getParameters()))

#     def forward(self, x):
#         # Extract features from classical model
#         classical_features = self.classical_model(x)
        
#         # Flatten and ensure 2D tensor
#         classical_features = classical_features.view(x.size(0), -1)
        
#         # Bridge layer transformation
#         bridged_features = self.bridge(classical_features)
        
#         # Quantum circuit processing
#         qnn_output = torch.stack([
#             self.quantum_circuit(feat, self.quantum_params) 
#             for feat in bridged_features
#         ]).unsqueeze(1)
        
#         return self.classifier(qnn_output).squeeze()

#     def predict(self, X):
#         with torch.no_grad():
#             # Ensure input is converted to a PyTorch tensor
#             if not isinstance(X, torch.Tensor):
#                 X = torch.tensor(X, dtype=torch.float32)
            
#             outputs = self.forward(X)
#             return (torch.sigmoid(outputs) > 0.5).float()
    
    
#     def fit(self, X, y, batch_size=32, epochs=10, lr=None):
#         # Use provided learning rate or default to class initialization
#         learning_rate = lr if lr is not None else self.lr
        
#         # Prepare data
#         X_train = torch.tensor(X, dtype=torch.float32)
#         y_train = torch.tensor(y, dtype=torch.long if self.n_class > 2 else torch.float32)
        
#         # Create data loader
#         data_loader = torch.utils.data.DataLoader(
#             list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
#         )
        
#         # Collect trainable parameters
#         params_to_optimize = []
#         if self.fine_tune_classical:
#             params_to_optimize.extend(self.classical_model.parameters())
#         params_to_optimize.extend(list(self.bridge.parameters()))
#         params_to_optimize.append(self.quantum_params)
#         params_to_optimize.extend(list(self.classifier.parameters()))
        
#         # Optimizer
#         optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
        
#         # Training loop
#         start_time = time()
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             for batch_X, batch_y in data_loader:
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 predictions = self.forward(batch_X)
#                 loss = self.criterion(predictions, batch_y)
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
            
#             # Print epoch progress
#             printer.print(f"\t\tEpoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
        
#         printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

class HybridQuantumClassicalModel(nn.Module):
    def __init__(self, 
                 classical_model, 
                 nqubits, 
                 backend=Backend.lightningQubit, 
                 ansatz='default', 
                 embedding='default', 
                 n_class=2, 
                 layers=1, 
                 shots=1000, 
                 lr=0.01, 
                 seed=1234,
                 fine_tune_classical=False,
                 input_shape=None):  # Add input_shape argument
        super().__init__()
        
        self.classical_model = classical_model
        self.fine_tune_classical = fine_tune_classical
        
        # Freeze or unfreeze classical model parameters
        for param in self.classical_model.parameters():
            param.requires_grad = fine_tune_classical
        
        # Quantum Neural Network configuration
        self.nqubits = nqubits
        self.backend = backend
        self.ansatz = ansatz
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.shots = shots
        self.lr = lr
        
        # Initialize quantum device
        self.deviceQ = qml.device(backend.value, wires=nqubits, seed=seed) if backend != Backend.lightningGPU else qml.device(backend.value, wires=nqubits)
        
        # Circuit factory
        self.circuit_factory = CircuitFactory(nqubits, nlayers=layers)
        
        # Build quantum circuit
        self._build_quantum_circuit()
        
        # Determine classical output dimension
        classical_output_dim = self._infer_classical_output_dim(input_shape)  # Pass input_shape to inference method
        
        # Bridge layer
        self.bridge = nn.Linear(classical_output_dim, nqubits)
        
        # Final classification layer
        if n_class == 2:
            self.classifier = nn.Linear(1, 1)
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.classifier = nn.Linear(1, n_class)
            self.criterion = nn.CrossEntropyLoss()

    def _infer_classical_output_dim(self, input_shape):
        # Use explicit input shape if provided
        if input_shape is not None:
            dummy_input = torch.zeros((1,) + input_shape)
        else:
            # Attempt to infer the input shape dynamically
            dummy_input = self._infer_classical_input_shape()
            if dummy_input is None:
                raise ValueError("Unable to infer the input shape of the classical model.")

        with torch.no_grad():
            classical_output = self.classical_model(dummy_input)
            return classical_output.numel()

    def _infer_classical_input_shape(self):
        # This method remains for automatic inference, but we now allow for manual override.
        # Here, you can attempt more advanced inference if desired.
        try:
            # Infer input shape for feedforward networks
            return (1, self.classical_model.input_shape[0])
        except AttributeError:
            return None

    def _build_quantum_circuit(self):
        ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        
        @qml.qnode(self.deviceQ, interface='torch', diff_method='adjoint')
        def quantum_circuit(x, theta):
            embedding.getCircuit()(x, wires=range(self.nqubits))
            ansatz.getCircuit()(theta, wires=range(self.nqubits))
            
            return qml.expval(qml.PauliZ(0))
        
        self.quantum_circuit = quantum_circuit
        
        # Initialize quantum parameters
        self.quantum_params = nn.Parameter(torch.randn(self.layers * ansatz.getParameters()))

    def forward(self, x):
        # Extract features from classical model
        classical_features = self.classical_model(x)

        # Handle LSTM outputs if they are 3D tensors
        if classical_features.dim() == 3:
            # Use the last time step
            classical_features = classical_features[:, -1, :]
            # Or flatten the entire sequence
            # classical_features = classical_features.reshape(x.size(0), -1)

        # Ensure the tensor is 2D
        classical_features = classical_features.view(x.size(0), -1)

        # Bridge layer transformation
        bridged_features = self.bridge(classical_features)

        # Quantum circuit processing
        qnn_output = torch.stack([
            self.quantum_circuit(feat, self.quantum_params)
            for feat in bridged_features
        ]).unsqueeze(1)

        return self.classifier(qnn_output).squeeze()

    def predict(self, X):
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            outputs = self.forward(X)
            return (torch.sigmoid(outputs) > 0.5).float()

    def fit(self, X, y, batch_size=32, epochs=10, lr=None):
        learning_rate = lr if lr is not None else self.lr
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.long if self.n_class > 2 else torch.float32)
        
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
        )
        
        params_to_optimize = []
        if self.fine_tune_classical:
            params_to_optimize.extend(self.classical_model.parameters())
        params_to_optimize.extend(list(self.bridge.parameters()))
        params_to_optimize.append(self.quantum_params)
        params_to_optimize.extend(list(self.classifier.parameters()))
        
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
        start_time = time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            printer.print(f"\t\tEpoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
        printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

            
"""
Example of use:
"""

class SimpleNN(nn.Module):
    def __init__(self, input_shape, output_classes):
        super().__init__()
        self.input_hape = input_shape
        self.output_classes = output_classes
        self.layers = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        
        # Final output layer (number of classes)
        self.output_layer = nn.Linear(32, self.output_classes)
    
    def forward(self, x):
        # Ensure x is flattened into [batch_size, features]
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        
        # Output layer to produce the class logits (for softmax or other loss functions)
        x = self.output_layer(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Ensure padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling reduces spatial dimensions
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 1 * 1, 128),  # Match flattened size
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Use the last hidden state
        hidden = hidden[-1]  # Take the final layer's hidden state
        return self.fc(hidden)


def main():
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import torch
    from torch import nn
    
    printer.set_verbose(True)
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classical_models = {
        "SimpleNN": (SimpleNN((X_train_scaled.shape[1],), output_classes=len(set(y))), (X_train_scaled.shape[1],)),
        "CNNModel": (CNNModel(input_channels=1, num_classes=len(set(y))), (1, 2, 2)),  # Simulated image shape
        "LSTMModel": (LSTMModel(input_dim=X_train_scaled.shape[1], hidden_dim=64, num_classes=len(set(y))), (X_train_scaled.shape[1],)),
    }

    for model_name, (classical_model, input_shape) in classical_models.items():
        print(f"\nTesting Hybrid Model with {model_name}...")
        
        if model_name == "CNNModel":
            # Reshape input into [batch_size, channels, height, width]
            # Assuming features can be reshaped into a 2x2 grid (Iris dataset has 4 features)
            X_train_processed = X_train_scaled.reshape(-1, 1, 2, 2)  # 1 channel, 2x2 grid
            X_test_processed = X_test_scaled.reshape(-1, 1, 2, 2)
        if model_name == "LSTMModel":
            # Reshape input into [batch_size, seq_len, features]
            X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])  # 1 time step
            X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        else:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled

        hybrid_model = HybridQuantumClassicalModel(
            classical_model, 
            nqubits=4,
            backend=Backend.lightningQubit,
            ansatz=Ansatzs.TREE_TENSOR,
            embedding=Embedding.RX,
            n_class=len(set(y)),
            layers=5,
            fine_tune_classical=True,
            lr=0.01,
            input_shape=input_shape
        )
        
        hybrid_model.fit(X_train_processed, y_train, epochs=10)
        predictions = hybrid_model.predict(X_test_processed)
        
        predictions = torch.tensor(predictions)
        if predictions.dim() == 1:
            predicted_classes = predictions
        elif predictions.dim() == 2:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
        
        y_test_tensor = torch.tensor(y_test)
        accuracy = (predicted_classes == y_test_tensor).float().mean()
        print(f"Hybrid Model with {model_name} Accuracy: {accuracy.item():.4f}")
        print(f"Predictions: {predicted_classes.tolist()}")

if __name__ == "__main__":
    main()