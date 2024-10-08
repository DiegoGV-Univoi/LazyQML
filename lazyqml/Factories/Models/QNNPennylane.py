from Interfaces.iModel import Model
from Interfaces.iAnsatz import Ansatz
from Interfaces.iCircuit import Circuit
from Factories.Circuits.fCircuits import *

from time import time
import pennylane as qml


class QNNPennylane(Model):
    def __init__(self, nqubits, backend, ansatz, embedding, n_class, layers, epochs, shots, lr=0.01, seed=1234) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.shots = shots
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.device = qml.device(backend.value, wires=nqubits, seed=seed)
        self.params_per_layer = None
        self.circuit_factory = CircuitFactory(nqubits)
        self.qnn = None
        self.opt = qml.GradientDescentOptimizer(stepsize=self.lr)
        self.params = None

        # Build the quantum neural network circuit
        self._build_circuit()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz : Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz).getCircuit()
        embedding : Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        # Retrieve parameters per layer from the ansatz
        self.params_per_layer = ansatz.getParameters()

        @qml.qnode(self.device, interface='autograd')
        def circuit(x, theta):
            embedding.getCircuit()(x, wires=range(self.nqubits))
            
            ansatz(theta, wires=range(self.nqubits), nlayers = self.layers)

            if self.n_class==2:
                observable = qml.expval(qml.PauliZ(0))
            else:
                observable = [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]
            return np.array(observable)
        
        self.qnn = circuit

    def fit(self, X, y):
        initial_params = np.random.normal(1234, size=self.layers * self.params_per_layer)
        self.params = np.copy(initial_params)

        X_train_grad = np.array(X, requires_grad=False)
        y_train_grad = np.array(y, requires_grad=False)
        self.params = np.array(self.params, requires_grad=True)

        start = time.time()

        for i in range(self.steps):
            self.params = self.opt.step(self.calculate_ce_cost, X_train_grad, y_train_grad, self.params)[2]
            if i % 10 == 0:  # Print every 10 steps
                print(f"Step {i}/{self.steps}")
        
        print(f"Training completed in {time.time() - start} seconds")
        return self  # Return the fitted model instance
    
    def predict(self, X):
        y_pred = self.qnn(X, self.params)
        y_pred = np.array([[y_pred[j, i] for j in range(self.n_class)] for i in range(y_pred.shape[1])])
        y_pred = np.exp(y_pred) / sum(np.exp(y_pred))  # Apply softmax to get probabilities
        return np.argmax(y_pred, axis=1)  # Return the class with the highest probability