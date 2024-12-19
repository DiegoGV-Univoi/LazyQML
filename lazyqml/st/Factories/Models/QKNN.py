from lazyqml.st.Interfaces.iModel import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pennylane as qml
from lazyqml.st.Factories.Circuits.fCircuits import CircuitFactory
from lazyqml.st.Utils.Utils import printer


class QKNN(Model):
    def __init__(self, nqubits, embedding, backend, shots, k=5, seed=1234):
        """
        Initialize the Quantum KNN model.
        Args:
            nqubits (int): Number of qubits for the quantum kernel.
            backend (str): Pennylane backend to use.
            shots (int): Number of shots for quantum measurements.
        """
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.k = k
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits, seed=seed, shots=self.shots)
        self.CircuitFactory = CircuitFactory(nqubits,nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None

    def _build_kernel(self):
        """Build the quantum kernel circuit."""

         # Get the embedding circuit from the circuit factory
        embedding_circuit = self.CircuitFactory.GetEmbeddingCircuit(self.embedding).getCircuit()

        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            qml.adjoint(embedding_circuit)(x2, wires=range(self.nqubits))
            return qml.probs(wires = range(self.nqubits))
        
        return kernel

    def _compute_distances(self, x1, x2):
        return 1-self.kernel_circ(x1, x2)[0]

    def fit(self, X, y):
        """
        Fit the Quantum KNN model.
        Args:
            X (ndarray): Training samples (n_samples, n_features).
            y (ndarray): Training labels (n_samples,).
        """
        self.X_train = X
        self.y_train = y
        self.q_distances = self._compute_distances
        printer.print("\t\tTraining the KNN...")
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric=self.q_distances)
        self.KNN.fit(X, y)

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            
            return self.KNN.predict(X)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise
        
    def getTrainableParameters(self):
        return "~"



    
    