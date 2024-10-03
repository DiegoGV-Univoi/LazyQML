from Interfaces.iModel import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pennylane as qml
from time import time
from Interfaces.iModel import Model
from Circuits.fCircuits import CircuitFactory

class QSVM(Model):
    def __init__(self, nqubits, embedding):
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.device = qml.device('lightning.gpu', wires=nqubits)
        self.CircuitFactory = CircuitFactory(nqubits)
        self.kernel = self._build_kernel()

    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        # Get the embedding circuit from the circuit factory
        embedding_circuit = self.CircuitFactory.GetEmbeddingCircuit(self.embedding)
        
        # Define the kernel circuit with adjoint embedding for the quantum kernel
        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            qml.adjoint(embedding_circuit)(x2, wires=range(self.nqubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.nqubits)]
        
        return kernel
    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""
        num_samples_1 = len(X1)
        num_samples_2 = len(X2)
        kernel_matrix = np.zeros((num_samples_1, num_samples_2))

        for i in range(num_samples_1):
            for j in range(num_samples_2):
                kernel_matrix[i, j] = self.kernel(X1[i], X2[j]).sum()

        return kernel_matrix

    def fit(self, X, y):
        # Train the classical SVM with the quantum kernel
        print("Training the SVM...")
        self.svm = SVC(kernel=self.kernel)
        self.svm.fit(X, y)
        print("SVM training complete.")

    def predict(self, X):
        y_pred = self.svm.predict(X)
        return y_pred