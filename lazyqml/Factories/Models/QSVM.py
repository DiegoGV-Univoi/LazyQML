from lazyqml.Interfaces.iModel import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pennylane as qml
from time import time
from lazyqml.Factories.Circuits.fCircuits import CircuitFactory
from lazyqml.Utils.Utils import printer

class QSVM(Model):
    def __init__(self, nqubits, embedding, backend, shots, seed=1234):
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits, seed=seed)
        self.CircuitFactory = CircuitFactory(nqubits,nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None
        
    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        # Get the embedding circuit from the circuit factory
        embedding_circuit = self.CircuitFactory.GetEmbeddingCircuit(self.embedding).getCircuit()
        
        # Define the kernel circuit with adjoint embedding for the quantum kernel
        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            qml.adjoint(embedding_circuit)(x2, wires=range(self.nqubits))
            return qml.probs(wires = range(self.nqubits))
        
        return kernel
    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""
        num_samples_1 = len(X1)
        num_samples_2 = len(X2)
        kernel_matrix = np.zeros((num_samples_1, num_samples_2))

        for i in range(num_samples_1):
            for j in range(num_samples_2):
                kernel_matrix[i, j] = self.kernel_circ(X1[i], X2[j]).sum()

        #return kernel_matrix
        return np.array([[self.kernel_circ(x1 , x2)[0] for x2 in X2]for x1 in X1])

    def fit(self, X, y):
        self.X_train = X
        self.qkernel = self._quantum_kernel(X,X)
        # Train the classical SVM with the quantum kernel
        printer.print("\t\tTraining the SVM...")
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tSVM training complete.")

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
        
            printer.print(f"\t\t\tComputing kernel between test and training data...")
            
            # Compute kernel between test data and training data
            kernel_test = self._quantum_kernel(X, self.X_train)
            
            if kernel_test.shape[1] == 0:
                raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
            
            return self.svm.predict(kernel_test)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise
    def getTrainableParameters(self):
        return "~"
