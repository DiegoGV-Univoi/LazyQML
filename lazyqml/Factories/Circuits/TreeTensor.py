from Interfaces.iAnsatz import Ansatz
import pennylane as qml
import numpy as np

class TreeTensor(Ansatz):
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def getCircuit(self):
        def tree_tensor_ansatz(theta , wires):
            """Implements a tree tensor network ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            n = len(self.wires)
            dim = int(np.log2(n))
            param_count = 0
            for i in range (dim+1):
                step = 2**i
                for j in np.arange(0 , n , 2*step):
                    qml.RY(theta[param_count] , wires = j)
                    if(i<dim):
                        qml.RY(theta[param_count + 1] , wires = j + step)
                        qml.CNOT(wires = [j , j + step])
                    param_count += 2
        return tree_tensor_ansatz

    def getParameters(self):
        return 2 ** (self.nqubits - 1) - 1