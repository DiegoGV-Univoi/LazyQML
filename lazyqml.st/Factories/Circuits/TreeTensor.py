from Interfaces.iAnsatz import Ansatz
import pennylane as qml
import numpy as np

class TreeTensor(Ansatz):
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers

    def getCircuit(self):
        def tree_tensor_ansatz(theta, wires):
            """Implements a tree tensor network ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            N = len(wires)

            dim = int(np.log2(N))

            param_count = 0

            for nl in range(self.nlayers):
                for i in range (dim+1):
                    step = 2**i
                    for j in np.arange(0 , N , 2*step):
                        qml.RY(theta[param_count] , wires = j)
                        param_count += 1
                        if(i<dim):
                            qml.RY(theta[param_count] , wires = j + step)
                            param_count +=1
                            qml.CNOT(wires = [j , j + step])

        return tree_tensor_ansatz

    def getParameters(self):
        return (2 ** (int(np.log2(self.nqubits)) + 1) - 1) 