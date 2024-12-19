from lazyqml.Interfaces.iAnsatz import Ansatz
import pennylane as qml
import numpy as np

class Annular(Ansatz):
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers

    def getCircuit(self):
        def annular(theta, wires):
            """Implements an annular ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """

            N=len(wires)

            param_count = 0

            for _ in range(self.nlayers):
                for i in range(N):
                    qml.X(wires=i)
                    qml.Hadamard(wires=i)

                for i in range(N - 1):
                    qml.CNOT(wires = [i, i + 1])
                    qml.RY(theta[param_count], wires = i+1)

                    param_count += 1

                qml.CNOT(wires=[N-1, 0])
                qml.RY(theta[param_count], wires = 0)
                param_count += 1    #just in case

        return annular

    def getParameters(self):
        return self.nqubits 