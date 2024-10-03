from Interfaces.iAnsatz import Ansatz
import pennylane as qml
import numpy as np

class TwoLocal(Ansatz):
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def getCircuit(self):
        def TwoLocal(theta, wires):
            """Implements a two-local ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            N=len(wires)
            for i in range(N):
                qml.RY(theta[i], wires = i)
            for i in range(N - 1):
                qml.CNOT(wires = [i, i + 1])
        return TwoLocal

    def getParameters(self):
        return  self.nqubits