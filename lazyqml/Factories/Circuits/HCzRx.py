from Interfaces.iAnsatz import Ansatz
import pennylane as qml

class HCzRx(Ansatz):
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def getCircuit():
        def HCzRx(theta, wires):
            """Implements an ansatz circuit composed of Hadamard, CZ, and RX gates.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            N=len(wires)

            for i in range(N):
                qml.Hadamard(wires = wires[i])
            
            for i in range(N-1):
                qml.CZ(wires=[wires[i], wires[i+1]])
            qml.CZ(wires=[wires[N-1],wires[0]])
            
            for i in range(N):
                qml.RX(theta[i], wires=wires[i])
        return HCzRx
    def getParameters(self):
        return self.nqubits