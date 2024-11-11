from lazyqml.Interfaces.iAnsatz import Ansatz
import pennylane as qml

class HCzRx(Ansatz):
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers

    def getCircuit(self):
        def HCzRx(theta, wires):
            """Implements an ansatz circuit composed of Hadamard, CZ, and RX gates.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            N = len(wires)

            param_count = 0

            for _ in range(int(self.nlayers)):
                for i in range(N):
                    qml.Hadamard(wires = wires[i])
                
                for i in range(N-1):
                    qml.CZ(wires=[wires[i], wires[i+1]])
                qml.CZ(wires=[wires[N-1],wires[0]])
                
                for i in range(N):
                    qml.RX(theta[param_count], wires=wires[i])
                    param_count += 1

        return HCzRx
    
    def getParameters(self):
        return self.nqubits 