from Interfaces.iAnsatz import Ansatz
import pennylane as qml

class HardwareEfficient(Ansatz):
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def getCircuit():
        def hardware_efficient_ansatz(theta, wires):
            """Implements a hardware-efficient ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            N = len(wires)
            assert len(theta) == 3 * N
            
            for i in range(N):
                qml.RX(theta[3 * i], wires=wires[i])
            
            for i in range(N-1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.CNOT(wires=[wires[N-1],wires[0]])
            
            for i in range(N):
                qml.RZ(theta[3 * i + 1], wires=wires[i])
            
            for i in range(N-1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.CNOT(wires=[wires[N-1],wires[0]])

            for i in range(N):
                qml.RX(theta[3 * i + 2], wires=wires[i])
            
            for i in range(N-1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.CNOT(wires=[wires[N-1],wires[0]])
        return hardware_efficient_ansatz

    def getParameters(self):
        return 3 * self.nqubits