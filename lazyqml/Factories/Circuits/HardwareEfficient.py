from Interfaces.iAnsatz import Ansatz
import pennylane as qml

class HardwareEfficient(Ansatz):
    def __init__(self, nqubits,nlayers):

        self.nqubits = nqubits
        self.nlayers = nlayers

    def getCircuit(self):
        def hardware_efficient_ansatz(theta, wires):
            """Implements a hardware-efficient ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """
            param_count = 0
            
            for _ in range(self.nlayers):

                N = len(wires)
                assert len(theta) == 3 * N * self.nlayers
                
                for i in range(N):
                    qml.RX(theta[param_count], wires=wires[i])
                    param_count +=1
                
                for i in range(N-1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                qml.CNOT(wires=[wires[N-1],wires[0]])
                
                for i in range(N):
                    qml.RZ(theta[param_count], wires=wires[i])
                    param_count +=1
                
                for i in range(N-1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                qml.CNOT(wires=[wires[N-1],wires[0]])

                for i in range(N):
                    qml.RX(theta[param_count], wires=wires[i])
                    param_count +=1
                
                for i in range(N-1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                qml.CNOT(wires=[wires[N-1],wires[0]])

        return hardware_efficient_ansatz
    
    def getParameters(self):
        return 3 * self.nqubits 