from Interfaces.iCircuit import Circuit
import pennylane as qml

class RzEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):  
        def rz_embedding(x, wires):
            """Embeds a quantum state into the quantum device using rotation around the Z-axis.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            for i in wires:
                qml.Hadamard(wires=i)
            qml.AngleEmbedding(x, wires=wires, rotation='Z')
        return rz_embedding