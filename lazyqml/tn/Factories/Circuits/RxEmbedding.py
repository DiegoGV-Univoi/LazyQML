from Interfaces.iCircuit import Circuit
import pennylane as qml

class RxEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):
        def rx_embedding(x, wires):
            """Embeds a quantum state into the quantum device using rotation around the X-axis.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            qml.AngleEmbedding(x, wires=wires, rotation='X')
        return rx_embedding