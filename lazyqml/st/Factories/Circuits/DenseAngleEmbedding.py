from lazyqml.Interfaces.iCircuit import Circuit
import pennylane as qml

class DenseAngleEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):
        def dense_angle_embedding(x, wires):
            """Embeds a quantum state into the quantum device using rotation around the Y-axis and global phase.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            half_len = len(x)//2
            N = len(wires)

            qml.AngleEmbedding(x[:half_len], wires=wires, rotation='Y')

            for i, e in enumerate(x[half_len:]):
                qml.PhaseShift(e, wires=i)
                
        return dense_angle_embedding