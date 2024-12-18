from Interfaces.iCircuit import Circuit
import pennylane as qml

class HigherOrderEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):
        def higher_order_embedding(x, wires):
            """Embeds a quantum state into the quantum device using Higher Order Embedding.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            
            N = len(x)

            qml.AngleEmbedding(x, wires=wires, rotation='Y')

            for i in range(1, N):
                qml.CNOT(wires = [i - 1, i])
                qml.RY(x[i-1]*x[i], wires=i)
        
        return higher_order_embedding