from Interfaces.iCircuit import Circuit
import pennylane as qml

class AmplitudeEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()
        
    def getCircuit(self):
        def amp_embedding(x, wires):
            """Embeds a quantum state into the quantum device using Amplitude Encoding.

            Args:
                x (array[float]): array of complex amplitudes
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            qml.AmplitudeEmbedding(x, wires=wires, pad_with=0, normalize=True)
        return amp_embedding