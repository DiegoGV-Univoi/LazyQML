from lazyqml.Interfaces.iCircuit import Circuit
import pennylane as qml
import numpy as np
from itertools import combinations

class ZzEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):  
        def ZZ_embedding(x,wires):
            """Embeds a quantum state into the quantum device using ZZ-rotation.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """    
            nload=min(len(x), len(wires))
            
            for i in range(nload):
                qml.Hadamard(i)
                qml.RZ(2.0*x[i],wires=i)
                

            for pair in list(combinations (range(nload), 2)):
                q0=pair[0]
                q1=pair[1]
                
                qml.CZ(wires=[q0,q1])
                qml.RZ(2.0*(np.pi-x[q0])*(np.pi-x[q1]),wires=q1)
                qml.CZ(wires=[q0,q1])

        return ZZ_embedding