# Importing Enums
from Global.globalEnums import Ansatz, Embedding
# Importing Ansatzs
from Factories.Circuits.TwoLocal import *
from Factories.Circuits.HardwareEfficient import *
from Factories.Circuits.TreeTensor import *
from Factories.Circuits.HCzRx import *
# Importing Embeddings
from Factories.Circuits.RxEmbedding import *
from Factories.Circuits.RyEmbedding import *
from Factories.Circuits.RzEmbedding import *
from Factories.Circuits.ZzEmbedding import *
from Factories.Circuits.AmplitudeEmbedding import *


class CircuitFactory:
    def __init__(self, Nqubits) -> None:
        self.nqubits = Nqubits 

    def GetAnsatzCircuit(self,ansatz):
        if ansatz == Ansatz.HARDWARE_EFFICIENT:
            return HardwareEfficient(self.nqubits)
        elif ansatz == Ansatz.HCZRX:
            return HCzRx(self.nqubits)
        elif ansatz == Ansatz.TREE_TENSOR:
            return TreeTensor(self.nqubits)
        elif ansatz == Ansatz.TWO_LOCAL:
            return TwoLocal(self.nqubits)

    def GetEmbeddingCircuit(self, embedding):
        if embedding == Embedding.RX:
            return RxEmbedding()
        elif embedding == Embedding.RY:
            return RyEmbedding()
        elif embedding == Embedding.RZ:
            return RzEmbedding()
        elif embedding == Embedding.ZZ:
            return ZzEmbedding()
        elif embedding == Embedding.AMP:
            return AmplitudeEmbedding()
    def GetKernelCircuit(self,embedding):
        pass
        
    
    def GetCircuit(self,embedding, ansatz):
        pass

