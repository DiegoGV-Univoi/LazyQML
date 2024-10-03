# Importing Enums
from Global.globalEnums import Ansatz, Embedding
# Importing Ansatzs
from Circuits import HardwareEfficient, HCzRx, TreeTensor, TwoLocal
# Importing Embeddings
from Circuits import RxEmbedding, RyEmbedding, RzEmbedding, ZzEmbedding

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

    def GetKernelCircuit(self,embedding):
        pass
        
    
    def GetCircuit(self,embedding, ansatz):
        pass

