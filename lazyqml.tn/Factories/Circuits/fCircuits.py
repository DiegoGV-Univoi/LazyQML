# Importing Enums
from Global.globalEnums import Ansatzs, Embedding
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
    def __init__(self, Nqubits,nlayers) -> None:
        self.nqubits = Nqubits 
        self.nlayers = nlayers

    def GetAnsatzCircuit(self,ansatz):
        if ansatz == Ansatzs.HARDWARE_EFFICIENT:
            return HardwareEfficient(self.nqubits,self.nlayers)
        elif ansatz == Ansatzs.HCZRX:
            return HCzRx(self.nqubits,self.nlayers)
        elif ansatz == Ansatzs.TREE_TENSOR:
            return TreeTensor(self.nqubits, nlayers=self.nlayers)
        elif ansatz == Ansatzs.TWO_LOCAL:
            return TwoLocal(self.nqubits, nlayers=self.nlayers)

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

