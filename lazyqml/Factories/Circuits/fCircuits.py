# Importing Enums
from lazyqml.Global.globalEnums import Ansatzs, Embedding
# Importing Ansatzs
from lazyqml.Factories.Circuits.TwoLocal import *
from lazyqml.Factories.Circuits.HardwareEfficient import *
from lazyqml.Factories.Circuits.TreeTensor import *
from lazyqml.Factories.Circuits.HCzRx import *
# Importing Embeddings
from lazyqml.Factories.Circuits.RxEmbedding import *
from lazyqml.Factories.Circuits.RyEmbedding import *
from lazyqml.Factories.Circuits.RzEmbedding import *
from lazyqml.Factories.Circuits.ZzEmbedding import *
from lazyqml.Factories.Circuits.AmplitudeEmbedding import *


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

