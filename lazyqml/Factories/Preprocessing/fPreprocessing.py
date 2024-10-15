# Importing Enums
from Global.globalEnums import *

# Importing Preprocessings
from Factories.Preprocessing.Pca import *
from Factories.Preprocessing.Sanitizer import *

from Factories.Preprocessing._PcaAmp import *
from Factories.Preprocessing._PcaTreeAmp import *
from Factories.Preprocessing._PcaTree import *

class PreprocessingFactory:
    def __init__(self, nqubits) -> None:
        self.nqubits = nqubits

    def GetSanitizer(self, imputerCat, imputerNum):
        return Sanitizer(imputerCat, imputerNum)

    def GetPreprocessing(self, embedding, ansatz):
        if embedding == Embedding.AMP and ansatz == Ansatzs.TREE_TENSOR:
            return PcaTreeAmp(self.nqubits)
        elif embedding == Embedding.AMP:
            return PcaAmp(self.nqubits)
        elif ansatz == Ansatzs.TREE_TENSOR:
            return PcaTree(self.nqubits)
        else:
            return Pca(self.nqubits)
