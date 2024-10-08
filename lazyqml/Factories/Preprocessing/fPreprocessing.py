# Importing Enums
from Global.globalEnums import *

# Importing Preprocessings
from Factories.Preprocessing.Pca import *
from Factories.Preprocessing.PcaAmp import *
from Factories.Preprocessing.PcaTree import *
from Factories.Preprocessing.PcaTreeAmp import *
from Factories.Preprocessing.Sanitizer import *


class PreprocessingFactory:
    def __init__(self, nqubits) -> None:
        self.nqubits = nqubits

    def GetSanitizer(self, imputerCat, imputerNum):
        return Sanitizer(imputerCat, imputerNum)

    def GetPreprocessing(self, embedding, ansatz):
        if embedding == Embedding.AMP and ansatz == Ansatz.TREE_TENSOR:
            return PcaTreeAmp(self.nqubits)
        elif embedding == Embedding.AMP:
            return PcaAmp(self.nqubits)
        elif ansatz == Ansatz.TREE_TENSOR:
            return PcaTree(self.nqubits)
        else:
            return Pca(self.nqubits)
       
        """
        Deprecated

        # if prep == Preprocessing.PCA:
        #     return Pca(self.nqubits)
        # elif prep == Preprocessing.PCA_AMP:
        #     return PcaAmp(self.nqubits)
        # elif prep == Preprocessing.PCA_TREE:
        #     return PcaTree(self.nqubits)
        # elif prep == Preprocessing.PCA_TREE_AMP:
        #     return PcaTreeAmp(self.nqubits)
        """