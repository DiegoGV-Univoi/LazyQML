# Importing Enums
from lazyqml.Global.globalEnums import *

# Importing Preprocessings
from lazyqml.Factories.Preprocessing.Pca import *
# from Factories.Preprocessing.PcaAmp import *
# from Factories.Preprocessing.PcaTree import *
# from Factories.Preprocessing.PcaTreeAmp import *
from lazyqml.Factories.Preprocessing.Sanitizer import *


class PreprocessingFactory:
    def __init__(self, nqubits) -> None:
        self.nqubits = nqubits

    def GetSanitizer(self, imputerCat, imputerNum):
        return Sanitizer(imputerCat, imputerNum)

    def GetPreprocessing(self, embedding, ansatz):
        if embedding == Embedding.AMP and ansatz == Ansatzs.TREE_TENSOR:
            return Pca(self.nqubits, 2**(2**(self.nqubits.bit_length()-1)))
        elif embedding == Embedding.AMP:
            return Pca(self.nqubits, 2**self.nqubits)
        elif ansatz == Ansatzs.TREE_TENSOR:
            return Pca(self.nqubits, 2**(self.nqubits.bit_length()-1))
        else:
            return Pca(self.nqubits, self.nqubits)

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
