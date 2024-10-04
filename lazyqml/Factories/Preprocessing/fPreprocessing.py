# Importing Enums
from Global.globalEnums import Preprocessing

# Importing Preprocessings
from Preprocessing.Pca import *
from Preprocessing.PcaAmp import *
from Preprocessing.PcaTree import *
from Preprocessing.PcaTreeAmp import *
from Preprocessing.Sanitizer import *


class PreprocessingFactory:
    def __init__(self, nqubits) -> None:
        self.nqubits = nqubits

    def GetPreprocessingTypes(self, imputerCat, imputerNum):
        return Sanitizer(imputerCat, imputerNum)

    def GetPreprocessing(self, prep):
        if prep == Preprocessing.PCA:
            return Pca(self.nqubits)
        elif prep == Preprocessing.PCA_AMP:
            return PcaAmp(self.nqubits)
        elif prep == Preprocessing.PCA_TREE:
            return PcaTree(self.nqubits)
        elif prep == Preprocessing.PCA_TREE_AMP:
            return PcaTreeAmp(self.nqubits)