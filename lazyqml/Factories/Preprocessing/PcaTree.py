from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class PcaTree(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits

        self.preprocessing = PCA(n_components=2**(self.nqubits.bit_length()-1))

    def fit(self, X, y):
        return self.preprocessing.fit(X, y)

    def fit_transform(self, X, y):
        return self.preprocessing.fit_transform(X, y)

    def transform(self, X):
        return self.preprocessing.tranform(X)