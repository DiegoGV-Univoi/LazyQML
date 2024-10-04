from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class PcaTree(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits

        self.preprocessing = PCA(n_components=2**(self.nqubits.bit_length()-1))

    def fit(X, y, self):
        return self.preprocessing.fit(X, y)

    def fit_transform(X, y, self):
        return self.preprocessing.fit(X, y)

    def transform(X, self):
        return self.preprocessing.fit(X)