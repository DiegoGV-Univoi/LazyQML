from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class PcaAmp(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits

        self.preprocessing = PCA(n_components=2**self.nqubits)

    def fit(X, y, self):
        return self.preprocessing.fit(X, y)

    def fit_transform(X, y, self):
        return self.preprocessing.fit(X, y)

    def transform(X, self):
        return self.preprocessing.fit(X)