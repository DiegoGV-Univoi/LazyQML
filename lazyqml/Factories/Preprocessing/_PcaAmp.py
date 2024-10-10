from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class PcaAmp(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.preprocessing = None

    def fit(self, X, y):
        self.preprocessing = PCA(n_components=self.nqubits if X.shape[1] < 2**self.nqubits else 2**self.nqubits)
        return self.preprocessing.fit(X, y)

    def fit_transform(self, X, y):
        self.preprocessing = PCA(n_components=self.nqubits if X.shape[1] < 2**self.nqubits else 2**self.nqubits)
        return self.preprocessing.fit_transform(X, y)

    def transform(self, X):
        return self.preprocessing.transform(X)