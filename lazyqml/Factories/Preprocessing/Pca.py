from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class Pca(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.preprocessing = PCA(n_components=self.nqubits)

    def fit(self, X, y):
        return self.preprocessing.fit(X, y)

    def fit_transform(self, X, y):
        return self.preprocessing.fit_transform(X, y)

    def transform(self, X):
        return self.preprocessing.transform(X)