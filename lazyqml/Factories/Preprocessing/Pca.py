from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

class Pca(Preprocessing):
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.ncomponents = nqubits
        self.preprocessing = PCA(n_components=self.ncomponents)

    def fit(self, X, y):
        return self.preprocessing.fit(X, y) if self.ncomponents <= X.shape[1] else X

    def fit_transform(self, X, y):
        return self.preprocessing.fit_transform(X, y) if self.ncomponents <= X.shape[1] else X

    def transform(self, X):
        try:
            fitted_X = self.preprocessing.transform(X)
        except:
            fitted_X = X
        return fitted_X
