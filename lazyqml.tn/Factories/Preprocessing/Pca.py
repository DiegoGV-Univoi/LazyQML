# Importing from
from Interfaces.iPreprocessing import Preprocessing
from sklearn.decomposition import PCA

class Pca(Preprocessing):
    def __init__(self, nqubits, ncomponents):
        self.nqubits = nqubits
        self.ncomponents = ncomponents
        self.fitted = False
        self.preprocessing = PCA(n_components=self.ncomponents)

    def fit(self, X, y):
        if self.ncomponents <= X.shape[1]:
            self.fitted = True
            fitted_X = self.preprocessing.fit(X, y)
        else:
            fitted_X = X
        return fitted_X

    def fit_transform(self, X, y):
        if self.ncomponents <= X.shape[1]:
            self.fitted = True
            fitted_X = self.preprocessing.fit_transform(X, y)
        else:
            fitted_X = X
        return fitted_X

    def transform(self, X):
        if self.fitted:
            fitted_X = self.preprocessing.transform(X)
        else:
            fitted_X = X
        return fitted_X
