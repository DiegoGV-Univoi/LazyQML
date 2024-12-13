from abc import ABC, abstractmethod

class Preprocessing(ABC):
    @abstractmethod
    def fit(X, y, self):
        pass
    
    @abstractmethod
    def fit_transform(X, y, self):
        pass

    @abstractmethod
    def transform(X, self):
        pass