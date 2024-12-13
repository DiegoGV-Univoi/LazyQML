from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(X, y, self):
        pass
    
    @abstractmethod
    def predict(X, self):
        pass