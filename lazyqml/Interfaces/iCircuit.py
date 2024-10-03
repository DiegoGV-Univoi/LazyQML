from abc import ABC, abstractmethod

class Circuit(ABC):
    @abstractmethod
    def getCircuit(self):
        pass