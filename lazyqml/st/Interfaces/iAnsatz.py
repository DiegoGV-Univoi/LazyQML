from abc import abstractmethod
from lazyqml.Interfaces.iCircuit import Circuit

class Ansatz(Circuit):
    @abstractmethod
    def getParameters(self):
        pass