from abc import ABC, abstractmethod
from iCircuit import Circuit
class Ansatz(Circuit):
    @abstractmethod
    def getParameters(self):
        pass