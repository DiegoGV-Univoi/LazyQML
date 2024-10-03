from abc import ABC, abstractmethod
from Interfaces.iCircuit import Circuit

class Ansatz(Circuit):
    @abstractmethod
    def getParameters(self):
        pass