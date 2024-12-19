from abc import abstractmethod
from lazyqml.st.Interfaces.iCircuit import Circuit

class Ansatz(Circuit):
    @abstractmethod
    def getParameters(self):
        pass