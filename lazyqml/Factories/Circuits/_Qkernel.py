from lazyqml.Interfaces.iCircuit import Circuit
from lazyqml.Factories.Circuits.fCircuits import *
import pennylane as qml

class QkernelCircuit(Circuit):
    def __init__(self, nqubits, embedding) -> None:
        self.nqubits = nqubits
        self.embedding = embedding
        self.CircuitFactory = CircuitFactory()

    def getCircuit(self):
        def kernel_circ(a, b):
            self.CircuitFactory.GetEmbeddingCircuit(self.embedding)(a, wires = range(self.nqubits))
            qml.adjoint(self.CircuitFactory.GetEmbeddingCircuit(self.embedding))(b, wires = range(self.nqubits)))

        return kernel_circ