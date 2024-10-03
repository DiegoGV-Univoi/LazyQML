from lazyqml.Interfaces.iCircuit import Circuit
from Circuits.fCircuits import *
import pennylane as qml

class QkernelCircuit(Circuit):
    def __init__(self, nqubits, embedding) -> None:

        self.nqubits = nqubits
        self.embedding = embedding

    def getCircuit(self):

        def kernel_circ(a, b):
            EmbeddingFactory.GetEmbeddingCircuit(self.embedding)(a, wires = range(self.nqubits))
            qml.adjoint(EmbeddingFactory.GetEmbeddingCircuit(self.embedding))(b, wires = range(self.nqubits)))

        return kernel_circ