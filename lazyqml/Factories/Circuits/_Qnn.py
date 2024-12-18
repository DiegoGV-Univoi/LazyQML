from lazyqml.Interfaces.iCircuit import Circuit
from Circuits.fCircuits import *

class QnnCircuit(Circuit):
    def __init__(self, nqubits, nlayers, embedding, ansatz) -> None:
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.embedding = embedding
        self.ansatz = ansatz

    def getCircuit(self):
        def qnn_circ(a):
            CircuitFactory.GetEmbeddingCircuit(self.embedding)(a, wires = range(self.nqubits))
            CircuitFactory.GetAnsatzCircuit(self.ansatz)(self.nqubits, self.nlayers)

        return qnn_circ
    
