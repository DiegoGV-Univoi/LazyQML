class CircuitFactory:
    def __init__(self, Nqubits) -> None:
        self.nqubits = Nqubits 

    def GetAnsatzCircuit(self,ansatz, repetitions=1):
        pass

    def GetEmbeddingCircuit(self, embedding):
        pass

    def GetKernelCircuit(self,embedding):
        pass
