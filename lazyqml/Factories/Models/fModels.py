class ModelFactory:
    def __init__(self, Nqubits, Layers) -> None:
        self.nqubits = Nqubits 
        self.layers = Layers

    def GetQSVM(self,ansatz, repetitions=1):
        pass

    def GetQNN(self, embedding, ansatz,layers):
        pass

    def GetQNNBag(self,embedding, ansatz,layers):
        pass
