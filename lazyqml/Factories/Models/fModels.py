from Models.QSVM import *
from Models.QNNBag import *
from Models.QNNTorch import *
from Models.QNNPennylane import *

class ModelFactory:
    def __init__(self, Nqubits, Layers, Embedding, Ansatz, Shots, N_class, 
                 Max_samples, Max_features, LearningRate=0.01, 
                 BatchSize=8,  Epoch=50) -> None:
        
        self.nqubits = Nqubits 
        self.layers = Layers
        self.embedding = Embedding
        self.ansatz = Ansatz
        self.shots = Shots
        self.lr = LearningRate
        self.batch = BatchSize
        self.n_class = N_class
        self.epoch = Epoch
        self.max_samples = Max_samples
        self.max_features = Max_features

    def GetQSVM(self):

        return QSVM(nqubits=self.nqubits, embedding=self.embedding, shots=self.shots)
        

    def GetQNN(self):

        return QNNTorch(nqubits=self.nqubits, ansatz=self.ansatz, 
                        embedding=self.embedding, n_class=self.n_class, 
                        layers=self.layers, epochs=self.epoch, shots=self.shots, 
                        lr=self.lr, batch_size=self.batch)

    def GetQNNBag(self):

        return QNNBag(nqubits=self.nqubits, ansatz=self.ansatz, embedding=self.embedding, 
                      n_class=self.n_class, layers=self.layers, epochs=self.epoch, 
                      max_samples=self.max_samples, max_features=self.max_features,
                      shots=self.shots, lr=self.lr, batch_size=self.batch)
