from Factories.Models.QSVM import *
from Factories.Models.QNNBag import *
from Factories.Models.QNNTorch import *
from Factories.Models.QNNPennylane import *
from Global.globalEnums import *

class ModelFactory:
    def __init__(self, Nqubits, Embedding, Ansatz, N_class,  Layers=5, Shots=1,
                 Max_samples=1.0, Max_features=1.0, LearningRate=0.01, 
                 BatchSize=8,  Epoch=50, seed=1234,backend=Backend.lightningQubit) -> None:
        
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
        self.seed = seed
        self.backend = backend

    def GetQSVM(self):

        return QSVM(nqubits=self.nqubits, embedding=self.embedding, shots=self.shots, seed=self.seed,backend=self.backend)
        

    def GetQNN(self):

        return QNNTorch(nqubits=self.nqubits, ansatz=self.ansatz, 
                        embedding=self.embedding, n_class=self.n_class, 
                        layers=self.layers, epochs=self.epoch, shots=self.shots, 
                        lr=self.lr, batch_size=self.batch, seed=self.seed,backend=self.backend)

    def GetQNNBag(self):

        return QNNBag(nqubits=self.nqubits, ansatz=self.ansatz, embedding=self.embedding, 
                      n_class=self.n_class, layers=self.layers, epochs=self.epoch, 
                      max_samples=self.max_samples, max_features=self.max_features,
                      shots=self.shots, lr=self.lr, batch_size=self.batch, seed=self.seed,backend=self.backend)
