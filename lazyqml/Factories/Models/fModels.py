from Factories.Models.QSVM import *
from Factories.Models.QNNBag import *
from Factories.Models.QNNTorch import *
from Global.globalEnums import *

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, Nqubits, Embedding, Ansatz, N_class,  Layers=5, Shots=1,
                 Max_samples=1.0, Max_features=1.0, LearningRate=0.01, 
                 BatchSize=8,  Epoch=50, seed=1234,backend=Backend.lightningQubit,numPredictors=10):
        
        if model == Model.QSVM:
            return QSVM(nqubits=Nqubits, embedding=Embedding, shots=Shots, seed=seed,backend=backend)
        elif model == Model.QNN:
            return QNNTorch(nqubits=Nqubits, ansatz=Ansatz, 
                        embedding=Embedding, n_class=N_class, 
                        layers=Layers, epochs=Epoch, shots=Shots, 
                        lr=LearningRate, batch_size=BatchSize, seed=seed,backend=backend)
        elif model == Model.QNN_BAG:
            return QNNBag(nqubits=Nqubits, ansatz=Ansatz, embedding=Embedding, 
                      n_class=N_class, layers=Layers, epochs=Epoch, 
                      max_samples=Max_samples, max_features=Max_features,
                      shots=Shots, lr=LearningRate, batch_size=BatchSize, seed=seed,backend=backend,n_estimators=numPredictors,n_features=Max_features)