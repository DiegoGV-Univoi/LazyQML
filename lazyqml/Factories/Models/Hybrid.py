import torch
import pennylane as qml
from time import time
import numpy as np
from Factories.Models.QNNTorch import QNNTorch
from Interfaces.iAnsatz import Ansatz
from Interfaces.iCircuit import Circuit
from Factories.Circuits.fCircuits import *

class HybridModel(QNNTorch):
    def __init__(self, classicModel, output_shape, nqubits, backend, ansatz, embedding, n_class, layers, epochs, shots, lr, batch_size, seed=1234) -> None:
        super().__init__(nqubits, backend, ansatz, embedding, n_class, layers, epochs, shots, lr, batch_size, seed)

        self.classicModel = classicModel
        self.classicParameters = sum(p.numel() for p in self.classicModel.parameters() if p.requires_grad)

        self.bridgeLayer = torch.nn.Linear(output_shape, self.nqubits)

    def forward(self, x, theta):
        # Aqui lo que tengo que hacer es meterle la capa intermedia de fully connected para que cuadren los tamaños de vectores. Algo asi vvv
        x = self.classicModel(x)

        # x = torch.nn.Flatten()(x)
        x = self.bridgeLayer(x)

        qnn_output = self.qnn(x, theta)
        if self.n_class == 2:
            #return (qnn_output + 1) / 2
            return qnn_output.reshape(-1)
        else:
            # If qnn_output is a list, apply the transformation to each element
            #return torch.tensor([(output + 1) / 2 for output in qnn_output])
            return torch.stack([output for output in qnn_output]).T
    

output_shape = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(4, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, output_shape)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
testModel = TinyModel()