"""
This file is devoted to define the global enums for easening the interface.
"""

from enum import Enum

# Enum for selecting the Ansatz circuits
class Ansatz(Enum):
    ALL = 1
    HCZRX = 2
    TREE_TENSOR = 3
    TWO_LOCAL = 4
    HARDWARE_EFFICIENT = 5

# Enum for selecting the Embedding circuits
class Embedding(Enum):
    ALL = 1
    RX = 2
    RZ = 3
    RY = 4
    ZZ = 5
    AMP = 6

# Enum for selecting the Models
class Model(Enum):
    ALL = 1
    QNN = 2
    QNN_BAG = 3
    QSVM = 4

# Enum for selecting the Preprocessings
class Preprocessing(Enum):
    PCA = 1
    PCA_AMP = 2
    PCA_TREE = 3
    PCA_TREE_AMP = 4
    TYPES = 5

class Backend(Enum):
    defaultQubit = "default.qubit"
    lightningQubit = "lightning.qubit"
    lightningGPU = "lightning.gpu"