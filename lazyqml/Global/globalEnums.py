"""
------------------------------------------------------------------------------------------------------------------
    This file is devoted to define the global enums for easening the interface.
------------------------------------------------------------------------------------------------------------------
"""

from enum import Enum

class BaseEnum(Enum):
    @classmethod
    def list(cls):
        return list(cls)

# Enum for selecting the Ansatz circuits
class Ansatzs(BaseEnum):
    ALL = 1
    HCZRX = 2
    TREE_TENSOR = 3
    TWO_LOCAL = 4
    HARDWARE_EFFICIENT = 5

# Enum for selecting the Embedding circuits
class Embedding(BaseEnum):
    ALL = 1
    RX = 2
    RZ = 3
    RY = 4
    ZZ = 5
    AMP = 6

# Enum for selecting the Models
class Model(BaseEnum):
    ALL = 1
    QNN = 2
    QNN_BAG = 3
    QSVM = 4

class Backend(Enum):
    defaultQubit = "default.qubit"
    lightningQubit = "lightning.qubit"
    lightningGPU = "lightning.gpu"