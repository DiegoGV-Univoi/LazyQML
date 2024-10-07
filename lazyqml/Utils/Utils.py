from Global.globalEnums import *
import numpy as np

def adjustQubits(nqubits, numClasses):
    adjustedQubits = nqubits
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2

    return adjustedQubits