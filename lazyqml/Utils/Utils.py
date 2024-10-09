from Global.globalEnums import *
from itertools import product
import numpy as np
import torch

def adjustQubits(nqubits, numClasses):
    adjustedQubits = nqubits
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2
    return adjustedQubits

def create_combinations(classifiers, embeddings, ansatzs, features):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []

    if Model.ALL in classifiers:
        classifier_list = Model.list()
        classifier_list.remove(Model.ALL)
    else:
        classifier_list = classifiers

    if Embedding.ALL in embeddings:
        embedding_list = Embedding.list()
        embedding_list.remove(Embedding.ALL)
    else:
        embedding_list = embeddings

    if Ansatzs.ALL in ansatzs:
        ansatzs_list = Ansatzs.list()
        ansatzs_list.remove(Ansatzs.ALL)
    else: 
        ansatzs_list = ansatzs

    combinations = []

    for classifier in classifier_list:
        if classifier == Model.QSVM:
            combinations.extend(list(product([classifier], embedding_list, [None], [None])))
        elif classifier == Model.QNN:
            combinations.extend(list(product([classifier], embedding_list, ansatzs_list, [None])))
        elif classifier == Model.QNN_BAG:
            combinations.extend(list(product([classifier], embedding_list, ansatzs_list, features)))

    return combinations

def create_enum_combinations(classifiers, embeddings, ansatzs, features):
    combinations = []
    
    # Check for "ALL" in the classifiers, embeddings, and ansatzs
    if Model.ALL in classifiers:
        classifiers = [Model.QSVM, Model.QNN, Model.QNN_BAG]
    if Embedding.ALL in embeddings:
        embeddings = [Embedding.RX, Embedding.RZ, Embedding.RY, Embedding.ZZ, Embedding.AMP]
    if Ansatzs.ALL in ansatzs:
        ansatzs = [Ansatzs.HCZRX, Ansatzs.TREE_TENSOR, Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT, None]

    # Generate combinations with the constraints
    for classifier, embedding, ansatz, feature in product(classifiers, embeddings, ansatzs, features + [None]):
        if classifier == Model.QSVM:
            # For "QSVM", ansatz and feature should be None
            if ansatz is None and feature is None:
                combinations.append((classifier, embedding, ansatz, feature))
        elif classifier == Model.QNN:
            # For "QNN", feature should be None and ansatz cannot be None
            if feature is None and ansatz is not None:
                combinations.append((classifier, embedding, ansatz, feature))
        elif classifier == Model.QNN_BAG:
            # For "QNN_BAG", ansatz and feature cannot be None
            if embedding is not None and ansatz is not None and feature is not None:
                combinations.append((classifier, embedding, ansatz, feature))
                
    return combinations

def fixSeed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)


# if __name__ == "__main__":
#     from Global.globalEnums import *
#     from Utils.Utils import create_combinations

#     classifiers = [Model.QNN, Model.QSVM]
#     embeddings = [Embedding.RX]
#     ansatzs = [Ansatzs.TWO_LOCAL]
#     features = [0.3]

#     print(create_combinations(classifiers, embeddings, ansatzs, features)) 