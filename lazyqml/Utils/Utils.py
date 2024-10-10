from Global.globalEnums import *
from itertools import product
import numpy as np
import torch

def adjustQubits(nqubits, numClasses):
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2
    return int(nqubits)

def create_combinations(classifiers, embeddings, ansatzs, features):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []

    # Make sure don't have duplicated items
    classifiers = list(classifiers)
    embeddings = list(embeddings)
    ansatzs = list(ansatzs)

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

def fixSeed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
