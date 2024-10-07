from itertools import product
from Global.globalEnums import Ansatz, Embedding, Model

def create_combinations(classifiers, embeddings, ansatzs, features):

    combinations = []

    if Model.ALL in classifiers:
        classifier_list = Model.list()
        classifier_list.remove(Model.ALL)
    
    if Embedding.ALL in embeddings:
        embedding_list = Embedding.list()
        embedding_list.remove(Embedding.ALL)
    
    if Ansatz.ALL in ansatzs:
        ansatzs_list = Ansatz.list()
        ansatzs_list.remove(Ansatz.ALL)

    combinations = []
    for classifier in classifier_list:
        if classifier == Model.QSVM:
            combinations.extend(list(product(classifier, embedding_list, [None], [None])))
        elif classifier == Model.QNN:
            combinations.extend(list(product(classifier, embedding_list, ansatzs_list, [None])))
        elif classifier == Model.QNN_BAG:
            combinations.extend(list(product(classifier, embedding_list, ansatzs_list, features)))

    return combinations