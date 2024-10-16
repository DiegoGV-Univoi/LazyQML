from Global.globalEnums import *
from itertools import product
import numpy as np
import torch
import psutil

class VerbosePrinter:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VerbosePrinter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not VerbosePrinter._initialized:
            self.verbose = False
            VerbosePrinter._initialized = True

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def print(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}")
        else:
            pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VerbosePrinter()
        return cls._instance

def adjustQubits(nqubits, numClasses):
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2
    return int(nqubits)


def create_combinationsCV(classifiers, embeddings, ansatzs, features, qubits, FoldID, RepeatID):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []

    # Make sure we don't have duplicated items
    classifiers = list(classifiers)
    embeddings = list(embeddings)
    ansatzs = list(ansatzs)
    qubits = sorted(list(qubits))   # Convert the qubits set to a list as well
    FoldID = sorted(list(FoldID))
    RepeatID = sorted(list(RepeatID))

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
            # QSVM doesn't use ansatzs or features but uses qubits (first in the product)
            combinations.extend(list(product(qubits, [classifier], embedding_list, [None], [None], RepeatID, FoldID)))
        elif classifier == Model.QNN:
            # QNN uses ansatzs and qubits (qubits first)
            combinations.extend(list(product(qubits, [classifier], embedding_list, ansatzs_list, [None], RepeatID, FoldID)))
        elif classifier == Model.QNN_BAG:
            # QNN_BAG uses ansatzs, features, and qubits (qubits first)
            combinations.extend(list(product(qubits, [classifier], embedding_list, ansatzs_list, features, RepeatID, FoldID)))

    return combinations

def create_combinations(classifiers, embeddings, ansatzs, features, qubits):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []

    # Make sure we don't have duplicated items
    classifiers = list(classifiers)
    embeddings = list(embeddings)
    ansatzs = list(ansatzs)
    qubits = sorted(list(qubits))   # Convert the qubits set to a list as well

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
            # QSVM doesn't use ansatzs or features but uses qubits (first in the product)
            combinations.extend(list(product(qubits, [classifier], embedding_list, [None], [None])))
        elif classifier == Model.QNN:
            # QNN uses ansatzs and qubits (qubits first)
            combinations.extend(list(product(qubits, [classifier], embedding_list, ansatzs_list, [None])))
        elif classifier == Model.QNN_BAG:
            # QNN_BAG uses ansatzs, features, and qubits (qubits first)
            combinations.extend(list(product(qubits, [classifier], embedding_list, ansatzs_list, features)))

    return combinations

def fixSeed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def calculate_quantum_memory(num_qubits, overhead=2):
    # Each qubit state requires 2 complex numbers (amplitude and phase)
    # Each complex number uses 2 double-precision floats (16 bytes)
    bytes_per_qubit_state = 16

    # Number of possible states is 2^n, where n is the number of qubits
    num_states = 2 ** num_qubits

    # Total memory in bytes
    total_memory_bytes = num_states * bytes_per_qubit_state * overhead

    # Convert to more readable units

    return total_memory_bytes / (1024**3)

def calculate_free_memory():
        # Use psutil to get available system memory (in GB)
        mem = psutil.virtual_memory()
        free_ram_gb = mem.available / (1024 ** 3)  # Convert bytes to GB
        return free_ram_gb

printer = VerbosePrinter()
