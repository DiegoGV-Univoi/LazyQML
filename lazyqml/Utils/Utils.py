# Imports
import pandas as pd
import numpy as np
import torch
import psutil
import GPUtil

# Importing from
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from Global.globalEnums import *
from itertools import product

"""
------------------------------------------------------------------------------------------------------------------
    Verbose printer class
        - This class implements the functionlity to print or not depending on a boolean flag
        - The message is preceded by "[VERBOSE] {message}" 
        - It is implemented as a Singleton Object
------------------------------------------------------------------------------------------------------------------
"""
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
    
"""
------------------------------------------------------------------------------------------------------------------
                                            Miscelaneous Utils
------------------------------------------------------------------------------------------------------------------
"""

def adjustQubits(nqubits, numClasses):
    """
        Adjust the number of qubits to be able to solve the problem
    """
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2
    return int(nqubits)

def calculate_quantum_memory(num_qubits, overhead=2):
    """
        Estimates the memory in MiB used by the quantum circuits.
    """
    # Each qubit state requires 2 complex numbers (amplitude and phase)
    # Each complex number uses 2 double-precision floats (16 bytes)
    bytes_per_qubit_state = 16

    # Number of possible states is 2^n, where n is the number of qubits
    num_states = 2 ** num_qubits

    # Total memory in bytes
    total_memory_bytes = num_states * bytes_per_qubit_state * overhead

    # Convert to more readable units

    return total_memory_bytes / (1024**2)

def calculate_free_memory():
    """
        Calculates the amount of free RAM
    """
    # Use psutil to get available system memory (in MiB)
    mem = psutil.virtual_memory()
    free_ram_mb = mem.available / (1024 ** 2)  # Convert bytes to MiB
    return free_ram_mb

def calculate_free_video_memory():
    """
        Calculates the amount of free Video Memory
    """
    # Use psutil to get available system memory (in MiB)
    return GPUtil.getGPUs()[0].memoryFree


def create_combinations(classifiers, embeddings, ansatzs, features, qubits, FoldID, RepeatID):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []
    
    # Make sure we don't have duplicated items
    classifiers = list(classifiers)
    embeddings = list(embeddings)
    ansatzs = list(ansatzs)
    qubit_values = sorted(list(qubits))
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
    # Create all base combinations first
    for qubits in qubit_values:
        for classifier in classifier_list:
            temp_combinations = []
            if classifier == Model.QSVM:
                # QSVM doesn't use ansatzs or features but uses qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, [None], [None], RepeatID, FoldID))
            elif classifier == Model.QNN:
                # QNN uses ansatzs and qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, ansatzs_list, [None], RepeatID, FoldID))
            elif classifier == Model.QNN_BAG:
                # QNN_BAG uses ansatzs, features, and qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, ansatzs_list, features, RepeatID, FoldID))
            
            # Add memory calculation for each combination
            for combo in temp_combinations:
                memory = calculate_quantum_memory(combo[0])  # Calculate memory based on number of qubits
                combinations.append(combo + (memory,))
    
    return combinations

def fixSeed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)


def generate_cv_indices(X, y, mode="cross-validation", test_size=0.4, n_splits=5, n_repeats=1, random_state=None):
    """
    Generate train and test indices for either cross-validation, holdout split, or leave-one-out.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): The features matrix.
        y (pd.Series or np.ndarray): The target vector.
        mode (str): "cross-validation", "holdout", or "leave-one-out".
        test_size (float): Test set proportion for the holdout split (ignored for CV and LOO).
        n_splits (int): Number of folds in StratifiedKFold (ignored for holdout and LOO).
        n_repeats (int): Number of repeats for cross-validation (ignored for holdout and LOO).
        random_state (int): Random state for reproducibility.
    
    Returns:
        dict: A dictionary of train/test indices.
    """
    cv_indices = {}
    
    if mode == "holdout":
        # Single train-test split for holdout
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        cv_indices[(0, 0)] = {
            'train_idx': train_idx,
            'test_idx': test_idx
        }
    
    elif mode == "cross-validation":
        # StratifiedKFold for cross-validation splits
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat if random_state is not None else None)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                cv_indices[(repeat, fold)] = {
                    'train_idx': train_idx,
                    'test_idx': test_idx
                }
    
    elif mode == "leave-one-out":
        # LeaveOneOut cross-validation
        loo = LeaveOneOut()
        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            cv_indices[(0, fold)] = {
                'train_idx': train_idx,
                'test_idx': test_idx
            }
    
    else:
        raise ValueError("Invalid mode. Choose 'holdout', 'cross-validation', or 'leave-one-out'.")
    
    return cv_indices

def get_train_test_split(cv_indices, repeat_id=0, fold_id=0):
    """
    Retrieve the train and test indices for a given repeat and fold ID.
    
    Parameters:
        cv_indices (dict): The cross-validation indices dictionary.
        repeat_id (int): The repeat ID (0 to n_repeats-1 or 0 for holdout/LOO).
        fold_id (int): The fold ID within the specified repeat.
    
    Returns:
        tuple: (train_idx, test_idx) arrays for the specified fold and repeat.
    """
    indices = cv_indices.get((repeat_id, fold_id))
    if indices is None:
        print(f"RepeatID {repeat_id}, FoldID{fold_id}")
        raise ValueError("Invalid repeat_id or fold_id specified.")
    
    return indices['train_idx'], indices['test_idx']



def dataProcessing(X, y, prepFactory, customImputerCat, customImputerNum, 
                train_idx, test_idx, ansatz=None, embedding=None):
    """
    Process data for specific train/test indices.
    
    Parameters:
    - X: Input features
    - y: Target variable
    - prepFactory: Preprocessing factory object
    - customImputerCat: Categorical imputer
    - customImputerNum: Numerical imputer
    - train_idx: Training set indices
    - test_idx: Test set indices
    - ansatz: Optional preprocessing ansatz
    - embedding: Optional embedding method
    
    Returns:
    Tuple of (X_train_processed, X_test_processed, y_train, y_test)
    """
    # Split the data using provided indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create sanitizer and preprocess
    sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)
    X_train = pd.DataFrame(sanitizer.fit_transform(X_train))
    X_test = pd.DataFrame(sanitizer.transform(X_test))
    
    # Apply additional preprocessing if ansatz/embedding provided
    if ansatz is not None or embedding is not None:
        preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
        X_train_processed = np.array(preprocessing.fit_transform(X_train, y=y_train))
        X_test_processed = np.array(preprocessing.transform(X_test))
    else:
        X_train_processed = np.array(X_train)
        X_test_processed = np.array(X_test)
    
    # Convert target variables to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train_processed, X_test_processed, y_train, y_test

printer = VerbosePrinter()