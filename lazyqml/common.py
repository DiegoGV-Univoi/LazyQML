#======================================================================================
# The common module contains common functions and classes used by the other modules.
#======================================================================================

"""
 Import Packages
"""
import math
import logging
import pennylane as qml
import jax.numpy as jnp
import optax
from itertools import combinations
import pennylane as qml
import jax
import numpy as np
import pandas as pd
import pennylane as qml
import inspect
import os
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import datetime
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from itertools import product
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)


#jax.config.update("jax_disable_jit",True)
"""
 Embeddings / Feature Maps
"""

def rx_embedding(x, wires):
    """Embeds a quantum state into the quantum device using rotation around the X-axis.

    Args:
        x (array[float]): array of rotation angles for each qubit
        wires (Sequence[int]): wires that the operation acts on

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation='X')

def ry_embedding(x, wires):
    """Embeds a quantum state into the quantum device using rotation around the Y-axis.

    Args:
        x (array[float]): array of rotation angles for each qubit
        wires (Sequence[int]): wires that the operation acts on

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation='Y')

def rz_embedding(x, wires):
    """Embeds a quantum state into the quantum device using rotation around the Z-axis.

    Args:
        x (array[float]): array of rotation angles for each qubit
        wires (Sequence[int]): wires that the operation acts on

    Returns:
        None
    """
    for i in wires:
        qml.Hadamard(wires=i)
    qml.AngleEmbedding(x, wires=wires, rotation='Z')

def ZZ_embedding(x,wires):
    """Embeds a quantum state into the quantum device using ZZ-rotation.

    Args:
        x (array[float]): array of rotation angles for each qubit
        wires (Sequence[int]): wires that the operation acts on

    Returns:
        None
    """    
    nload=min(len(x), len(wires))
    #nload=len(wires)
        
    
    for i in range(nload):
        qml.Hadamard(i)
        qml.RZ(2.0*x[i],wires=i)
        

    for pair in list(combinations (range(nload), 2)):
        q0=pair[0]
        q1=pair[1]
        
        qml.CZ(wires=[q0,q1])
        qml.RZ(2.0*(jnp.pi-x[q0])*(jnp.pi-x[q1]),wires=q1)
        qml.CZ(wires=[q0,q1])

def amp_embedding (x , wires):
    """Embeds a quantum state into the quantum device using Amplitude Encoding.

    Args:
        x (array[float]): array of complex amplitudes
        wires (Sequence[int]): wires that the operation acts on

    Returns:
        None
    """
    qml.AmplitudeEmbedding(x , wires , pad_with = 0 , normalize = True)

def get_embedding(embedd):
    """Returns the embedding function based on the specified embedding type.

    Args:
        embedd (str): embedding type ('rx_embedding', 'ry_embedding', 'rz_embedding', 'ZZ_embedding', or 'amplitude_embedding')

    Returns:
        function: embedding function corresponding to the specified type
    """
    if embedd == 'rx_embedding':
        return rx_embedding
    elif embedd == 'ry_embedding':
        return ry_embedding
    elif embedd == 'rz_embedding':
        return rz_embedding
    elif embedd == 'ZZ_embedding':
        return ZZ_embedding
    elif embedd == 'amplitude_embedding':
        return amp_embedding

"""
 Quantum Kernels
"""

def qkernel (embedding, n_qubits):
    """Creates a quantum kernel function for quantum support vector machines.

    Args:
        embedding (str): embedding type ('rx_embedding', 'ry_embedding', 'rz_embedding', 'ZZ_embedding', or 'amplitude_embedding')
        n_qubits (int): number of qubits

    Returns:
        function: quantum kernel function
    """
    jax.config.update("jax_enable_x64", True)
    # embedding_circ = get_embedding(embedding)

    device = qml.device("default.qubit.jax", wires = n_qubits)
    
    @jax.jit
    @qml.qnode(device, interface='jax')
    def kernel_circ(a , b):
        get_embedding(embedding)(a , wires=range(n_qubits))
        qml.adjoint(get_embedding(embedding))(b , wires=range(n_qubits))
        return qml.probs(wires = range(n_qubits))

    def kernel(A, B):
        return np.array([[kernel_circ(a , b)[0] for b in B]for a in A])
    
    return kernel

"""
 Ansatzs
"""

def hardware_efficient_ansatz(theta, wires):
    """Implements a hardware-efficient ansatz circuit.

    Args:
        theta (array[float]): array of parameters for the ansatz circuit
        wires (Sequence[int]): wires that the ansatz circuit acts on

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 3 * N
    
    for i in range(N):
        qml.RX(theta[3 * i], wires=wires[i])
    
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[N-1],wires[0]])
    
    for i in range(N):
        qml.RZ(theta[3 * i + 1], wires=wires[i])
    
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[N-1],wires[0]])

    for i in range(N):
        qml.RX(theta[3 * i + 2], wires=wires[i])
    
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[N-1],wires[0]])

def tree_tensor_ansatz(theta , wires):
    """Implements a tree tensor network ansatz circuit.

    Args:
        theta (array[float]): array of parameters for the ansatz circuit
        wires (Sequence[int]): wires that the ansatz circuit acts on

    Returns:
        None
    """
    n = len(wires)
    dim = int(np.log2(n))
    param_count = 0
    for i in range (dim+1):
        step = 2**i
        for j in np.arange(0 , n , 2*step):
            qml.RY(theta[param_count] , wires = j)
            if(i<dim):
                qml.RY(theta[param_count + 1] , wires = j + step)
                qml.CNOT(wires = [j , j + step])
            param_count += 2

def n_param_tree_tensor(nqubits):
    """Calculates the number of parameters in a tree tensor network ansatz.

    Args:
        nqubits (int): number of qubits

    Returns:
        int: number of parameters in the ansatz
    """
    w_1 = nqubits
    w_0 = 0
    while w_1 != 1:
        w_0 = w_0+w_1
        w_1 = w_1/2
    return w_0+1

def HPzRx(theta, wires):
    """Implements an ansatz circuit composed of Hadamard, CZ, and RX gates.

    Args:
        theta (array[float]): array of parameters for the ansatz circuit
        wires (Sequence[int]): wires that the ansatz circuit acts on

    Returns:
        None
    """
    N=len(wires)

    for i in range(N):
        qml.Hadamard(wires = wires[i])
    
    for i in range(N-1):
        qml.CZ(wires=[wires[i], wires[i+1]])
    qml.CZ(wires=[wires[N-1],wires[0]])
    
    for i in range(N):
        qml.RX(theta[i], wires=wires[i])

def TwoLocal(theta, wires):
    """Implements a two-local ansatz circuit.

    Args:
        theta (array[float]): array of parameters for the ansatz circuit
        wires (Sequence[int]): wires that the ansatz circuit acts on

    Returns:
        None
    """
    N=len(wires)
    for i in range(N):
        qml.RY(theta[i], wires = i)
    for i in range(N - 1):
            qml.CNOT(wires = [i, i + 1])

def get_ansatz(ansatz, n_qubits):
    """Returns the ansatz function and the number of parameters based on the specified ansatz type.

    Args:
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        n_qubits (int): number of qubits

    Returns:
        tuple: ansatz function and the number of parameters
    """
    if ansatz == 'hardware_efficient':
        return hardware_efficient_ansatz, 3 * n_qubits
    if ansatz == 'tree_tensor':
        return tree_tensor_ansatz , 2**(n_qubits+1)-1
    if ansatz == 'HPzRx':
        return HPzRx , n_qubits
    if ansatz == 'two_local':
        return TwoLocal, n_qubits
"""
 Auxiliary Functions
"""    

def create_circuit(n_qubits,layers,embedding,ansatz,n_class, backend='jax'):
    """Creates a quantum circuit for classification tasks.

    Args:
        n_qubits (int): number of qubits
        layers (int): number of layers in the circuit
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        n_class (int): number of classes
        backend (str, optional): quantum backend to use ('jax'). Defaults to 'jax'.

    Raises:
        ValueError: if an unknown backend is provided

    Returns:
        function: compiled quantum circuit
    """
    if backend == 'jax':
        device = qml.device("default.qubit.jax", wires=n_qubits)
    else:
        raise ValueError(f"Backend {backend} is unknown")
    ansatz, params_per_layer = get_ansatz(ansatz,n_qubits)
    
    @qml.qnode(device, interface = 'jax')
    def circuit(x, theta):
        get_embedding(embedding)(x , wires=range(n_qubits))

        for i in range(layers):
            ansatz(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(n_qubits))
        observable=[]
        for n in range(n_class):
            observable.append(qml.expval(qml.PauliZ(wires=n)))
        #verboseprint(observable)
        return observable

    return jax.jit(circuit)

def create_circuit_binary(n_qubits,layers,embedding,ansatz,backend="jax"):
    """Creates a quantum circuit for binary classification tasks.

    Args:
        n_qubits (int): number of qubits
        layers (int): number of layers in the circuit
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        backend (str, optional): quantum backend to use ('jax'). Defaults to 'jax'.

    Raises:
        ValueError: if an unknown backend is provided

    Returns:
        function: compiled quantum circuit
    """
    if backend == 'jax':
        device = qml.device("default.qubit.jax", wires=n_qubits)
    else:
        raise ValueError(f"Backend {backend} is unknown")
    ansatz, params_per_layer = get_ansatz(ansatz,n_qubits)

    state_0=[[1],[0]]
    M=state_0*np.conj(state_0).T
    @qml.qnode(device, interface = 'jax')
    def circuit(x, theta):
        #embedding(x, wires=range(n_qubits))
        get_embedding(embedding)(x , wires=range(n_qubits))
        for i in range(layers):
            #ry_embedding(x, wires=range(n_qubits))
            ansatz(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(n_qubits))
        return qml.expval(qml.Hermitian(M, wires=0))

    return jax.jit(circuit)#

def get_thetas(params):
    """Converts JAX parameters to NumPy.

    Args:
        params (array): JAX parameters

    Returns:
        array: NumPy parameters
    """
    def jnp_to_np(value):
        try:
            value_numpy = np.array(value)
            return value_numpy
        except:
            try:
                value_numpy = np.array(value.primal)
                return value_numpy
            except:
                try:
                    value_numpy = np.array(value.primal.aval)
                    return value_numpy
                except:
                    raise ValueError(f"Cannot convert to numpy value {value}")
    return jnp_to_np(params)
    
    
def evaluate_bagging_predictor(qnn, n_estimators, max_features, max_samples, optimizer, n_qubits, runs, epochs, layers, ansatz, X_train, X_test, y_train, y_test,seed,verboseprint,ignore_warnings=True):
    """Evaluates a bagging predictor composed of a quantum neural network (QNN) ensemble.

    Args:
        qnn (function): quantum neural network function
        n_estimators (int): number of estimators in the bagging predictor
        max_features (float): maximum fraction of features to consider for each estimator
        max_samples (float): maximum fraction of samples to consider for each estimator
        optimizer (object): JAX optimizer
        n_qubits (int): number of qubits
        runs (int): number of runs
        epochs (int): number of training epochs
        layers (int): number of layers in the ansatz
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        X_train (array[float]): training input data
        X_test (array[float]): test input data
        y_train (array[float]): training target data
        y_test (array[float]): test target data
        seed (int): random seed
        ignore_warnings (bool, optional): whether to ignore warnings. Defaults to True.

    Returns:
        tuple: predictions, accuracy score, balanced accuracy score, weighted F1 score, ROC AUC score
    """

    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        
        yp = jax.nn.softmax(yp)
        cost = cross_entropy_loss(y, yp)
        
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy()
    y_test = jnp.argmax(y_test, axis=1)
    
    verboseprint(f"QNN_BAG\t{ansatz}")
    verboseprint('='*50)
    for i in range(runs):
        
        # array to gather estimators' predictions
        predictions = []
        predictions_train = []
        predictions_softmax = []
        predictions_softmax_train = []
                
        for j in range(n_estimators):
            
            # seed
            key = jax.random.PRNGKey(seed)
                
            random_estimator_samples = jax.random.choice(key, a=X_train.shape[0], shape=(int(max_samples*X_train.shape[0]),), p=max_samples*jnp.ones(X_train.shape[0]))
            X_train_est = X_train[random_estimator_samples,:]
            y_train_est = y_train[random_estimator_samples,:]
            random_estimator_features = jax.random.choice(key, a=X_train_est.shape[1], shape=(max(1,int(max_features*X_train_est.shape[1])),), replace=False, p=max_features*jnp.ones(X_train_est.shape[1]))
            X_train_est = X_train_est[:,random_estimator_features]
        
            # get number of circuit params
            _, params_per_layer = get_ansatz(ansatz, n_qubits)
            
            # initialize circuit params
            initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
            params = jnp.copy(initial_params)
    
            # initialize optimizer
            opt_state = optimizer.init(initial_params)
            
            start_time_tr = time.time() 
            ##### fit #####
            for epoch in range(epochs):
                key = jax.random.split(key)[0]
                params, opt_state, cost = optimizer_update(opt_state, params, X_train_est, y_train_est)
                if epoch % 5 == 0:
                    verboseprint(f'epoch: {epoch} - cost: {cost}')
    
            end_time_tr = time.time()-start_time_tr
            verboseprint('Optimization time: ',end_time_tr)
               
            ##### predict #####
            start_time_ts = time.time() 
            y_predict = qnn(X_test[:,random_estimator_features], params)
            y_predict = jax.nn.softmax(y_predict)
            y_predict_softmax = y_predict.copy()
            y_predict_train = qnn(X_train_est, params)
            y_predict_train = jax.nn.softmax(y_predict_train)
            y_predict_softmax_train = y_predict_train.copy()
            verboseprint(f'Error of bagging estimator {j} on test set: {cross_entropy_loss(y_test_ohe,y_predict)}')
            
            y_predict = jnp.argmax(y_predict, axis=1)
            y_predict_train = jnp.argmax(y_predict_train, axis=1)
            y_train_aux = jnp.argmax(y_train_est, axis=1)
            
            
            verboseprint(f'Accuracy of bagging estimator {j} on test set: {accuracy_score(y_test,y_predict)}')
            
            predictions.append(y_predict)
            predictions_train.append(y_predict_train)
            predictions_softmax.append(y_predict_softmax)
            predictions_softmax_train.append(y_predict_softmax_train)
            
        
        # transform list of predictions into an array
        predictions = np.array(predictions)
        predictions_train = np.array(predictions_train)
        predictions_softmax = np.array(predictions_softmax)
        predictions_softmax_train = np.array(predictions_softmax_train)

        ##### predict #####
        start_time_ts = time.time() 
        
        maj_voting = False
        
        if maj_voting:
            # compute mode (majority voting) of estimators' predictions
            y_predict = jnp.mode(predictions).reshape(-1,1)
            y_predict_train = jnp.mode(predictions_train).reshape(-1,1)
            
            end_time_ts = time.time()-start_time_ts
            
            
            verboseprint(f'Accuracy of bagging on test set: {accuracy_score(y_test,y_predict)}\n')
        
        else:
            # compute average of estimators' predictions
            y_predict = jnp.mean(predictions_softmax,axis=0).reshape(-1,3)
            y_predict_train = jnp.mean(predictions_softmax_train,axis=0).reshape(-1,3)
            verboseprint(f'Error of bagging on test set: {cross_entropy_loss(y_test_ohe,y_predict)}\n')
            y_predict = jnp.argmax(y_predict, axis=1)
            
            y_predict_train = jnp.argmax(y_predict_train, axis=1)
            end_time_ts = time.time()-start_time_ts
            verboseprint(f'Accuracy of bagging on test set: {accuracy_score(y_test,y_predict)}\n')
        try:
            roc_auc = roc_auc_score(y_test, y_predict)
        except Exception as exception:
            roc_auc = None
            if ignore_warnings is False:
                verboseprint("ROC AUC couldn't be calculated")
                verboseprint(exception)

        return predictions, accuracy_score(y_test,y_predict), balanced_accuracy_score(y_test, y_predict), f1_score(y_test, y_predict, average="weighted"), roc_auc

def evaluate_bagging_predictor_binary(qnn, n_estimators, max_features, max_samples, optimizer, n_qubits, runs, epochs, layers, ansatz, X_train, X_test, y_train, y_test, seed, verboseprint,ignore_warnings=True):
    """Evaluates a binary classification bagging predictor composed of a quantum neural network (QNN) ensemble.

    Args:
        qnn (function): quantum neural network function
        n_estimators (int): number of estimators in the bagging predictor
        max_features (float): maximum fraction of features to consider for each estimator
        max_samples (float): maximum fraction of samples to consider for each estimator
        optimizer (object): JAX optimizer
        n_qubits (int): number of qubits
        runs (int): number of runs
        epochs (int): number of training epochs
        layers (int): number of layers in the ansatz
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        X_train (array[float]): training input data
        X_test (array[float]): test input data
        y_train (array[float]): training target data
        y_test (array[float]): test target data
        seed (int): random seed
        ignore_warnings (bool, optional): whether to ignore warnings. Defaults to True.

    Returns:
        tuple: predictions, accuracy score, balanced accuracy score, weighted F1 score, ROC AUC score
    """
    
    verboseprint(f"QNN_BAG\t{ansatz}")
    verboseprint('='*50)

    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        y_true=y_true.reshape(-1,1)
        y_pred=y_pred.reshape(-1,1)
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true + (1-y_true)*jnp.log(1-y_pred), axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        #yp = jax.nn.softmax(yp)
        cost = cross_entropy_loss(y, yp)
        
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy() 
    
    for i in range(runs):
        
        # array to gather estimators' predictions
        predictions = []
        predictions_train = []
                
        for j in range(n_estimators):
            
            # seed
            key = jax.random.PRNGKey(seed)
            
            random_estimator_samples = jax.random.choice(key, a=X_train.shape[0], shape=(int(max_samples*X_train.shape[0]),), p=max_samples*jnp.ones(X_train.shape[0]))
            X_train_est = X_train[random_estimator_samples,:]
            y_train_est = y_train[random_estimator_samples]
            random_estimator_features = jax.random.choice(key, a=X_train_est.shape[1], shape=(max(1,int(max_features*X_train_est.shape[1])),), replace=False, p=max_features*jnp.ones(X_train_est.shape[1]))
            X_train_est = X_train_est[:,random_estimator_features]
        
            # get number of circuit params
            _, params_per_layer = get_ansatz(ansatz, n_qubits)
            
            # initialize circuit params
            initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
            params = jnp.copy(initial_params)
    
            # initialize optimizer
            opt_state = optimizer.init(initial_params)
            
            start_time_tr = time.time() 
            ##### fit #####
            for epoch in range(epochs):
                key = jax.random.split(key)[0]
                params, opt_state, cost = optimizer_update(opt_state, params, X_train_est, y_train_est)
            
    
            end_time_tr = time.time()-start_time_tr
            
            ##### predict #####
            start_time_ts = time.time() 
            y_predict = qnn(X_test[:,random_estimator_features], params)
            y_predict_train = qnn(X_train_est, params)
            
            verboseprint(f'Error of bagging estimator {j} on test set: {cross_entropy_loss(y_test_ohe,y_predict)}')
            
            
            end_time_ts = time.time()-start_time_ts
            

            verboseprint(f'Accuracy of bagging estimator {j} on test set: {accuracy_score(y_test,y_predict>=0.5)}')
            verboseprint(f'Accuracy of bagging estimator {j} on train set: {accuracy_score(y_train_est,y_predict_train>=0.5)}')
            predictions.append(y_predict)
            predictions_train.append(y_predict_train)
        
        
        # transform list of predictions into an array
        predictions = np.array(predictions)
        predictions_train = np.array(predictions_train)

        ##### predict #####
        
        # compute average of estimators' predictions
        
        y_predict = jnp.mean(predictions, axis=0)
        y_predict_train = jnp.mean(predictions_train, axis=0)
        verboseprint(f'Error of bagging on test set: {cross_entropy_loss(y_test_ohe,y_predict)}')
        
        verboseprint(f'Accuracy of bagging on test set: {accuracy_score(y_test,y_predict>=0.5)}')        

        y_predict = y_predict>=0.5

        try:
            roc_auc = roc_auc_score(y_test, y_predict)
        except Exception as exception:
            roc_auc = None
            if ignore_warnings is False:
                verboseprint("ROC AUC couldn't be calculated")
                verboseprint(exception)

        return y_predict, accuracy_score(y_test,y_predict), balanced_accuracy_score(y_test, y_predict), f1_score(y_test, y_predict, average="weighted"), roc_auc



def evaluate_full_model_predictor(qnn, optimizer, n_qubits, runs, epochs, layers, ansatz, X_train, X_test, y_train, y_test, seed,verboseprint,ignore_warnings=True):
    """Evaluates a full model predictor composed of a quantum neural network (QNN).

    Args:
        qnn (function): quantum neural network function
        optimizer (object): JAX optimizer
        n_qubits (int): number of qubits
        runs (int): number of runs
        epochs (int): number of training epochs
        layers (int): number of layers in the ansatz
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        X_train (array[float]): training input data
        X_test (array[float]): test input data
        y_train (array[float]): training target data
        y_test (array[float]): test target data
        seed (int): random seed
        ignore_warnings (bool, optional): whether to ignore warnings. Defaults to True.

    Returns:
        tuple: predictions, accuracy score, balanced accuracy score, weighted F1 score, ROC AUC score
    """
    verboseprint(f"QNN\t{ansatz}")
    verboseprint('='*50)

    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        yp = jax.nn.softmax(yp)

        cost = cross_entropy_loss(y, yp)

        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy() 
    y_test = jnp.argmax(y_test, axis=1)
    for i in range(runs):
        # seed
        key = jax.random.PRNGKey(seed)
        
        # get number of circuit params
        _, params_per_layer = get_ansatz(ansatz, n_qubits)
        
        # initialize circuit params
        initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
        params = jnp.copy(initial_params)

        # initialize optimizer
        opt_state = optimizer.init(initial_params)
        
        start_time_tr = time.time() 
        ##### fit #####
        for epoch in range(epochs):
            key = jax.random.split(key)[0]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train, y_train)

        end_time_tr = time.time()-start_time_tr
        
        ##### predict #####
        start_time_ts = time.time() 
        y_predict = qnn(X_test, params)
        y_predict = jax.nn.softmax(y_predict)
        verboseprint(f'cross entropy loss: {cross_entropy_loss(y_test_ohe,y_predict)}')
        y_predict = jnp.argmax(y_predict, axis=1)
        
        end_time_ts = time.time()-start_time_ts
        verboseprint(f'Accuracy of fullmodel on test set: {accuracy_score(y_test,y_predict)}')

        try:
            roc_auc = roc_auc_score(y_test, y_predict)
        except Exception as exception:
            roc_auc = None
            if ignore_warnings is False:
                verboseprint("ROC AUC couldn't be calculated")
                verboseprint(exception)

        return y_predict, accuracy_score(y_test,y_predict), balanced_accuracy_score(y_test, y_predict), f1_score(y_test, y_predict, average="weighted"), roc_auc


def evaluate_full_model_predictor_binary(qnn, optimizer, n_qubits,  epochs, layers, ansatz, X_train, X_test, y_train, y_test, runs, seed, verboseprint, ignore_warnings=True):
    """Evaluates a binary classification full model predictor composed of a quantum neural network (QNN).

    Args:
        qnn (function): quantum neural network function
        optimizer (object): JAX optimizer
        n_qubits (int): number of qubits
        epochs (int): number of training epochs
        layers (int): number of layers in the ansatz
        ansatz (str): ansatz type ('hardware_efficient', 'tree_tensor', 'HPzRx', or 'two_local')
        X_train (array[float]): training input data
        X_test (array[float]): test input data
        y_train (array[float]): training target data
        y_test (array[float]): test target data
        runs (int): number of runs
        seed (int): random seed
        ignore_warnings (bool, optional): whether to ignore warnings. Defaults to True.

    Returns:
        tuple: predictions, accuracy score, balanced accuracy score, weighted F1 score, ROC AUC score
    """

    verboseprint(f"QNN\t{ansatz}")
    verboseprint('='*50)

    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        y_true=y_true.reshape(-1,1)
        y_pred=y_pred.reshape(-1,1)
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true + (1-y_true)*jnp.log(1-y_pred), axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        cost = cross_entropy_loss(y, yp)
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy()

    for i in range(runs):

        # seed
        key = jax.random.PRNGKey(seed)
        
        # get number of circuit params
        _, params_per_layer = get_ansatz(ansatz, n_qubits)
        
        # initialize circuit params
        initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
        params = jnp.copy(initial_params)

        # initialize optimizer
        opt_state = optimizer.init(initial_params)
        
        start_time_tr = time.time() 
        ##### fit #####
        for epoch in range(epochs):
            key = jax.random.split(key)[0]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train, y_train)

        end_time_tr = time.time()-start_time_tr          
        
        ##### predict #####
        start_time_ts = time.time() 
        y_predict = qnn(X_test, params)
        verboseprint(f'Cross entropy loss: {cross_entropy_loss(y_test_ohe,y_predict)}')
        end_time_ts = time.time()-start_time_ts
        
        
        verboseprint(f'Accuracy of fullmodel on test set: {accuracy_score(y_test,y_predict>=0.5)}')

        y_predict = y_predict>=0.5

        try:
            roc_auc = roc_auc_score(y_test, y_predict)
        except Exception as exception:
            roc_auc = None
            if ignore_warnings is False:
                verboseprint("ROC AUC couldn't be calculated")
                verboseprint(exception)

        return y_predict, accuracy_score(y_test,y_predict), balanced_accuracy_score(y_test, y_predict), f1_score(y_test, y_predict, average="weighted"), roc_auc
